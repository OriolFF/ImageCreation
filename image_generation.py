"""
Image generation module for managing diffusion pipelines and image generation.
Handles model loading, caching, cleanup, and generation logic.
"""

from diffusers import FluxPipeline, DiffusionPipeline
import torch
import gc
import asyncio
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for image generation pipeline."""
    cache_dir: Optional[str] = None
    revision: Optional[str] = None
    variant: Optional[str] = None
    dtype: str = "bfloat16"
    generator_device: str = "cpu"
    enable_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_cpu_offload: bool = False
    preload_models: bool = False


@dataclass
class ModelInfo:
    """Information about a model."""
    id: str
    type: str
    key: str


class ImageGenerator:
    """Manages image generation pipelines and model switching."""
    
    # Model registry
    AVAILABLE_MODELS = {
        "schnell": {"id": "black-forest-labs/FLUX.1-schnell", "type": "flux"},
        "dev": {"id": "black-forest-labs/FLUX.1-dev", "type": "flux"},
        "qwen": {"id": "Qwen/Qwen-Image", "type": "diffusion"},
    }
    
    def __init__(self, config: GenerationConfig, hf_token: Optional[str] = None):
        """
        Initialize the image generator.
        
        Args:
            config: Generation configuration
            hf_token: Hugging Face access token
        """
        self.config = config
        self.hf_token = hf_token
        
        # Setup device and dtype
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() 
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        if config.dtype.lower() == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
        
        # Pipeline cache and lock
        self._pipelines_cache: Dict[str, Any] = {}
        self._model_lock = asyncio.Lock()
        
        # Active model tracking
        self.active_model_id: Optional[str] = None
        self.active_model_type: Optional[str] = None
        self.active_model_key: Optional[str] = None
        
        print(f"[ImageGenerator] Initialized with device={self.device}, dtype={self.dtype}")
    
    def set_active_model(self, model_key: Optional[str] = None, model_id: Optional[str] = None):
        """
        Set the active model by key or explicit ID.
        
        Args:
            model_key: Model key from AVAILABLE_MODELS (e.g., "schnell", "dev")
            model_id: Explicit Hugging Face model ID (overrides model_key)
        """
        if model_id:
            self.active_model_id = model_id
            self.active_model_type = "flux"  # default to flux for custom models
            self.active_model_key = next(
                (k for k, v in self.AVAILABLE_MODELS.items() if v["id"] == model_id), 
                None
            ) or "custom"
        else:
            model_key = model_key or "schnell"
            model_config = self.AVAILABLE_MODELS.get(model_key, self.AVAILABLE_MODELS["schnell"])
            self.active_model_id = model_config["id"]
            self.active_model_type = model_config["type"]
            self.active_model_key = model_key if model_key in self.AVAILABLE_MODELS else "schnell"
        
        print(f"[ImageGenerator] Active model set: key={self.active_model_key}, id={self.active_model_id}")
    
    def get_active_model_info(self) -> ModelInfo:
        """Get information about the currently active model."""
        return ModelInfo(
            id=self.active_model_id,
            type=self.active_model_type,
            key=self.active_model_key
        )
    
    def _cleanup_pipeline(self, pipe):
        """Properly clean up a pipeline and release memory."""
        if pipe is None:
            return
        
        try:
            # Move pipeline to CPU first to free device memory
            pipe.to("cpu")
        except Exception as e:
            print(f"[Cleanup] Warning: failed to move pipeline to CPU: {e}")
        
        try:
            # Delete pipeline components if accessible
            if hasattr(pipe, 'transformer'):
                del pipe.transformer
            if hasattr(pipe, 'vae'):
                del pipe.vae
            if hasattr(pipe, 'text_encoder'):
                del pipe.text_encoder
            if hasattr(pipe, 'text_encoder_2'):
                del pipe.text_encoder_2
        except Exception as e:
            print(f"[Cleanup] Warning: failed to delete pipeline components: {e}")
        
        # Delete the pipeline itself
        del pipe
        
        # Force garbage collection
        gc.collect()
        
        # Clear device cache based on device type
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            try:
                torch.mps.empty_cache()
                torch.mps.synchronize()
            except Exception as e:
                print(f"[Cleanup] Warning: MPS cache clear failed: {e}")
        
        print("[Cleanup] Pipeline cleanup complete")
    
    def _build_pipeline(self, model_id: str, model_type: str = "flux"):
        """Build and configure a diffusion pipeline."""
        kwargs = {
            "torch_dtype": self.dtype,
            "token": self.hf_token,
        }
        if self.config.cache_dir:
            kwargs["cache_dir"] = self.config.cache_dir
        if self.config.revision:
            kwargs["revision"] = self.config.revision
        if self.config.variant:
            kwargs["variant"] = self.config.variant
        
        print(
            f"[Model] Loading pipeline: id={model_id}, type={model_type}, device={self.device}, "
            f"dtype={self.dtype}, cache_dir={self.config.cache_dir or 'default'}, "
            f"revision={self.config.revision or 'latest'}, variant={self.config.variant or 'default'}"
        )
        
        # Choose the appropriate pipeline class based on model type
        if model_type == "flux":
            pipe = FluxPipeline.from_pretrained(model_id, **kwargs)
        else:
            kwargs.setdefault("trust_remote_code", True)
            pipe = DiffusionPipeline.from_pretrained(model_id, **kwargs)
        
        try:
            pipe.to(self.device)
        except Exception:
            pass
        
        # Apply optimizations (only if supported by the pipeline)
        if self.config.enable_slicing and hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
        if self.config.enable_vae_tiling and hasattr(pipe, 'enable_vae_tiling'):
            pipe.enable_vae_tiling()
        if self.config.enable_cpu_offload and hasattr(pipe, 'enable_model_cpu_offload'):
            pipe.enable_model_cpu_offload()
        
        return pipe
    
    def preload_models(self):
        """Preload all available models to avoid first-request latency."""
        if not self.config.preload_models:
            return
        
        try:
            for k, model_config in self.AVAILABLE_MODELS.items():
                mid = model_config["id"]
                mtype = model_config["type"]
                if mid not in self._pipelines_cache:
                    print(f"[Model] Preloading model: key={k}, id={mid}, type={mtype}")
                    self._pipelines_cache[mid] = self._build_pipeline(mid, mtype)
            print("[Model] Preload complete")
        except Exception as e:
            print(f"[Model] Preload failed: {e}")
    
    async def switch_model(self, model: str) -> Tuple[bool, str, ModelInfo]:
        """
        Switch to a different model.
        
        Args:
            model: Model key or explicit Hugging Face repo ID
            
        Returns:
            Tuple of (success, message, model_info)
        """
        if not model:
            return False, "model is required", None
        
        # Determine new model parameters
        if model in self.AVAILABLE_MODELS:
            model_config = self.AVAILABLE_MODELS[model]
            new_model_id = model_config["id"]
            new_model_type = model_config["type"]
            new_model_key = model
        else:
            new_model_id = model
            new_model_type = "flux"  # default to flux for custom models
            new_model_key = next(
                (k for k, v in self.AVAILABLE_MODELS.items() if v["id"] == model), 
                None
            ) or "custom"
        
        # Atomically switch: clean up old pipelines, load new one, and set active
        async with self._model_lock:
            # Clean up all cached pipelines except the one we're switching to
            print(f"[Model] Cleaning up old pipelines before switching to {new_model_id}")
            for cached_id, cached_pipe in list(self._pipelines_cache.items()):
                if cached_id != new_model_id:
                    print(f"[Model] Removing cached pipeline: {cached_id}")
                    self._cleanup_pipeline(cached_pipe)
            
            # Clear cache of old pipelines
            self._pipelines_cache.clear()
            
            # Load the new pipeline
            if new_model_id not in self._pipelines_cache:
                self._pipelines_cache[new_model_id] = self._build_pipeline(new_model_id, new_model_type)
            
            self.active_model_id = new_model_id
            self.active_model_key = new_model_key
            self.active_model_type = new_model_type
        
        print(
            f"[Model] Switched active model: key={self.active_model_key}, id={self.active_model_id}, "
            f"type={self.active_model_type}, device={self.device}, dtype={self.dtype}"
        )
        
        return True, "Model switched successfully", self.get_active_model_info()
    
    @staticmethod
    def _round_to_multiple(x: int, multiple: int = 64) -> int:
        """Round a number to the nearest multiple."""
        if x <= 0:
            return multiple
        return (x // multiple) * multiple
    
    def _try_generate(
        self, 
        pipe, 
        *, 
        prompt: str, 
        height: int, 
        width: int, 
        steps: int,
        guidance: float, 
        max_seq_len: int, 
        generator: torch.Generator, 
        model_type: str = "flux"
    ):
        """Execute image generation with the given parameters."""
        # Build kwargs based on what the pipeline supports
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "generator": generator,
        }
        
        # FLUX models support these parameters
        if model_type == "flux":
            kwargs["height"] = height
            kwargs["width"] = width
            kwargs["guidance_scale"] = guidance
            kwargs["max_sequence_length"] = max_seq_len
        # Qwen-Image uses true_cfg_scale instead of guidance_scale
        elif model_type == "diffusion" and "Qwen" in str(pipe.__class__.__name__):
            kwargs["height"] = height
            kwargs["width"] = width
            kwargs["true_cfg_scale"] = guidance  # Qwen uses true_cfg_scale
            kwargs["negative_prompt"] = " "  # Required for CFG to work
        # Generic diffusion models typically support height/width
        else:
            kwargs["height"] = height
            kwargs["width"] = width
            # Only add guidance_scale if the pipeline supports it
            if hasattr(pipe, 'guidance_scale') or 'guidance_scale' in str(
                pipe.__class__.__init__.__code__.co_varnames
            ):
                kwargs["guidance_scale"] = guidance
        
        return pipe(**kwargs).images
    
    async def generate(
        self,
        prompt: str,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
        guidance_scale: float = 3.5,
        max_sequence_length: int = 256,
        seed: Optional[int] = None,
    ) -> Tuple[Any, bool, Dict[str, int], Dict[str, float]]:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            height: Image height in pixels
            width: Image width in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            max_sequence_length: Maximum sequence length for text encoding
            seed: Random seed (None for random)
            
        Returns:
            Tuple of (image, fallback_applied, original_params, timing_metrics)
        """
        start_time = time.time()
        
        # Build a device-appropriate generator
        gen_device_pref = self.config.generator_device.lower()
        if gen_device_pref == "cpu":
            gen_device = "cpu"
        elif gen_device_pref in ("mps", "cuda"):
            gen_device = gen_device_pref
        else:
            gen_device = self.device.type if self.device.type in ("cuda", "mps") else "cpu"
        
        generator = torch.Generator(gen_device)
        if isinstance(seed, int):
            generator = generator.manual_seed(seed)
        else:
            generator = generator.manual_seed(42)
        
        # Generate image using the active pipeline (snapshot under lock)
        async with self._model_lock:
            current_model_id = self.active_model_id
            current_model_type = self.active_model_type
            pipe = self._pipelines_cache.get(current_model_id)
            if pipe is None:
                pipe = self._build_pipeline(current_model_id, current_model_type)
                self._pipelines_cache[current_model_id] = pipe
            model_key_snapshot = self.active_model_key
        
        pipeline_load_time = time.time() - start_time
        print(f"[Gen] Using model: key={model_key_snapshot}, id={current_model_id}, type={current_model_type}")
        
        fallback_applied = False
        original_params = {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
        }
        
        # Start generation timing
        generation_start = time.time()
        try:
            images = self._try_generate(
                pipe,
                prompt=prompt,
                height=height,
                width=width,
                steps=num_inference_steps,
                guidance=guidance_scale,
                max_seq_len=max_sequence_length,
                generator=generator,
                model_type=current_model_type,
            )
        except RuntimeError as e:
            msg = str(e)
            # Handle MPS OOM by retrying at reduced resolution and enabling cpu offload if available
            if self.device.type == "mps" and "MPS backend out of memory" in msg:
                try:
                    # Reduce resolution by half (rounded to multiple of 64)
                    fallback_h = max(256, self._round_to_multiple(height // 2))
                    fallback_w = max(256, self._round_to_multiple(width // 2))
                    # Opportunistically enable cpu offload just for this pipe
                    try:
                        pipe.enable_model_cpu_offload()
                    except Exception:
                        pass
                    images = self._try_generate(
                        pipe,
                        prompt=prompt,
                        height=fallback_h,
                        width=fallback_w,
                        steps=max(8, num_inference_steps - 4),
                        guidance=guidance_scale,
                        max_seq_len=max_sequence_length,
                        generator=generator,
                        model_type=current_model_type,
                    )
                    # Update params to reflect fallback
                    height, width, num_inference_steps = fallback_h, fallback_w, max(8, num_inference_steps - 4)
                    fallback_applied = True
                except Exception as e2:
                    raise e2
            else:
                raise
        
        generation_time = time.time() - generation_start
        
        img = images[0]
        
        # MPS may require sync before returning to avoid unexpected stalls in high-throughput usage
        if self.device.type == "mps":
            try:
                torch.mps.synchronize()
            except Exception:
                pass
        
        # Calculate total time and build metrics
        total_time = time.time() - start_time
        timing_metrics = {
            "total_seconds": round(total_time, 3),
            "pipeline_load_seconds": round(pipeline_load_time, 3),
            "generation_seconds": round(generation_time, 3),
        }
        
        print(f"[Timing] Total: {timing_metrics['total_seconds']}s, "
              f"Pipeline: {timing_metrics['pipeline_load_seconds']}s, "
              f"Generation: {timing_metrics['generation_seconds']}s")
        
        return img, fallback_applied, original_params, timing_metrics
    
    def get_device_info(self) -> Dict[str, str]:
        """Get information about the current device configuration."""
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "cache_dir": self.config.cache_dir,
            "revision": self.config.revision,
            "variant": self.config.variant,
        }
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get complete configuration snapshot including runtime settings."""
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "config": {
                "cache_dir": self.config.cache_dir,
                "revision": self.config.revision,
                "variant": self.config.variant,
                "dtype_config": self.config.dtype,
                "generator_device": self.config.generator_device,
                "enable_slicing": self.config.enable_slicing,
                "enable_vae_tiling": self.config.enable_vae_tiling,
                "enable_cpu_offload": self.config.enable_cpu_offload,
                "preload_models": self.config.preload_models,
            },
            "cache": {
                "loaded_models": list(self._pipelines_cache.keys()),
                "num_cached_pipelines": len(self._pipelines_cache),
            }
        }
