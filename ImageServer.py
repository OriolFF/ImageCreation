"""
FastAPI server for image generation API.
Provides OpenAI-style endpoints for text-to-image generation.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import base64
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
import time

from image_generation import ImageGenerator, GenerationConfig

app = FastAPI()

# Minimal CORS to allow browser UI from localhost:5500 to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # development: accept any origin
    allow_credentials=False,  # must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables (for HF token and model selection)
load_dotenv()

# Get Hugging Face token
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_ACCESS_TOKEN")
if hf_token:
    suffix = hf_token[-4:]
    print(f"[Auth] Hugging Face token detected (length={len(hf_token)}, suffix=***{suffix})")
else:
    print("[Auth] Hugging Face token missing — set HUGGINGFACE_HUB_TOKEN or HF_ACCESS_TOKEN")

# Helper function to parse boolean env vars
def get_bool_env(key: str, default: bool) -> bool:
    """Parse boolean environment variable (1/0, true/false, yes/no)."""
    value = os.getenv(key, "").lower()
    if value in ("1", "true", "yes", "on"):
        return True
    elif value in ("0", "false", "no", "off"):
        return False
    return default

# Create generation configuration from environment variables
config = GenerationConfig(
    cache_dir=os.getenv("FLUX_CACHE_DIR") or None,
    revision=os.getenv("FLUX_REVISION") or None,
    variant=os.getenv("FLUX_VARIANT") or None,
    dtype=os.getenv("FLUX_DTYPE", "bfloat16"),
    generator_device=os.getenv("FLUX_GENERATOR_DEVICE", "cpu"),
    enable_slicing=get_bool_env("FLUX_ENABLE_SLICING", True),
    enable_vae_tiling=get_bool_env("FLUX_ENABLE_VAE_TILING", True),
    enable_cpu_offload=get_bool_env("FLUX_ENABLE_CPU_OFFLOAD", False),
    preload_models=get_bool_env("FLUX_PRELOAD_MODELS", False),
    warmup_enable=get_bool_env("FLUX_WARMUP_ENABLE", False),
)

print(f"[Config] Loaded configuration: dtype={config.dtype}, generator_device={config.generator_device}, "
      f"slicing={config.enable_slicing}, vae_tiling={config.enable_vae_tiling}, "
      f"cpu_offload={config.enable_cpu_offload}, preload={config.preload_models}, warmup={config.warmup_enable}")

# Initialize image generator
generator = ImageGenerator(config=config, hf_token=hf_token)

# Set initial active model from environment
env_model_key = os.getenv("FLUX_MODEL_KEY", "schnell")
env_model_id = os.getenv("FLUX_MODEL_ID")  # optional direct override
generator.set_active_model(model_key=env_model_key, model_id=env_model_id)

# Optionally preload models
generator.preload_models()

@app.get("/")
def root():
    model_info = generator.get_active_model_info()
    return {
        "status": "ok",
        "message": "Image Creator API",
        "model": {"active_key": model_info.key, "active_id": model_info.id, "active_type": model_info.type},
        "endpoints": [
            {"method": "POST", "path": "/v1/images/generations", "body": {"prompt": "..."}},
            {"method": "GET", "path": "/v1/models"},
            {"method": "POST", "path": "/v1/models/select", "body": {"model": "schnell|dev|qwen|<hf-repo>"}},
            {"method": "GET", "path": "/metrics"},
            {"method": "GET", "path": "/health"}
        ],
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    """Get generation metrics and statistics."""
    return generator.get_metrics()

@app.get("/v1/models")
def list_models():
    model_info = generator.get_active_model_info()
    full_config = generator.get_full_config()
    return {
        "available": generator.AVAILABLE_MODELS,
        "active": {"key": model_info.key, "id": model_info.id, "type": model_info.type},
        "runtime": full_config,
    }

@app.post("/v1/models/select")
async def select_model(request: dict):
    candidate = (request.get("model") or "").strip()
    if not candidate:
        return {"error": "model is required", "available": list(generator.AVAILABLE_MODELS.keys())}
    
    success, message, model_info = await generator.switch_model(candidate)
    
    if not success:
        return {"error": message, "available": list(generator.AVAILABLE_MODELS.keys())}
    
    return {
        "ok": True,
        "active": {"key": model_info.key, "id": model_info.id},
    }

@app.post("/v1/images/generations")
async def generate_image(request: dict):
    prompt = request.get("prompt", "").strip()
    if not prompt:
        return {"error": "prompt is required"}

    store_local = bool(request.get("store_local", True))

    # Optional generation parameters with fast defaults (tuned for M1 32GB)
    height = int(request.get("height", 512))
    width = int(request.get("width", 512))
    num_inference_steps = int(request.get("num_inference_steps", 4))
    guidance_scale = float(request.get("guidance_scale", 3.5))
    max_sequence_length = int(request.get("max_sequence_length", 256))
    seed = request.get("seed")
    if seed is not None:
        seed = int(seed)

    # Generate image using the generator
    img, fallback_applied, original_params, timing_metrics = await generator.generate(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length,
        seed=seed,
    )
    
    # Optionally store locally
    save_start = time.time()
    saved_path = None
    if store_local:
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = os.path.join("outputs", f"image_{timestamp}.png")
        try:
            img.save(saved_path, format="PNG")
        except Exception:
            saved_path = None
    save_time = time.time() - save_start
    
    # Convert to base64
    encode_start = time.time()
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    encode_time = time.time() - encode_start
    
    # Add server-side timing to metrics
    timing_metrics["save_seconds"] = round(save_time, 3)
    timing_metrics["encode_seconds"] = round(encode_time, 3)
    
    # Get current model and device info
    model_info = generator.get_active_model_info()
    device_info = generator.get_device_info()

    return {
        "data": [{
            "b64_json": img_str,
            **({"saved_path": saved_path} if saved_path else {}),
            "model": model_info.id,
            "model_key": model_info.key,
            "device": device_info["device"],
            "dtype": device_info["dtype"],
            "params": {
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "max_sequence_length": max_sequence_length,
                "seed": seed if isinstance(seed, int) else 42,
                "fallback_applied": fallback_applied,
                **({"original_params": original_params} if fallback_applied else {}),
            },
            "timing": timing_metrics,
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
