# server.py
from fastapi import FastAPI
from diffusers import FluxPipeline
import torch
import base64
from io import BytesIO
from PIL import Image
import uvicorn
import os
from dotenv import load_dotenv
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

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
# Keep token from env for security; other knobs are configured below.
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_ACCESS_TOKEN")

# Centralized, code-based configuration (no env vars needed)
CONFIG = {
    # Snapshot/cache controls (leave None to use defaults)
    "CACHE_DIR": None,           # e.g., os.path.expanduser("~/.cache/huggingface")
    "REVISION": None,            # e.g., "main" or a specific commit/tag
    "VARIANT": None,             # e.g., "fp16"

    # Precision and RNG device
    "DTYPE": "bfloat16",        # "bfloat16" or "fp16"
    "GENERATOR_DEVICE": "cpu",  # "cpu" | "auto" | "mps" | "cuda"

    # Performance/memory trade-offs
    "ENABLE_SLICING": True,
    "ENABLE_VAE_TILING": True,
    # Match previous server default: enable CPU offload by default
    "ENABLE_CPU_OFFLOAD": True,

    # Preload all AVAILABLE_MODELS at startup
    "PRELOAD_MODELS": True,
}

# Unpack CONFIG for convenience
cache_dir = CONFIG["CACHE_DIR"]
model_revision = CONFIG["REVISION"]
model_variant = CONFIG["VARIANT"]
dtype_pref = CONFIG["DTYPE"].lower()
generator_device_pref = CONFIG["GENERATOR_DEVICE"].lower()
enable_slicing = CONFIG["ENABLE_SLICING"]
enable_vae_tiling = CONFIG["ENABLE_VAE_TILING"]
enable_cpu_offload = CONFIG["ENABLE_CPU_OFFLOAD"]
preload_models = CONFIG["PRELOAD_MODELS"]

# Centralized model registry. Keys are friendly names users can switch at runtime.
AVAILABLE_MODELS = {
    "schnell": "black-forest-labs/FLUX.1-schnell",
    "dev": "black-forest-labs/FLUX.1-dev",
}

# Select initial active model by key or explicit HF repo id via env.
env_model_key = os.getenv("FLUX_MODEL_KEY", "schnell")
env_model_id = os.getenv("FLUX_MODEL_ID")  # optional direct override
if env_model_id:
    active_model_id = env_model_id
    active_model_key = next((k for k, v in AVAILABLE_MODELS.items() if v == env_model_id), None) or "custom"
else:
    active_model_id = AVAILABLE_MODELS.get(env_model_key, AVAILABLE_MODELS["schnell"])
    active_model_key = env_model_key if env_model_key in AVAILABLE_MODELS else "schnell"

# Choose device and dtype
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
if dtype_pref == "bfloat16":
    dtype = torch.bfloat16
else:
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Image Creator API",
        "model": {"active_key": active_model_key, "active_id": active_model_id},
        "endpoints": [
            {"method": "POST", "path": "/v1/images/generations", "body": {"prompt": "..."}},
            {"method": "GET", "path": "/v1/models"},
            {"method": "POST", "path": "/v1/models/select", "body": {"model": "schnell|dev|<hf-repo>"}},
            {"method": "GET", "path": "/health"}
        ],
    }

@app.get("/health")
def health():
    return {"status": "ok"}

_pipelines_cache = {}
_model_lock = asyncio.Lock()

def _build_pipeline(model_id: str) -> FluxPipeline:
    kwargs = {
        "torch_dtype": dtype,
        "token": hf_token,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if model_revision:
        kwargs["revision"] = model_revision
    if model_variant:
        kwargs["variant"] = model_variant
    print(
        f"[Model] Loading pipeline: id={model_id}, device={device}, dtype={dtype}, "
        f"cache_dir={cache_dir or 'default'}, revision={model_revision or 'latest'}, variant={model_variant or 'default'}"
    )
    pipe = FluxPipeline.from_pretrained(model_id, **kwargs)
    try:
        pipe.to(device)
    except Exception:
        pass
    if enable_slicing:
        pipe.enable_attention_slicing()
    if enable_vae_tiling:
        pipe.enable_vae_tiling()
    if enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    return pipe

def get_active_pipeline() -> FluxPipeline:
    pipe = _pipelines_cache.get(active_model_id)
    if pipe is None:
        pipe = _build_pipeline(active_model_id)
        _pipelines_cache[active_model_id] = pipe
    return pipe

# Optionally preload all available models at startup to avoid first-request latency
if preload_models:
    try:
        for k, mid in AVAILABLE_MODELS.items():
            if mid not in _pipelines_cache:
                print(f"[Model] Preloading model: key={k}, id={mid}")
                _pipelines_cache[mid] = _build_pipeline(mid)
        print("[Model] Preload complete")
    except Exception as e:
        print(f"[Model] Preload failed: {e}")

def _round_to_multiple(x: int, multiple: int = 64) -> int:
    if x <= 0:
        return multiple
    return (x // multiple) * multiple

def _try_generate(pipe: FluxPipeline, *, prompt: str, height: int, width: int, steps: int,
                  guidance: float, max_seq_len: int, generator: torch.Generator):
    return pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        max_sequence_length=max_seq_len,
        generator=generator,
    ).images

@app.get("/v1/models")
def list_models():
    return {
        "available": AVAILABLE_MODELS,
        "active": {"key": active_model_key, "id": active_model_id},
        "device": str(device),
        "dtype": str(dtype),
        "cache_dir": cache_dir,
        "revision": model_revision,
        "variant": model_variant,
    }

@app.post("/v1/models/select")
async def select_model(request: dict):
    global active_model_id, active_model_key
    candidate = (request.get("model") or "").strip()
    if not candidate:
        return {"error": "model is required", "available": list(AVAILABLE_MODELS.keys())}

    if candidate in AVAILABLE_MODELS:
        new_model_id = AVAILABLE_MODELS[candidate]
        new_model_key = candidate
    else:
        new_model_id = candidate
        new_model_key = next((k for k, v in AVAILABLE_MODELS.items() if v == candidate), None) or "custom"

    # Atomically switch: ensure pipeline exists and then set active
    async with _model_lock:
        if new_model_id not in _pipelines_cache:
            _pipelines_cache[new_model_id] = _build_pipeline(new_model_id)
        active_model_id = new_model_id
        active_model_key = new_model_key

    print(
        f"[Model] Switched active model: key={active_model_key}, id={active_model_id}, "
        f"device={device}, dtype={dtype}"
    )

    return {
        "ok": True,
        "active": {"key": active_model_key, "id": active_model_id},
    }

@app.post("/v1/images/generations")
async def generate_image(request: dict):
    prompt = request.get("prompt", "").strip()
    if not prompt:
        return {"error": "prompt is required"}

    store_local = bool(request.get("store_local", True))

    # Optional generation parameters with fast defaults (tuned for M1 32GB)
    # Defaults chosen for responsiveness; increase via request body if desired
    height = int(request.get("height", 512))
    width = int(request.get("width", 512))
    num_inference_steps = int(request.get("num_inference_steps", 4))
    guidance_scale = float(request.get("guidance_scale", 3.5))
    max_sequence_length = int(request.get("max_sequence_length", 256))
    seed = request.get("seed")

    # Build a device-appropriate generator (configurable)
    if generator_device_pref == "cpu":
        gen_device = "cpu"
    elif generator_device_pref in ("mps", "cuda"):
        gen_device = generator_device_pref
    else:
        gen_device = device.type if device.type in ("cuda", "mps") else "cpu"
    generator = torch.Generator(gen_device)
    if isinstance(seed, int):
        generator = generator.manual_seed(seed)
    else:
        generator = generator.manual_seed(42)

    # Generate image using the active pipeline (snapshot under lock)
    async with _model_lock:
        current_model_id = active_model_id
        pipe = _pipelines_cache.get(current_model_id)
        if pipe is None:
            pipe = _build_pipeline(current_model_id)
            _pipelines_cache[current_model_id] = pipe
        model_key_snapshot = active_model_key
    print(f"[Gen] Using model: key={model_key_snapshot}, id={current_model_id}")
    fallback_applied = False
    original_params = {
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
    }
    try:
        images = _try_generate(
            pipe,
            prompt=prompt,
            height=height,
            width=width,
            steps=num_inference_steps,
            guidance=guidance_scale,
            max_seq_len=max_sequence_length,
            generator=generator,
        )
    except RuntimeError as e:
        msg = str(e)
        # Handle MPS OOM by retrying at reduced resolution and enabling cpu offload if available
        if device.type == "mps" and "MPS backend out of memory" in msg:
            try:
                # Reduce resolution by half (rounded to multiple of 64)
                fallback_h = max(256, _round_to_multiple(height // 2))
                fallback_w = max(256, _round_to_multiple(width // 2))
                # Opportunistically enable cpu offload just for this pipe
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pass
                images = _try_generate(
                    pipe,
                    prompt=prompt,
                    height=fallback_h,
                    width=fallback_w,
                    steps=max(8, num_inference_steps - 4),
                    guidance=guidance_scale,
                    max_seq_len=max_sequence_length,
                    generator=generator,
                )
                # Informative note in response params below
                height, width, num_inference_steps = fallback_h, fallback_w, max(8, num_inference_steps - 4)
                fallback_applied = True
            except Exception as e2:
                raise e2
        else:
            raise
    img = images[0]
    
    # Optionally store locally
    saved_path = None
    if store_local:
        os.makedirs("outputs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = os.path.join("outputs", f"image_{timestamp}.png")
        try:
            img.save(saved_path, format="PNG")
        except Exception:
            saved_path = None
    
    # Convertir a base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    # MPS may require sync before returning to avoid unexpected stalls in high-throughput usage
    if device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass

    return {
        "data": [{
            "b64_json": img_str,
            **({"saved_path": saved_path} if saved_path else {}),
            "model": active_model_id,
            "model_key": active_model_key,
            "device": str(device),
            "dtype": str(dtype),
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
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
