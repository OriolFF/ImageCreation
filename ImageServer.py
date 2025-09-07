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

# Load environment variables (for HF token)
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_ACCESS_TOKEN")

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Image Creator API",
        "endpoints": [
            {"method": "POST", "path": "/v1/images/generations", "body": {"prompt": "..."}},
            {"method": "GET", "path": "/health"}
        ],
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# Cargar modelo al iniciar
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    token=hf_token,
)
pipeline.enable_model_cpu_offload()

@app.post("/v1/images/generations")
async def generate_image(request: dict):
    prompt = request.get("prompt", "")
    store_local = bool(request.get("store_local", True))
    
    # Generar imagen
    images = pipeline(
        prompt,
        num_inference_steps=4,
        generator=torch.Generator("cpu").manual_seed(42)
    ).images
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
    
    return {
        "data": [{
            "b64_json": img_str,
            **({"saved_path": saved_path} if saved_path else {})
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
