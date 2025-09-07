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

app = FastAPI()

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
    
    # Generar imagen
    images = pipeline(
        prompt,
        num_inference_steps=4,
        generator=torch.Generator("cpu").manual_seed(42)
    ).images
    img = images[0]
    
    # Convertir a base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return {
        "data": [{
            "b64_json": img_str
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
