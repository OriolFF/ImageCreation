# ImageCreatorApi

FastAPI-based image generation server powered by Hugging Face Diffusers (`black-forest-labs/FLUX.1-schnell`) plus a simple web client in `web/`.

The server exposes an OpenAI-style endpoint at `/v1/images/generations` that accepts a text prompt and returns a base64-encoded PNG image. The web client provides a minimal UI to try it out.

## Project Structure

- `ImageServer.py` — FastAPI app (Uvicorn entrypoint) exposing the image generation API.
- `web/` — Static web client (HTML/CSS/JS) that talks to the API at `http://localhost:8000`.
- `requirements.txt` — Python dependencies.
- `.gitignore` — Git ignore rules.

## Prerequisites

- Python 3.10+ recommended
- A Hugging Face access token with permissions to pull the model
  - Set either `HUGGINGFACE_HUB_TOKEN` or `HF_ACCESS_TOKEN`
- Sufficient disk space and RAM (model weights will download on first run)
- Optional but recommended: a virtual environment

## Setup

You can create a virtual environment with either `uv` or built-in `venv`.

### Option A: Using uv

```bash
# From project root
uv venv  # creates .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Option B: Using Python venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

Set your Hugging Face token so the model can be downloaded the first time:

```bash
export HUGGINGFACE_HUB_TOKEN=hf_xxx   # or HF_ACCESS_TOKEN
```

You can also put this in a `.env` file in the project root; `ImageServer.py` loads it with `python-dotenv`.

## Running the Image Creator Server

Start the FastAPI server (Uvicorn) from the project root:

```bash
python3 ImageServer.py
```

- Default address: `http://localhost:8000`
- Health check: `GET /health` → `{ "status": "ok" }`
- Root: `GET /` → basic info

Notes:
- On first run, model weights will be downloaded; this can take several minutes.
- The app enables CORS for all origins in development.

## Running the Web Client

The web client is static files under `web/`. It expects the API to be at `http://localhost:8000` (see `web/main.js`, constant `API_URL`). Serve it with any static server; for example:

### Python built-in HTTP server (recommended)

From the project root:
```bash
python3 -m http.server 5500 -d web
```

Or from the `web/` directory:
```bash
python3 -m http.server 5500
```

Then open: http://localhost:5500

### Node (optional)

```bash
npx serve -l 5500 web
```

If you change the port, the FastAPI CORS settings already allow all origins for development, so no change is needed.

## API Reference

Base URL: `http://localhost:8000`

- `POST /v1/images/generations`
  - Request JSON body:
    - `prompt` (string, required): text prompt for the image
    - `store_local` (boolean, optional, default `true`): if true, server also saves the image under `outputs/`
  - Response JSON:
    ```json
    {
      "data": [
        {
          "b64_json": "<base64-encoded-png>",
          "saved_path": "outputs/image_YYYYMMDD_hhmmss.png"  // present only if saved
        }
      ]
    }
    ```

- `GET /health` → `{ "status": "ok" }`

## Usage from Other Projects

Below are examples of calling the API from different environments.

### cURL

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a watercolor painting of a mountain village at sunrise", "store_local": true}' \
  http://localhost:8000/v1/images/generations | \
  jq -r '.data[0].b64_json' > image.b64

# Decode to PNG
base64 --decode image.b64 > output.png
```

### Python (requests)

```python
import base64
import json
import requests

url = "http://localhost:8000/v1/images/generations"
payload = {"prompt": "a cinematic photo of a red vintage car", "store_local": True}

resp = requests.post(url, json=payload)
resp.raise_for_status()
obj = resp.json()

b64 = obj["data"][0]["b64_json"]
with open("output.png", "wb") as f:
    f.write(base64.b64decode(b64))

print("Saved output.png")
```

### JavaScript (browser fetch)

```html
<script>
  async function generate() {
    const res = await fetch("http://localhost:8000/v1/images/generations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: "a cozy cabin in the woods, pixel art", store_local: true })
    });
    const json = await res.json();
    const b64 = json?.data?.[0]?.b64_json;
    const img = document.createElement("img");
    img.src = `data:image/png;base64,${b64}`;
    document.body.appendChild(img);
  }
  generate();
</script>
```

### Node.js (fetch)

```js
import fs from 'node:fs/promises';

const res = await fetch('http://localhost:8000/v1/images/generations', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ prompt: 'an ancient library, isometric perspective', store_local: false })
});
const json = await res.json();
const b64 = json.data[0].b64_json;
await fs.writeFile('output.png', Buffer.from(b64, 'base64'));
console.log('Saved output.png');
```

### Kotlin (JVM, OkHttp)

Add the dependency (Gradle Kotlin DSL shown):

```kotlin
dependencies {
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
}
```

Example usage:

```kotlin
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.util.Base64
import java.nio.file.Files
import java.nio.file.Paths

fun main() {
    val client = OkHttpClient()

    val json = """
        {"prompt": "a serene lake at dusk, watercolor", "store_local": true}
    """.trimIndent()

    val request = Request.Builder()
        .url("http://localhost:8000/v1/images/generations")
        .post(json.toRequestBody("application/json".toMediaType()))
        .build()

    client.newCall(request).execute().use { response ->
        if (!response.isSuccessful) error("HTTP ${'$'}{response.code}")
        val body = response.body?.string() ?: error("Empty response body")

        // Very small JSON extraction; for production use a JSON library like kotlinx.serialization or Jackson
        val regex = Regex("\"b64_json\"\s*:\s*\"([^\"]+)\"")
        val match = regex.find(body) ?: error("b64_json not found in response")
        val b64 = match.groupValues[1]

        val bytes = Base64.getDecoder().decode(b64)
        val out = Paths.get("output.png")
        Files.write(out, bytes)
        println("Saved ${'$'}out")
    }
}
```

## Model management

The server exposes endpoints to list and switch models at runtime. A small registry is built-in and you can also select a specific Hugging Face repo directly.

- Built-in registry (`AVAILABLE_MODELS`):
  - `schnell` → `black-forest-labs/FLUX.1-schnell`
  - `dev` → `black-forest-labs/FLUX.1-dev`

Endpoints:

- List models
```bash
curl http://localhost:8000/v1/models | jq
```

- Select model by key or by repo id
```bash
# by key
curl -X POST -H "Content-Type: application/json" \
  -d '{"model":"dev"}' \
  http://localhost:8000/v1/models/select

# by explicit repo id
curl -X POST -H "Content-Type: application/json" \
  -d '{"model":"black-forest-labs/FLUX.1-schnell"}' \
  http://localhost:8000/v1/models/select
```

Startup configuration (environment variables):

```bash
# choose a registry key (default: schnell)
export FLUX_MODEL_KEY=schnell   # or dev

# or force a specific model repo (overrides key and marks as custom)
export FLUX_MODEL_ID=black-forest-labs/FLUX.1-dev

# optional: enable CPU offload if memory-constrained
export FLUX_ENABLE_CPU_OFFLOAD=1
```

### Web UI model selector

The web client includes a model selector in the header. It fetches available models from `GET /v1/models` and switches with `POST /v1/models/select`.

Steps:
- Start the server: `python3 ImageServer.py`
- Serve the web client: `python3 -m http.server 5500 -d web`
- Open `http://localhost:5500` and use the “Model” dropdown to switch between `schnell` and `dev` (or a custom active model).

## Performance tuning and caching

### Environment variables

These env vars control performance/memory trade-offs and caching behavior:

```bash
# Prefer a stable cache location
export HF_CACHE_DIR="$HOME/.cache/huggingface"

# Pin a model snapshot (commit/tag) to avoid fetching newer snapshots unexpectedly
export FLUX_REVISION=main   # or a specific commit hash/tag

# Optional model variant (e.g., fp16) if provided by the repo
export FLUX_VARIANT=fp16

# Optional: offline-only mode (requires that the model is already in the cache)
export HF_HUB_OFFLINE=1

# Performance knobs (defaults shown)
export FLUX_DISABLE_SLICING=0        # set to 1 to disable attention slicing
export FLUX_DISABLE_VAE_TILING=0     # set to 1 to disable VAE tiling
export FLUX_ENABLE_CPU_OFFLOAD=0     # set to 1 to offload parts to CPU (more stable, lower GPU util)

# Optional: preload all AVAILABLE_MODELS at startup to avoid first-request latency
export FLUX_PRELOAD_MODELS=0         # set to 1 to preload
```

Notes:
- Disabling slicing/tiling may increase peak memory but improve throughput/GPU utilization.
- CPU offload reduces GPU memory usage but can lower GPU utilization and add CPU overhead.
- Preloading eliminates the “first generation after switching is slow” effect.

### UI presets and fallback notices

The web client includes a Preset selector:
- Low: 512×512, 4 steps, guidance=3.0 (fastest)
- Medium: 512×512, 12 steps, guidance=3.0 (balanced)
- High: 640×640, 16 steps, guidance=3.5 (quality; may OOM on dev)

If the server applies a memory fallback (e.g., on MPS OOM), the UI shows a notice detailing how the requested parameters were reduced. The API response includes `params.fallback_applied` and `params.original_params` for introspection.

## Troubleshooting

- Model download/auth errors (401/403): ensure `HUGGINGFACE_HUB_TOKEN` or `HF_ACCESS_TOKEN` is set and valid.
- First run is slow: weights download and load into memory; subsequent runs are faster.
- Out-of-memory: generation uses substantial RAM/VRAM; server is configured with `enable_model_cpu_offload()` to reduce GPU memory use.
- Port already in use: change ports, e.g. `python3 ImageServer.py` still uses 8000; for the web client use `python3 -m http.server 5501 -d web`.
- CORS: development CORS is permissive (`*`). For production, restrict `allow_origins` appropriately in `ImageServer.py`.

## External references

https://github.com/Xza85hrf/flux_pipeline

## License

This repository uses models from Hugging Face/Black Forest Labs, which may carry their own licenses and use restrictions. Review and comply with the model's license and terms of use.
