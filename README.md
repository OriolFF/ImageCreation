# ImageCreatorApi

FastAPI-based image generation server powered by Hugging Face Diffusers with runtime model switching, warmup, and memory management routines. A companion web client in `web/` provides an interactive front end.

The server exposes OpenAI-style endpoints under `/v1/*`, including text-to-image generation, model management, runtime metrics, and cache cleanup. Responses now include timing details and runtime parameters to aid debugging and benchmarking.

## Project Structure

- `ImageServer.py` — FastAPI app (Uvicorn entrypoint) exposing the image generation API and wiring configuration from environment variables.
- `image_generation.py` — Image pipeline management, model switching, warmup, metrics, and memory cleanup logic.
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

On macOS/Linux, `source .venv/bin/activate` activates the environment.

On Windows (PowerShell), activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
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

Copy `.env.example` to `.env` to explore the full set of runtime knobs. Key optional variables include:

- `FLUX_MODEL_KEY` / `FLUX_MODEL_ID` — choose the startup model by registry key (`schnell`, `dev`, `qwen`) or explicit Hugging Face repo id.
- `FLUX_DTYPE`, `FLUX_GENERATOR_DEVICE` — control precision (`bfloat16`, `fp16`, `fp32`) and RNG device (`cpu`, `mps`, `cuda`, `auto`).
- `FLUX_ENABLE_SLICING`, `FLUX_ENABLE_VAE_TILING`, `FLUX_ENABLE_CPU_OFFLOAD` — toggle memory/performance strategies.
- `FLUX_PRELOAD_MODELS`, `FLUX_WARMUP_ENABLE` — preload all pipelines and run a dummy warmup pass on startup.
- `FLUX_CACHE_DIR`, `FLUX_REVISION`, `FLUX_VARIANT` — override Hugging Face cache location, pin revisions, or pick variants.
- `FLUX_STRUCTURED_LOGS`, `FLUX_LOG_LEVEL` — adjust logging verbosity/output format.
- `IMAGE_SERVER_PORT` — override the FastAPI listening port (default `8000`).

## Running the Image Creator Server

Start the FastAPI server (Uvicorn) from the project root:

```bash
python3 ImageServer.py
```

### On Windows (PowerShell)

From the project root, after activating your virtual environment:

```powershell
python .\ImageServer.py
```

- Default address: `http://localhost:8000` (configurable via `IMAGE_SERVER_PORT`)
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

### On Windows (PowerShell)

From the project root:

```powershell
python -m http.server 5500 -d web
```

Or from the `web/` directory:

```powershell
python -m http.server 5500
```

Then open: <http://localhost:5500>

### Node (optional)

```bash
npx serve -l 5500 web
```

If you change the port, the FastAPI CORS settings already allow all origins for development, so no change is needed.

## API Reference

Base URL: `http://localhost:8000`

### `POST /v1/images/generations`

Generate an image from a prompt. Optional parameters fine-tune resolution and quality.

#### Request body

```json
{
  "prompt": "cyberpunk skyline at dusk",          // required
  "store_local": true,                            // optional, default true
  "height": 512, "width": 512,                  // optional, defaults 512
  "num_inference_steps": 4,                       // optional, defaults tuned per model
  "guidance_scale": 3.5,                          // optional, defaults model-specific
  "max_sequence_length": 256,                     // optional token budget
  "seed": 1234                                    // optional deterministic seed
}
```

#### Response body

```json
{
  "data": [
    {
      "b64_json": "<base64-encoded-png>",
      "saved_path": "outputs/image_20240101_120001.png",  // present when store_local is true
      "model": "black-forest-labs/FLUX.1-schnell",
      "model_key": "schnell",
      "device": "mps",
      "dtype": "torch.float16",
      "params": {
        "height": 512,
        "width": 512,
        "num_inference_steps": 4,
        "guidance_scale": 3.5,
        "max_sequence_length": 256,
        "seed": 1234,
        "fallback_applied": false
      },
      "timing": {
        "load_seconds": 0.01,
        "generation_seconds": 2.05,
        "save_seconds": 0.04,
        "encode_seconds": 0.02
      }
    }
  ]
}
```

### `GET /v1/models`

Lists available models, the active selection, and the resolved runtime configuration snapshot. Useful for debugging environment overrides and cache state.

### `POST /v1/models/select`

Switch the active model by registry key or Hugging Face repo id.

```json
{
  "model": "dev"            // e.g. "schnell", "dev", "qwen", or a repo id
}
```

Returns `{ "ok": true, "active": { "key": "dev", "id": "black-forest-labs/FLUX.1-dev" } }` on success.

### `POST /v1/memory/release`

Flushes all cached pipelines, releases GPU/MPS memory, and forces garbage collection. Returns a report including how many pipelines were cleared.

### `GET /metrics`

Expose rolling generation counts, success/failure tallies, and aggregate timing metrics suitable for dashboards or health checks.

### `GET /health`

Simple readiness probe returning `{ "status": "ok" }`.

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

### Built-in registry

- `schnell` → `black-forest-labs/FLUX.1-schnell`
- `dev` → `black-forest-labs/FLUX.1-dev`
- `qwen` → `Qwen/Qwen-Image`

### List available models

```bash
curl http://localhost:8000/v1/models | jq
```

### Select a model by key or repo id

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

### Startup configuration snippet

```bash
# choose a registry key (default: schnell)
export FLUX_MODEL_KEY=schnell   # schnell, dev, or qwen

# or force a specific model repo (overrides key and marks as custom)
export FLUX_MODEL_ID=black-forest-labs/FLUX.1-dev

# optional: enable CPU offload if memory-constrained
export FLUX_ENABLE_CPU_OFFLOAD=1
```

### Available Models

- `schnell`: FLUX.1-schnell (fast inference, tuned defaults for speed)
- `dev`: FLUX.1-dev (higher quality, prioritizes fidelity)
- `qwen`: Qwen-Image (Apache 2.0, excels at typography and multilingual prompts)

### Web UI model selector

The web client includes a model selector in the header. It fetches available models from `GET /v1/models` and switches with `POST /v1/models/select`.

To use the model selector:

1. Start the server: `python3 ImageServer.py`
2. Serve the web client: `python3 -m http.server 5500 -d web`
3. Open <http://localhost:5500> and use the “Model” dropdown to switch between `schnell` and `dev` (or a custom active model).

#### Web UI Runtime Controls

- **Free Memory** button calls `POST /v1/memory/release` to drop cached pipelines when VRAM is low.
- Runtime info banner surfaces active model, device, and dtype from `GET /v1/models`.
- All actions surface toast/status messages to highlight API errors or cache flush results.

## Performance tuning and caching

### Performance environment variables

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

Notes:
- Disabling slicing/tiling may increase peak memory but improve throughput/GPU utilization.
- CPU offload reduces GPU memory usage but can lower GPU utilization and add CPU overhead.
- Preloading eliminates the “first generation after switching is slow” effect.

### UI Presets and Fallback Notices

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

<https://github.com/Xza85hrf/flux_pipeline>

## License

This repository uses models from Hugging Face/Black Forest Labs, which may carry their own licenses and use restrictions. Review and comply with the model's license and terms of use.
