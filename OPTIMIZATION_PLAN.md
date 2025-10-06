# Image Server Optimization Checklist

## Phase 1 — Precision & Device Selection
- [ ] **Config auto-detection**: Add `select_device_dtype()` helper in `ImageServer.py` to resolve device/dtype automatically.
- [ ] **Env overrides**: Support `.env` variables (e.g., `FLUX_DTYPE`, `FLUX_DEVICE`) that feed into `CONFIG` defaults.
- [ ] **Logging**: Emit resolved device/dtype at startup for verification.

## Phase 2 — Generator Reuse & Seeding
- [ ] **Generator cache**: Maintain per-device `torch.Generator` instances in `ImageServer.py` instead of recreating per request.
- [ ] **Seed control**: Re-seed cached generator only when `request['seed']` provided; return seed in response payload.
- [ ] **Determinism test**: Confirm identical outputs given same seed twice in a row.

## Phase 3 — Pipeline Optimizations
- [ ] **Progress bar**: Disable tqdm via `pipe.set_progress_bar_config(disable=True)` after `_build_pipeline()`.
- [ ] **Advanced slicing**: Allow configurable attention/vae slicing chunk sizes via config/env flags.
- [ ] **Offload modes**: Choose between `enable_model_cpu_offload()` and `enable_sequential_cpu_offload()` based on new config option.
- [ ] **README update**: Document new optimization knobs in `README.md`.

## Phase 4 — MPS Error Recovery Enhancements
- [ ] **Cache cleanup**: Call `torch.mps.empty_cache()` before retrying after MPS OOM.
- [ ] **Adaptive fallback**: Optionally switch to `torch.float32` and reduce steps/resolution with detailed `fallback_reason` metadata.
- [ ] **Cross-backend guard**: Detect similar CUDA OOM messages and reuse fallback logic.

## Phase 5 — Concurrency & Locking
- [ ] **Per-model locks**: Replace global `_model_lock` with per-model locks in `ImageServer.py`.
- [ ] **Semaphore**: Introduce configurable `asyncio.Semaphore` to limit concurrent generations.
- [ ] **Load test**: Run concurrency benchmark on macOS (M-series) and CPU-only setups to validate throughput.

## Phase 6 — Configuration Surface
- [ ] **dotenv alignment**: Extend `.env.example` with new flags (dtype/device/slicing/offload/queue/warmup).
- [ ] **Runtime snapshot**: Update `/v1/models` response to expose active config values.
- [ ] **Config helper**: Centralize env parsing to keep `CONFIG` consistent across modules.

## Phase 7 — Warmup Routine
- [ ] **Config flag**: Introduce `CONFIG['WARMUP_ENABLE']` (and env override) for warmup control.
- [ ] **Warmup run**: After pipeline creation/preload, execute low-resolution, single-step dummy generation when enabled.
- [ ] **Warmup logging**: Log duration and outcome per model.

## Phase 8 — Metrics & Logging
- [ ] **Timing metrics**: Wrap generation path with perf counters, log total/request/save durations.
- [ ] **Structured logging**: Add optional JSON logging when `FLUX_STRUCTURED_LOGS=1`.
- [ ] **Metrics endpoint**: Consider lightweight `/metrics` exposing rolling averages or integrate with Prometheus if feasible.
