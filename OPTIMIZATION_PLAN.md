# Image Server Optimization Checklist

## 📁 **CODE STRUCTURE**
The codebase is now split into two modules:
- **`ImageServer.py`** - FastAPI server logic, routes, and HTTP handling
- **`image_generation.py`** - Image generation logic, pipeline management, and model switching

**Note**: Most optimizations below apply to `image_generation.py` unless explicitly stated otherwise.

## ⚠️ **SAFETY NOTICE**
This plan contains **compatibility risks**. Test each phase on a separate branch before merging to main.

## Phase 1 — Precision & Device Selection ⚠️ **HIGH RISK**
**CRITICAL**: `torch.bfloat16` is NOT supported on MPS (Apple Silicon). Current code works due to fallback logic.

- [ ] **Config auto-detection**: Add `select_device_dtype()` helper in `image_generation.py` (`ImageGenerator.__init__`) to resolve device/dtype automatically.
  - ⚠️ **Risk**: Must handle MPS `bfloat16` → `float16` conversion properly
  - 🛡️ **Mitigation**: Test on M4 Max before deploying, ensure fallback logic preserved
  - 📍 **Location**: `image_generation.py` - `ImageGenerator.__init__()` method
- [ ] **Env overrides**: Support `.env` variables (e.g., `FLUX_DTYPE`, `FLUX_DEVICE`) that feed into `GenerationConfig` defaults in `ImageServer.py`.
  - 📍 **Location**: `ImageServer.py` - config initialization section
- [ ] **Logging**: Emit resolved device/dtype at startup for verification (already implemented in `ImageGenerator.__init__`).
- [ ] **Testing**: Verify on MPS, CUDA, and CPU backends before proceeding

## Phase 2 — Generator Reuse & Seeding ⚠️ **MEDIUM RISK**
**WARNING**: Generator device mismatches cause `"Expected 'mps:0' generator but found 'cpu'"` crashes.

- [ ] **Generator cache**: Maintain per-device `torch.Generator` instances in `image_generation.py` (`ImageGenerator` class) instead of recreating per request.
  - ⚠️ **Risk**: Device mismatch between generator and tensors will crash
  - 🛡️ **Mitigation**: Ensure generator device matches pipeline device, add device validation
  - 📍 **Location**: `image_generation.py` - Add `_generator_cache` dict to `ImageGenerator` class
- [ ] **Seed control**: Re-seed cached generator only when `request['seed']` provided; return seed in response payload.
  - 📍 **Location**: `image_generation.py` - `ImageGenerator.generate()` method
- [ ] **Determinism test**: Confirm identical outputs given same seed twice in a row.
- [ ] **Device validation**: Add checks to ensure generator device matches pipeline device
  - 📍 **Location**: `image_generation.py` - `ImageGenerator.generate()` method

## Phase 3 — Pipeline Optimizations ⚠️ **MEDIUM RISK**
**WARNING**: CPU offload changes can break performance or cause crashes with PyTorch compile.

- [ ] **Progress bar**: Disable tqdm via `pipe.set_progress_bar_config(disable=True)` after `_build_pipeline()`.
  - ✅ **Safe**: Low risk optimization
  - 📍 **Location**: `image_generation.py` - `ImageGenerator._build_pipeline()` method
- [ ] **Advanced slicing**: Allow configurable attention/vae slicing chunk sizes via config/env flags.
  - ✅ **Safe**: Existing feature, just making it configurable
  - 📍 **Location**: `image_generation.py` - Add to `GenerationConfig` dataclass and `_build_pipeline()` method
- [ ] **Offload modes**: Choose between `enable_model_cpu_offload()` and `enable_sequential_cpu_offload()` based on new config option.
  - ⚠️ **Risk**: Sequential offload is 3x slower, model offload conflicts with PyTorch compile
  - 🛡️ **Mitigation**: Keep current offload as default, make new modes opt-in only
  - 📍 **Location**: `image_generation.py` - Add to `GenerationConfig` and `_build_pipeline()` method
- [ ] **Performance testing**: Benchmark before/after on M4 Max to ensure no regression
- [ ] **README update**: Document new optimization knobs in `README.md`.

## Phase 4 — MPS Error Recovery Enhancements ⚠️ **LOW RISK**
**NOTE**: MPS memory management is still buggy, these help but don't guarantee fixes.

- [ ] **Cache cleanup**: Call `torch.mps.empty_cache()` before retrying after MPS OOM (already implemented in `_cleanup_pipeline()`).
  - ⚠️ **Risk**: May not reliably prevent OOM, MPS cache behavior is unpredictable
  - 🛡️ **Mitigation**: Keep existing fallback logic as primary recovery method
  - 📍 **Location**: `image_generation.py` - `ImageGenerator.generate()` method (OOM handling)
- [ ] **Adaptive fallback**: Optionally switch to `torch.float32` and reduce steps/resolution with detailed `fallback_reason` metadata.
  - ✅ **Safe**: Additive improvement to existing fallback
  - 📍 **Location**: `image_generation.py` - `ImageGenerator.generate()` method
- [ ] **Cross-backend guard**: Detect similar CUDA OOM messages and reuse fallback logic.
  - ✅ **Safe**: Extends existing logic to other backends
  - 📍 **Location**: `image_generation.py` - `ImageGenerator.generate()` method

## Phase 5 — Concurrency & Locking ✅ **LOW RISK**
- [ ] **Per-model locks**: Replace global `_model_lock` with per-model locks in `image_generation.py`.
  - ⚠️ **Risk**: Potential deadlocks if not implemented carefully
  - 🛡️ **Mitigation**: Use timeout-based locks, thorough testing
  - 📍 **Location**: `image_generation.py` - `ImageGenerator` class (replace `_model_lock` with dict of locks)
- [ ] **Semaphore**: Introduce configurable `asyncio.Semaphore` to limit concurrent generations.
  - ✅ **Safe**: Standard concurrency control
  - 📍 **Location**: `image_generation.py` - Add to `ImageGenerator.__init__()` and `generate()` method
- [ ] **Load test**: Run concurrency benchmark on macOS (M-series) and CPU-only setups to validate throughput.

## Phase 6 — Configuration Surface ✅ **SAFE**
- [x] **dotenv alignment**: Extend `.env.example` with new flags (dtype/device/slicing/offload/queue/warmup).
  - 📍 **Location**: `.env.example` file
- [x] **Runtime snapshot**: Update `/v1/models` response to expose active config values (partially implemented via `get_device_info()`).
  - 📍 **Location**: `ImageServer.py` - `/v1/models` endpoint; `image_generation.py` - expand `get_device_info()` method
- [x] **Config helper**: Centralize env parsing in `ImageServer.py` to keep `GenerationConfig` consistent.
  - 📍 **Location**: `ImageServer.py` - config initialization section

## Phase 7 — Warmup Routine ✅ **SAFE**
- [x] **Config flag**: Introduce `GenerationConfig.warmup_enable` (and env override in `ImageServer.py`) for warmup control.
  - 📍 **Location**: `image_generation.py` - Add to `GenerationConfig` dataclass; `ImageServer.py` - config init
- [x] **Warmup run**: After pipeline creation/preload, execute low-resolution, single-step dummy generation when enabled.
  - 📍 **Location**: `image_generation.py` - Add `_warmup_pipeline()` method, call from `_build_pipeline()` or `preload_models()`
- [x] **Warmup logging**: Log duration and outcome per model.
  - 📍 **Location**: `image_generation.py` - `_warmup_pipeline()` method

## Phase 8 — Metrics & Logging ✅ **SAFE**
- [x] **Timing metrics**: Wrap generation path with perf counters, log total/request/save durations.
  - 📍 **Location**: `image_generation.py` - `ImageGenerator.generate()` method; `ImageServer.py` - `/v1/images/generations` endpoint
- [x] **Structured logging**: Add optional JSON logging when `FLUX_STRUCTURED_LOGS=1`.
  - 📍 **Location**: Both `image_generation.py` and `ImageServer.py` - replace print statements with structured logger
- [x] **Metrics endpoint**: Consider lightweight `/metrics` exposing rolling averages or integrate with Prometheus if feasible.
  - 📍 **Location**: `ImageServer.py` - new `/metrics` endpoint; `image_generation.py` - add metrics collection to `ImageGenerator`
  - ℹ️ Added `/v1/memory/release` endpoint and UI control to complement cache management.

---

## 🚀 **RECOMMENDED IMPLEMENTATION ORDER**

### **Start Here (Safest)**
1. **Phase 6** - Configuration Surface
2. **Phase 8** - Metrics & Logging  
3. **Phase 7** - Warmup Routine

### **Then (With Caution)**
4. **Phase 4** - MPS Error Recovery (test thoroughly)
5. **Phase 5** - Concurrency & Locking (test with load)

### **Finally (High Risk - Separate Branch)**
6. **Phase 1** - Precision & Device Selection (⚠️ **TEST ON M4 MAX FIRST**)
7. **Phase 3** - Pipeline Optimizations (benchmark performance)
8. **Phase 2** - Generator Reuse (validate device matching)

### **Testing Strategy**
- Create feature branch for each phase
- Test on M4 Max before merging
- Keep rollback plan ready
- Monitor performance metrics after each phase

---

## ✅ **COMPLETED IMPROVEMENTS**

### **Code Refactoring (Completed)**
- ✅ **Module separation**: Split monolithic `ImageServer.py` into:
  - `ImageServer.py` (180 lines) - FastAPI routes and HTTP handling
  - `image_generation.py` (400+ lines) - Image generation logic and pipeline management
- ✅ **Memory cleanup**: Implemented proper pipeline cleanup with `_cleanup_pipeline()` method
  - Moves pipelines to CPU before deletion
  - Deletes pipeline components (transformer, VAE, text encoders)
  - Forces garbage collection
  - Clears device cache (CUDA/MPS)
- ✅ **Model switching**: Enhanced model switching to clean up old pipelines before loading new ones
  - Prevents memory accumulation when switching models
  - Properly releases GPU/MPS memory
- ✅ **Class-based architecture**: `ImageGenerator` class encapsulates all generation logic
  - `GenerationConfig` dataclass for configuration
  - `ModelInfo` dataclass for model information
  - Clean separation of concerns
- ✅ **Async support**: All generation and model switching operations are async-compatible
