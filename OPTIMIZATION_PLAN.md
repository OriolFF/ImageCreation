# Image Server Optimization Checklist

## ⚠️ **SAFETY NOTICE**
This plan contains **compatibility risks**. Test each phase on a separate branch before merging to main.

## Phase 1 — Precision & Device Selection ⚠️ **HIGH RISK**
**CRITICAL**: `torch.bfloat16` is NOT supported on MPS (Apple Silicon). Current code works due to fallback logic.

- [ ] **Config auto-detection**: Add `select_device_dtype()` helper in `ImageServer.py` to resolve device/dtype automatically.
  - ⚠️ **Risk**: Must handle MPS `bfloat16` → `float16` conversion properly
  - 🛡️ **Mitigation**: Test on M4 Max before deploying, ensure fallback logic preserved
- [ ] **Env overrides**: Support `.env` variables (e.g., `FLUX_DTYPE`, `FLUX_DEVICE`) that feed into `CONFIG` defaults.
- [ ] **Logging**: Emit resolved device/dtype at startup for verification.
- [ ] **Testing**: Verify on MPS, CUDA, and CPU backends before proceeding

## Phase 2 — Generator Reuse & Seeding ⚠️ **MEDIUM RISK**
**WARNING**: Generator device mismatches cause `"Expected 'mps:0' generator but found 'cpu'"` crashes.

- [ ] **Generator cache**: Maintain per-device `torch.Generator` instances in `ImageServer.py` instead of recreating per request.
  - ⚠️ **Risk**: Device mismatch between generator and tensors will crash
  - 🛡️ **Mitigation**: Ensure generator device matches pipeline device, add device validation
- [ ] **Seed control**: Re-seed cached generator only when `request['seed']` provided; return seed in response payload.
- [ ] **Determinism test**: Confirm identical outputs given same seed twice in a row.
- [ ] **Device validation**: Add checks to ensure generator device matches pipeline device

## Phase 3 — Pipeline Optimizations ⚠️ **MEDIUM RISK**
**WARNING**: CPU offload changes can break performance or cause crashes with PyTorch compile.

- [ ] **Progress bar**: Disable tqdm via `pipe.set_progress_bar_config(disable=True)` after `_build_pipeline()`.
  - ✅ **Safe**: Low risk optimization
- [ ] **Advanced slicing**: Allow configurable attention/vae slicing chunk sizes via config/env flags.
  - ✅ **Safe**: Existing feature, just making it configurable
- [ ] **Offload modes**: Choose between `enable_model_cpu_offload()` and `enable_sequential_cpu_offload()` based on new config option.
  - ⚠️ **Risk**: Sequential offload is 3x slower, model offload conflicts with PyTorch compile
  - 🛡️ **Mitigation**: Keep current offload as default, make new modes opt-in only
- [ ] **Performance testing**: Benchmark before/after on M4 Max to ensure no regression
- [ ] **README update**: Document new optimization knobs in `README.md`.

## Phase 4 — MPS Error Recovery Enhancements ⚠️ **LOW RISK**
**NOTE**: MPS memory management is still buggy, these help but don't guarantee fixes.

- [ ] **Cache cleanup**: Call `torch.mps.empty_cache()` before retrying after MPS OOM.
  - ⚠️ **Risk**: May not reliably prevent OOM, MPS cache behavior is unpredictable
  - 🛡️ **Mitigation**: Keep existing fallback logic as primary recovery method
- [ ] **Adaptive fallback**: Optionally switch to `torch.float32` and reduce steps/resolution with detailed `fallback_reason` metadata.
  - ✅ **Safe**: Additive improvement to existing fallback
- [ ] **Cross-backend guard**: Detect similar CUDA OOM messages and reuse fallback logic.
  - ✅ **Safe**: Extends existing logic to other backends

## Phase 5 — Concurrency & Locking ✅ **LOW RISK**
- [ ] **Per-model locks**: Replace global `_model_lock` with per-model locks in `ImageServer.py`.
  - ⚠️ **Risk**: Potential deadlocks if not implemented carefully
  - 🛡️ **Mitigation**: Use timeout-based locks, thorough testing
- [ ] **Semaphore**: Introduce configurable `asyncio.Semaphore` to limit concurrent generations.
  - ✅ **Safe**: Standard concurrency control
- [ ] **Load test**: Run concurrency benchmark on macOS (M-series) and CPU-only setups to validate throughput.

## Phase 6 — Configuration Surface ✅ **SAFE**
- [ ] **dotenv alignment**: Extend `.env.example` with new flags (dtype/device/slicing/offload/queue/warmup).
- [ ] **Runtime snapshot**: Update `/v1/models` response to expose active config values.
- [ ] **Config helper**: Centralize env parsing to keep `CONFIG` consistent across modules.

## Phase 7 — Warmup Routine ✅ **SAFE**
- [ ] **Config flag**: Introduce `CONFIG['WARMUP_ENABLE']` (and env override) for warmup control.
- [ ] **Warmup run**: After pipeline creation/preload, execute low-resolution, single-step dummy generation when enabled.
- [ ] **Warmup logging**: Log duration and outcome per model.

## Phase 8 — Metrics & Logging ✅ **SAFE**
- [ ] **Timing metrics**: Wrap generation path with perf counters, log total/request/save durations.
- [ ] **Structured logging**: Add optional JSON logging when `FLUX_STRUCTURED_LOGS=1`.
- [ ] **Metrics endpoint**: Consider lightweight `/metrics` exposing rolling averages or integrate with Prometheus if feasible.

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
