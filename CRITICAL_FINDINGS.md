
# Critical Findings: Image Generation Precision

## The Issue: Black Images
- **Symptom**: Generated images are full black and small file size (~800 bytes).
- **Cause**: "Not a Number" (NaN) overflows in the VAE (Visual AutoEncoder) when running in `fp16` (Standard Float16) precision on Apple Silicon (MPS).
- **Why**: `fp16` has a limited dynamic range. Intermediate values during image decoding exceed this range, turning into NaNs.

## The Fixes
1. **Use `bfloat16`**: `bfloat16` (Brain Float 16) has the same dynamic range as `float32` (8 exponent bits) but lower precision. This prevents the range overflow. It is supported on RTX 30+ (Ampere/Ada/Blackwell) and Apple Silicon.
2. **Force VAE to `float32`**: The code now includes a safety check. If `device="mps"`, it forces the VAE component to `float32` regardless of the main model dtype. This allows using `fp16` for the UNet (speed) while keeping the VAE safe (correctness).

## Recommended Configuration (Cross-Platform)

### Windows (RTX 5080)
- **Env**: `FLUX_DTYPE=bfloat16`
- **Device**: `FLUX_GENERATOR_DEVICE=cuda`
- **Notes**: RTX 5080 has native hardware acceleration for `bfloat16`. This is the optimal setting.

### macOS (Apple Silicon)
- **Env**: `FLUX_DTYPE=bfloat16`
- **Device**: `FLUX_GENERATOR_DEVICE=cpu` (for RNG) / `mps` (auto-detected for model)
- **Notes**: `bfloat16` is safe. `fp16` is also safe ONLY because we patched the code to force VAE to `float32`.

## Do Not Regress
- **Never** remove the `if self.device.type == "mps" ... pipe.vae.to(dtype=torch.float32)` block in `image_generation.py`.
- **Default** to `bfloat16` in configuration files.
