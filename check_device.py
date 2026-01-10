import sys
import platform
import os

try:
    import torch
except ImportError:
    print("PyTorch is not installed. Assuming CPU only.")
    sys.exit(0)

try:
    import psutil  # optional, for detailed RAM info
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


def bytes_to_gb(x: int) -> float:
    return round(x / (1024 ** 3), 2)


print("=== System info ===")
print("OS:", platform.platform())
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)

cpu_count = os.cpu_count() or 1
print("CPU cores:", cpu_count)

if psutil is not None:
    vm = psutil.virtual_memory()
    print(f"System RAM: {bytes_to_gb(vm.total)} GB total, {bytes_to_gb(vm.available)} GB available")
else:
    print("System RAM: psutil not installed (run `pip install psutil` for detailed RAM info)")

print("\n=== PyTorch backends ===")
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print("MPS available (Apple Silicon GPU):", has_mps)

if cuda_available:
    num_devices = torch.cuda.device_count()
    print(f"\n=== CUDA devices ({num_devices}) ===")
    for idx in range(num_devices):
        props = torch.cuda.get_device_properties(idx)
        name = props.name
        total_mem_gb = bytes_to_gb(props.total_memory)
        cc = getattr(props, "major", None), getattr(props, "minor", None)
        print(f"GPU {idx}: {name}")
        print(f"  VRAM: {total_mem_gb} GB")
        if cc != (None, None):
            print(f"  Compute capability: {cc[0]}.{cc[1]}")

    current = torch.cuda.current_device()
    print("Current CUDA device index:", current)
    print("Current CUDA device name:", torch.cuda.get_device_name(current))

print("\n=== Recommended default device for models ===")
if cuda_available:
    print("Use: device='cuda' (or 'cuda:0')")
elif has_mps:
    print("Use: device='mps' (Apple Silicon GPU)")
else:
    print("Use: device='cpu' (no GPU backend detected)")
