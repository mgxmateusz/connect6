import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime bundled with PyTorch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Install a CUDA-enabled PyTorch build.")

print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("Architectures in this PyTorch build:", torch.cuda.get_arch_list())

# Krótki test BF16 dla RTX 50 / Blackwell.
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
c = a @ b
torch.cuda.synchronize()
print("BF16 CUDA smoke test: OK", tuple(c.shape), c.dtype)
