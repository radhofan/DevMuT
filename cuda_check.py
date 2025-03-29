import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Try a simple CUDA operation
    x = torch.rand(5, 5).cuda()
    print(f"Tensor on CUDA: {x.device}")
else:
    print("CUDA is not available. Using CPU only.")
    
# Test a simple operation
print("Running a test computation...")
a = torch.ones(3, 3)
b = torch.ones(3, 3)
if torch.cuda.is_available():
    a = a.cuda()
    b = b.cuda()
c = a + b
print(f"1 + 1 = {c[0][0].item()}")
print(f"Computation device: {c.device}")