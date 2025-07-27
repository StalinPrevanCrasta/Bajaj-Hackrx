import torch
import sys

print("🖥️  System Device Information:")
print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyTorch Built with CUDA: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("❌ Using CPU - No CUDA GPU detected")
    print("Possible reasons:")
    print("1. PyTorch CPU-only version installed")
    print("2. NVIDIA drivers not installed")
    print("3. CUDA toolkit not installed")
    print("4. GPU not compatible")

# Check if NVIDIA GPU exists on system
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("\n🎯 nvidia-smi output:")
        print(result.stdout)
    else:
        print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")
except:
    print("❌ Could not run nvidia-smi command")