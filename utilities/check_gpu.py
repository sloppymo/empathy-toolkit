import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capabilities: {torch.cuda.get_device_capability(0)}")
    
    # Test CUDA with a simple operation
    print("\nTesting CUDA with a simple tensor operation...")
    x = torch.rand(5, 3).cuda()
    y = torch.rand(5, 3).cuda()
    z = x + y
    print(f"Operation successful, result device: {z.device}")
else:
    print("\nCUDA is not available. Checking for potential issues:")
    
    # Check if NVIDIA GPU is detected by the system
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("\nNVIDIA GPU detected by system but not by PyTorch:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            print("\nThis suggests PyTorch was installed without CUDA support or there's a version mismatch.")
        else:
            print("\nNVIDIA-SMI command failed. GPU drivers may not be installed properly.")
    except Exception as e:
        print(f"\nError checking GPU with nvidia-smi: {e}")
    
    # Check PyTorch build information
    print("\nPyTorch build information:")
    print(f"CUDA built: {torch.version.cuda}")
    print(f"Debug build: {torch.version.debug}")
    
    # Suggest solutions
    print("\nPossible solutions:")
    print("1. Reinstall PyTorch with CUDA support:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("2. Update NVIDIA drivers")
    print("3. Check for environment conflicts (e.g., multiple PyTorch installations)")
