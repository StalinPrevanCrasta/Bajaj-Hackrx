"""
Simple device detection for RAG systems
"""
import torch

def main():
    print("🖥️  Device Detection")
    print("=" * 30)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device.upper()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print("✅ GPU acceleration available")
    else:
        print("💻 Using CPU only")
        print("💡 For GPU support:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")

if __name__ == "__main__":
    main()
