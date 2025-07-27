"""
Test GPU/CPU detection and performance for RAG systems
"""
import sys
import os
import time
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_device_availability():
    """Check GPU/CPU availability and specs"""
    print("🖥️  Device Detection Test")
    print("=" * 50)
    
    # PyTorch CUDA detection
    print(f"🔍 PyTorch Information:")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   CUDA Version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"\n🎯 GPU Information:")
        for i in range(torch.cuda.device_count()):
            print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"   Memory: {props.total_memory / 1024**3:.1f}GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
        
        # Test GPU memory
        print(f"\n💾 GPU Memory Status:")
        print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
        print(f"   Cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f}MB")
        print(f"   Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.1f}GB")
        
    else:
        print(f"\n💻 CPU-Only Mode:")
        print(f"   For GPU support, install CUDA-enabled PyTorch:")
        print(f"   pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    return torch.cuda.is_available()

def test_embedding_performance():
    """Test embedding generation performance"""
    print(f"\n⚡ Performance Test")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing embeddings on: {device.upper()}")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Initialize embeddings
        print("🔄 Loading embedding model...")
        start_time = time.time()
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
            model_kwargs={'device': device}
        )
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Test embedding generation
        test_texts = [
            "What is covered under mental illness treatment?",
            "How do I make a cashless claim?",
            "What are the exclusions for dental treatment?",
            "What is the waiting period for pre-existing diseases?",
            "What expenses are covered for air ambulance?"
        ]
        
        print(f"\n📊 Testing {len(test_texts)} embeddings:")
        
        # Single embedding test
        start_time = time.time()
        single_embedding = embeddings.embed_query(test_texts[0])
        single_time = time.time() - start_time
        print(f"   Single embedding: {single_time:.3f}s")
        print(f"   Embedding dimension: {len(single_embedding)}")
        
        # Batch embedding test
        start_time = time.time()
        batch_embeddings = embeddings.embed_documents(test_texts)
        batch_time = time.time() - start_time
        avg_time = batch_time / len(test_texts)
        print(f"   Batch embeddings: {batch_time:.3f}s ({avg_time:.3f}s per text)")
        
        # Performance analysis
        speedup = single_time / avg_time if avg_time > 0 else 1
        print(f"   Batch speedup: {speedup:.1f}x faster")
        
        if torch.cuda.is_available():
            print(f"\n💾 GPU Memory After Embeddings:")
            print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
            print(f"   Cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f}MB")
        
        return avg_time
        
    except ImportError as e:
        print(f"❌ Required packages not installed: {e}")
        print("Install with: pip install langchain-huggingface sentence-transformers")
        return None
    except Exception as e:
        print(f"❌ Error testing embeddings: {e}")
        return None

def test_rag_systems():
    """Test both RAG systems with device detection"""
    print(f"\n🚀 RAG System Device Test")
    print("=" * 50)
    
    test_query = "What is mental illness coverage?"
    
    # Test Original RAG System
    try:
        print(f"\n📊 Original RAG System:")
        from src.rag_system import PolicyExpertRAG
        
        start_time = time.time()
        original_rag = PolicyExpertRAG()
        init_time = time.time() - start_time
        print(f"   Initialization: {init_time:.2f}s")
        
        start_time = time.time()
        response = original_rag.query(test_query)
        query_time = time.time() - start_time
        print(f"   Query time: {query_time:.2f}s")
        print(f"   Response length: {len(response)} characters")
        
    except Exception as e:
        print(f"❌ Original RAG error: {e}")
    
    # Test Optimized RAG System
    try:
        print(f"\n🚀 Optimized RAG System:")
        from src.optimized_rag_system import OptimizedPolicyExpertRAG
        
        start_time = time.time()
        optimized_rag = OptimizedPolicyExpertRAG()
        opt_init_time = time.time() - start_time
        print(f"   Initialization: {opt_init_time:.2f}s")
        
        start_time = time.time()
        opt_response = optimized_rag.fast_query(test_query)
        opt_query_time = time.time() - start_time
        print(f"   Query time: {opt_query_time:.2f}s")
        
        # Handle different response formats
        if isinstance(opt_response, dict):
            response_text = opt_response.get('answer', str(opt_response))
        else:
            response_text = str(opt_response)
        
        print(f"   Response length: {len(response_text)} characters")
        
    except Exception as e:
        print(f"❌ Optimized RAG error: {e}")

def main():
    """Main test function"""
    print("🔥 Device Detection and Performance Test")
    print("=" * 60)
    
    # Check device availability
    gpu_available = check_device_availability()
    
    # Test embedding performance
    embedding_time = test_embedding_performance()
    
    # Test RAG systems
    test_rag_systems()
    
    # Summary
    print(f"\n🏆 Summary")
    print("=" * 50)
    device_type = "GPU" if gpu_available else "CPU"
    print(f"🖥️  Device: {device_type}")
    
    if embedding_time:
        print(f"⚡ Embedding Performance: {embedding_time:.3f}s per text")
        if gpu_available:
            print("✅ GPU acceleration active - optimal performance")
        else:
            print("⚠️  CPU-only mode - consider GPU for 2-4x speedup")
    
    if not gpu_available:
        print(f"\n💡 To enable GPU acceleration:")
        print(f"   1. Install CUDA-enabled PyTorch:")
        print(f"      pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print(f"   2. Ensure NVIDIA GPU drivers are installed")
        print(f"   3. Restart your terminal/IDE")

if __name__ == "__main__":
    main()
