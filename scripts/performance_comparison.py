import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_system import PolicyExpertRAG
from src.optimized_rag_system import OptimizedPolicyExpertRAG

def compare_performance():
    """Compare performance between original and optimized systems"""
    
    test_queries = [
        "What is covered under mental illness treatment?",
        "How do I make a cashless claim?",
        "What are the exclusions for dental treatment?",
        "What is the waiting period for pre-existing diseases?",
        "What expenses are covered for air ambulance?"
    ]
    
    print("🔥 RAG System Performance Comparison")
    print("=" * 60)
    
    # Test Original System
    print("\n📊 Testing Original System:")
    print("-" * 40)
    original_init_start = time.time()
    try:
        original_rag = PolicyExpertRAG()
        original_init_time = time.time() - original_init_start
        print(f"✅ Original system initialized in {original_init_time:.2f}s")
        
        original_times = []
        original_success = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Original Query {i}: {query[:50]}...")
            start = time.time()
            try:
                response = original_rag.query(query)
                elapsed = time.time() - start
                original_times.append(elapsed)
                original_success += 1
                print(f"⏱️  Time: {elapsed:.2f}s")
                print(f"✅ Response: {response[:80]}...")
            except Exception as e:
                elapsed = time.time() - start
                print(f"❌ Failed in {elapsed:.2f}s: {str(e)[:50]}...")
    
    except Exception as e:
        print(f"❌ Original system failed to initialize: {e}")
        original_times = []
        original_init_time = 0
        original_success = 0
    
    # Test Optimized System
    print(f"\n{'='*60}")
    print("🚀 Testing Optimized System:")
    print("-" * 40)
    
    optimized_init_start = time.time()
    try:
        optimized_rag = OptimizedPolicyExpertRAG()
        optimized_init_time = time.time() - optimized_init_start
        print(f"✅ Optimized system initialized in {optimized_init_time:.2f}s")
        
        optimized_times = []
        optimized_success = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Optimized Query {i}: {query[:50]}...")
            start = time.time()
            try:
                response = optimized_rag.fast_query(query)
                elapsed = time.time() - start
                optimized_times.append(elapsed)
                optimized_success += 1
                print(f"⏱️  Time: {elapsed:.2f}s")
                print(f"✅ Response: {response[:80]}...")
            except Exception as e:
                elapsed = time.time() - start
                print(f"❌ Failed in {elapsed:.2f}s: {str(e)[:50]}...")
    
    except Exception as e:
        print(f"❌ Optimized system failed to initialize: {e}")
        optimized_times = []
        optimized_init_time = 0
        optimized_success = 0
    
    # Performance Analysis
    print(f"\n{'='*60}")
    print("📈 PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Initialization Comparison
    print(f"\n🚀 Initialization Performance:")
    print(f"   Original System:  {original_init_time:.2f}s")
    print(f"   Optimized System: {optimized_init_time:.2f}s")
    if original_init_time > 0:
        init_improvement = ((original_init_time - optimized_init_time) / original_init_time) * 100
        print(f"   Improvement:      {init_improvement:.1f}% faster")
    
    # Query Performance Comparison
    if original_times and optimized_times:
        print(f"\n⏱️  Query Performance:")
        
        orig_avg = sum(original_times) / len(original_times)
        opt_avg = sum(optimized_times) / len(optimized_times)
        
        print(f"   Original Average:   {orig_avg:.2f}s")
        print(f"   Optimized Average:  {opt_avg:.2f}s")
        
        if orig_avg > 0:
            speed_improvement = ((orig_avg - opt_avg) / orig_avg) * 100
            print(f"   Speed Improvement:  {speed_improvement:.1f}% faster")
        
        print(f"\n📊 Detailed Comparison:")
        print(f"{'Query':<45} {'Original':<10} {'Optimized':<10} {'Improvement'}")
        print("-" * 80)
        
        for i, (orig_time, opt_time) in enumerate(zip(original_times, optimized_times)):
            query_short = test_queries[i][:40] + "..." if len(test_queries[i]) > 40 else test_queries[i]
            improvement = ((orig_time - opt_time) / orig_time * 100) if orig_time > 0 else 0
            print(f"{query_short:<45} {orig_time:<10.2f} {opt_time:<10.2f} {improvement:>8.1f}%")
    
    # Success Rate
    print(f"\n🎯 Success Rate:")
    print(f"   Original System:  {original_success}/{len(test_queries)} queries")
    print(f"   Optimized System: {optimized_success}/{len(test_queries)} queries")
    
    # Overall Rating
    if original_times and optimized_times:
        orig_avg = sum(original_times) / len(original_times)
        opt_avg = sum(optimized_times) / len(optimized_times)
        
        if opt_avg < 2:
            rating = "🟢 Excellent Performance"
        elif opt_avg < 4:
            rating = "🟡 Good Performance"
        elif opt_avg < 8:
            rating = "🟠 Average Performance"
        else:
            rating = "🔴 Needs Further Optimization"
        
        print(f"\n🏆 Overall Performance Rating: {rating}")
        
        # Throughput comparison
        orig_throughput = 60 / orig_avg if orig_avg > 0 else 0
        opt_throughput = 60 / opt_avg if opt_avg > 0 else 0
        
        print(f"\n🔄 Throughput (queries per minute):")
        print(f"   Original:  {orig_throughput:.1f} queries/min")
        print(f"   Optimized: {opt_throughput:.1f} queries/min")

def stress_test_optimized():
    """Stress test the optimized system"""
    print(f"\n{'='*60}")
    print("🔥 STRESS TEST - Optimized System")
    print("=" * 60)
    
    optimized_rag = OptimizedPolicyExpertRAG()
    
    # Test with repeated queries to check caching
    test_query = "What is covered under mental illness treatment?"
    
    print(f"\n🔄 Testing caching with repeated queries...")
    print(f"Query: {test_query}")
    
    cache_times = []
    for i in range(5):
        start = time.time()
        response = optimized_rag.fast_query(test_query)
        elapsed = time.time() - start
        cache_times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    print(f"\n📊 Caching Performance:")
    print(f"   First run (no cache): {cache_times[0]:.2f}s")
    print(f"   Cached runs average:  {sum(cache_times[1:]) / len(cache_times[1:]):.2f}s")
    print(f"   Cache speedup:        {(cache_times[0] / cache_times[1]):.1f}x faster")

if __name__ == "__main__":
    print("🚀 RAG System Performance Comparison Tool")
    print("=" * 60)
    
    choice = input("""
Choose test option:
1. Full performance comparison (recommended)
2. Stress test optimized system only
3. Both tests

Enter choice (1-3): """).strip()
    
    if choice == "1":
        compare_performance()
    elif choice == "2":
        stress_test_optimized()
    elif choice == "3":
        compare_performance()
        stress_test_optimized()
    else:
        print("Invalid choice. Running full comparison...")
        compare_performance()