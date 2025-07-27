import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_system import PolicyExpertRAG

def benchmark_rag_system():
    """Benchmark the current RAG system performance"""
    
    test_queries = [
        "What is covered under mental illness treatment?",
        "How do I make a cashless claim?",
        "What are the exclusions for dental treatment?",
        "What is the waiting period for pre-existing diseases?",
        "What expenses are covered for air ambulance?",
        "What is the sum insured limit?",
        "How do I file a reimbursement claim?",
        "What documents are required for claims?"
    ]
    
    print("🔥 Benchmarking RAG System Performance")
    print("=" * 50)
    
    # Initialize system
    print("🚀 Initializing RAG System...")
    init_start = time.time()
    rag_system = PolicyExpertRAG()
    init_time = time.time() - init_start
    print(f"✅ System initialized in {init_time:.2f}s")
    
    print(f"\n📊 Testing {len(test_queries)} queries...")
    print("-" * 50)
    
    query_times = []
    successful_queries = 0
    failed_queries = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}/{len(test_queries)}: {query}")
        
        start_time = time.time()
        try:
            response = rag_system.query(query)
            elapsed_time = time.time() - start_time
            query_times.append(elapsed_time)
            successful_queries += 1
            
            # Show first 100 characters of response
            response_preview = response[:100] + "..." if len(response) > 100 else response
            print(f"⏱️  Time: {elapsed_time:.2f}s")
            print(f"✅ Response: {response_preview}")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            failed_queries += 1
            print(f"❌ Failed in {elapsed_time:.2f}s - Error: {str(e)}")
    
    # Calculate statistics
    print(f"\n📈 Performance Summary:")
    print("=" * 50)
    print(f"🎯 Successful Queries: {successful_queries}/{len(test_queries)}")
    print(f"❌ Failed Queries: {failed_queries}")
    print(f"🚀 System Initialization: {init_time:.2f}s")
    
    if query_times:
        avg_time = sum(query_times) / len(query_times)
        min_time = min(query_times)
        max_time = max(query_times)
        total_time = sum(query_times)
        
        print(f"\n⏱️  Query Performance:")
        print(f"   • Average Query Time: {avg_time:.2f}s")
        print(f"   • Fastest Query: {min_time:.2f}s")
        print(f"   • Slowest Query: {max_time:.2f}s")
        print(f"   • Total Query Time: {total_time:.2f}s")
        print(f"   • Queries per Minute: {60/avg_time:.1f}")
        
        # Performance rating
        if avg_time < 3:
            rating = "🟢 Excellent"
        elif avg_time < 5:
            rating = "🟡 Good"
        elif avg_time < 10:
            rating = "🟠 Average"
        else:
            rating = "🔴 Needs Optimization"
        
        print(f"\n🏆 Performance Rating: {rating}")
        
        # Detailed breakdown
        print(f"\n📊 Individual Query Times:")
        for i, (query, time_taken) in enumerate(zip(test_queries[:len(query_times)], query_times), 1):
            short_query = query[:40] + "..." if len(query) > 40 else query
            print(f"   {i:2d}. {time_taken:5.2f}s - {short_query}")
    
    else:
        print("❌ No successful queries to analyze")

def benchmark_specific_query(query, iterations=3):
    """Benchmark a specific query multiple times"""
    print(f"🎯 Benchmarking specific query {iterations} times:")
    print(f"Query: {query}")
    print("-" * 50)
    
    rag_system = PolicyExpertRAG()
    times = []
    
    for i in range(iterations):
        start = time.time()
        response = rag_system.query(query)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage time for '{query}': {avg_time:.2f}s")
    return avg_time

def memory_usage_test():
    """Test memory usage during queries"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    print("🧠 Memory Usage Test")
    print("-" * 30)
    
    # Before initialization
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory before init: {mem_before:.1f} MB")
    
    # After initialization
    rag_system = PolicyExpertRAG()
    mem_after_init = process.memory_info().rss / 1024 / 1024
    print(f"Memory after init: {mem_after_init:.1f} MB")
    print(f"Initialization overhead: {mem_after_init - mem_before:.1f} MB")
    
    # After some queries
    for i in range(3):
        rag_system.query("What is covered under mental illness treatment?")
    
    mem_after_queries = process.memory_info().rss / 1024 / 1024
    print(f"Memory after queries: {mem_after_queries:.1f} MB")
    print(f"Query overhead: {mem_after_queries - mem_after_init:.1f} MB")

if __name__ == "__main__":
    print("🚀 RAG System Benchmarking Tool")
    print("=" * 50)
    
    choice = input("""
Choose benchmarking option:
1. Full system benchmark (recommended)
2. Specific query benchmark
3. Memory usage test
4. All tests

Enter choice (1-4): """).strip()
    
    if choice == "1":
        benchmark_rag_system()
    elif choice == "2":
        custom_query = input("Enter your query: ").strip()
        iterations = int(input("Number of iterations (default 3): ") or "3")
        benchmark_specific_query(custom_query, iterations)
    elif choice == "3":
        memory_usage_test()
    elif choice == "4":
        benchmark_rag_system()
        print("\n" + "="*50)
        memory_usage_test()
    else:
        print("Invalid choice. Running full benchmark...")
        benchmark_rag_system()