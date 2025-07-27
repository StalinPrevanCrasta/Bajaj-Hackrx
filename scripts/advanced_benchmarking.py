import time
import sys
import os
import statistics
import psutil
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.optimized_rag_system import OptimizedPolicyExpertRAG

class AdvancedBenchmark:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.get_system_info(),
            'tests': {}
        }
    
    def get_system_info(self):
        """Get system information"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'python_version': sys.version,
            'platform': sys.platform
        }
    
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function"""
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        
        # Execute function
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Get final memory
        final_memory = process.memory_info().rss / (1024**2)  # MB
        memory_delta = final_memory - initial_memory
        
        return {
            'result': result,
            'execution_time': execution_time,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_delta_mb': memory_delta
        }
    
    def benchmark_initialization(self):
        """Benchmark system initialization"""
        print("🚀 Benchmarking System Initialization...")
        
        # Test multiple initializations
        init_times = []
        memory_usages = []
        
        for i in range(3):
            print(f"  Init test {i+1}/3...")
            
            measurement = self.measure_memory_usage(OptimizedPolicyExpertRAG)
            init_times.append(measurement['execution_time'])
            memory_usages.append(measurement['memory_delta_mb'])
        
        self.results['tests']['initialization'] = {
            'times': init_times,
            'avg_time': statistics.mean(init_times),
            'min_time': min(init_times),
            'max_time': max(init_times),
            'memory_usage': {
                'avg_mb': statistics.mean(memory_usages),
                'min_mb': min(memory_usages),
                'max_mb': max(memory_usages)
            }
        }
        
        print(f"  ✅ Average init time: {statistics.mean(init_times):.2f}s")
        print(f"  💾 Average memory usage: {statistics.mean(memory_usages):.1f}MB")
    
    def benchmark_query_performance(self, rag_system):
        """Comprehensive query performance testing"""
        print("\n📊 Benchmarking Query Performance...")
        
        test_categories = {
            'simple': [
                "What is mental illness coverage?",
                "How to claim?",
                "What is UIN number?"
            ],
            'complex': [
                "What are the specific exclusions for mental illness treatment including waiting periods and sub-limits?",
                "Explain the complete cashless claim process including required documents and approval procedures?",
                "What are all the covered expenses for air ambulance including geographical limitations?"
            ],
            'edge_cases': [
                "What is not mentioned in the policy?",
                "How much does it cost?",
                "When was this policy created?"
            ]
        }
        
        query_results = {}
        
        for category, queries in test_categories.items():
            print(f"\n  🔍 Testing {category} queries...")
            category_times = []
            category_memory = []
            category_success = 0
            
            for i, query in enumerate(queries, 1):
                print(f"    Query {i}/{len(queries)}: {query[:50]}...")
                
                measurement = self.measure_memory_usage(
                    rag_system.fast_query, query
                )
                
                category_times.append(measurement['execution_time'])
                category_memory.append(measurement['memory_delta_mb'])
                
                # Check if query succeeded (not an error message)
                if not measurement['result'].startswith('Error'):
                    category_success += 1
                
                print(f"      ⏱️  {measurement['execution_time']:.2f}s")
            
            query_results[category] = {
                'times': category_times,
                'avg_time': statistics.mean(category_times),
                'success_rate': category_success / len(queries),
                'memory_usage': statistics.mean(category_memory)
            }
        
        self.results['tests']['query_performance'] = query_results
        
        # Print summary
        for category, results in query_results.items():
            print(f"\n  📈 {category.title()} Queries:")
            print(f"    Average time: {results['avg_time']:.2f}s")
            print(f"    Success rate: {results['success_rate']:.1%}")
            print(f"    Memory usage: {results['memory_usage']:.1f}MB")
    
    def benchmark_caching_performance(self, rag_system):
        """Test caching effectiveness"""
        print("\n🔄 Benchmarking Caching Performance...")
        
        test_query = "What is covered under mental illness treatment?"
        
        # First run (no cache)
        first_run = self.measure_memory_usage(rag_system.fast_query, test_query)
        
        # Subsequent runs (with cache)
        cached_runs = []
        for i in range(5):
            measurement = self.measure_memory_usage(rag_system.fast_query, test_query)
            cached_runs.append(measurement['execution_time'])
        
        cache_speedup = first_run['execution_time'] / statistics.mean(cached_runs)
        
        self.results['tests']['caching'] = {
            'first_run_time': first_run['execution_time'],
            'cached_avg_time': statistics.mean(cached_runs),
            'cache_speedup': cache_speedup,
            'cached_times': cached_runs
        }
        
        print(f"  First run: {first_run['execution_time']:.2f}s")
        print(f"  Cached average: {statistics.mean(cached_runs):.2f}s")
        print(f"  Speedup: {cache_speedup:.1f}x faster")
    
    def benchmark_batch_processing(self, rag_system):
        """Test batch processing performance"""
        print("\n🔄 Benchmarking Batch Processing...")
        
        batch_queries = [
            "What is mental illness coverage?",
            "How to make cashless claim?",
            "What are dental exclusions?",
            "What is waiting period?",
            "What is air ambulance coverage?"
        ]
        
        # Individual processing
        individual_times = []
        for query in batch_queries:
            measurement = self.measure_memory_usage(rag_system.fast_query, query)
            individual_times.append(measurement['execution_time'])
        
        individual_total = sum(individual_times)
        
        # Batch processing
        batch_measurement = self.measure_memory_usage(
            rag_system.batch_query, batch_queries
        )
        
        batch_efficiency = (individual_total - batch_measurement['execution_time']) / individual_total
        
        self.results['tests']['batch_processing'] = {
            'individual_times': individual_times,
            'individual_total': individual_total,
            'batch_time': batch_measurement['execution_time'],
            'efficiency_gain': batch_efficiency
        }
        
        print(f"  Individual total: {individual_total:.2f}s")
        print(f"  Batch processing: {batch_measurement['execution_time']:.2f}s")
        print(f"  Efficiency gain: {batch_efficiency:.1%}")
    
    def benchmark_concurrent_load(self, rag_system):
        """Test system under concurrent load simulation"""
        print("\n⚡ Benchmarking Concurrent Load Simulation...")
        
        import threading
        import queue
        
        test_queries = [
            "What is mental illness coverage?",
            "How to claim?",
            "What exclusions exist?",
            "What is waiting period?",
            "Coverage limits?"
        ]
        
        results_queue = queue.Queue()
        
        def worker_thread(query, thread_id):
            start_time = time.time()
            try:
                response = rag_system.fast_query(f"{query} (thread {thread_id})")
                elapsed = time.time() - start_time
                results_queue.put(('success', elapsed, thread_id))
            except Exception as e:
                elapsed = time.time() - start_time
                results_queue.put(('error', elapsed, thread_id, str(e)))
        
        # Simulate concurrent requests
        num_threads = 5
        threads = []
        
        start_time = time.time()
        
        for i in range(num_threads):
            query = test_queries[i % len(test_queries)]
            thread = threading.Thread(target=worker_thread, args=(query, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        total_concurrent_time = time.time() - start_time
        
        # Collect results
        concurrent_times = []
        successful_threads = 0
        
        while not results_queue.empty():
            result = results_queue.get()
            if result[0] == 'success':
                concurrent_times.append(result[1])
                successful_threads += 1
        
        self.results['tests']['concurrent_load'] = {
            'num_threads': num_threads,
            'total_time': total_concurrent_time,
            'successful_threads': successful_threads,
            'avg_thread_time': statistics.mean(concurrent_times) if concurrent_times else 0,
            'thread_times': concurrent_times
        }
        
        print(f"  Threads: {num_threads}")
        print(f"  Total time: {total_concurrent_time:.2f}s")
        print(f"  Successful: {successful_threads}/{num_threads}")
        if concurrent_times:
            print(f"  Avg thread time: {statistics.mean(concurrent_times):.2f}s")
    
    def save_results(self, filename="benchmark_results.json"):
        """Save benchmark results to file"""
        results_dir = "benchmark_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(results_dir, f"{timestamp}_{filename}")
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
        return filepath
    
    def generate_report(self):
        """Generate a human-readable report"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE BENCHMARK REPORT")
        print("="*60)
        
        # System info
        print(f"\n🖥️  System Information:")
        print(f"   CPU Cores: {self.results['system_info']['cpu_count']}")
        print(f"   Total Memory: {self.results['system_info']['memory_total']:.1f}GB")
        print(f"   Platform: {self.results['system_info']['platform']}")
        
        # Initialization
        if 'initialization' in self.results['tests']:
            init = self.results['tests']['initialization']
            print(f"\n🚀 Initialization Performance:")
            print(f"   Average Time: {init['avg_time']:.2f}s")
            print(f"   Memory Usage: {init['memory_usage']['avg_mb']:.1f}MB")
        
        # Query performance
        if 'query_performance' in self.results['tests']:
            print(f"\n📊 Query Performance:")
            for category, data in self.results['tests']['query_performance'].items():
                print(f"   {category.title()}: {data['avg_time']:.2f}s (success: {data['success_rate']:.1%})")
        
        # Caching
        if 'caching' in self.results['tests']:
            cache = self.results['tests']['caching']
            print(f"\n🔄 Caching Performance:")
            print(f"   Speedup: {cache['cache_speedup']:.1f}x faster")
            print(f"   First run: {cache['first_run_time']:.2f}s")
            print(f"   Cached avg: {cache['cached_avg_time']:.2f}s")
        
        # Overall performance rating
        if 'query_performance' in self.results['tests']:
            simple_avg = self.results['tests']['query_performance']['simple']['avg_time']
            
            if simple_avg < 1:
                rating = "🟢 Excellent"
            elif simple_avg < 2:
                rating = "🟡 Good"
            elif simple_avg < 4:
                rating = "🟠 Average"
            else:
                rating = "🔴 Needs Optimization"
            
            print(f"\n🏆 Overall Performance Rating: {rating}")
    
    def run_full_benchmark(self):
        """Run all benchmark tests"""
        print("🔥 Running Comprehensive Benchmark Suite")
        print("="*60)
        
        # Initialize system
        self.benchmark_initialization()
        
        # Create system instance for other tests
        print("\n🔧 Creating system instance for testing...")
        rag_system = OptimizedPolicyExpertRAG()
        
        # Run all benchmarks
        self.benchmark_query_performance(rag_system)
        self.benchmark_caching_performance(rag_system)
        self.benchmark_batch_processing(rag_system)
        self.benchmark_concurrent_load(rag_system)
        
        # Generate report
        self.generate_report()
        
        # Save results
        return self.save_results()

if __name__ == "__main__":
    benchmark = AdvancedBenchmark()
    
    print("🚀 Advanced RAG System Benchmarking")
    print("="*50)
    
    choice = input("""
Choose benchmark level:
1. Quick benchmark (5 minutes)
2. Full benchmark suite (10-15 minutes)
3. Custom test selection

Enter choice (1-3): """).strip()
    
    if choice == "1":
        # Quick benchmark
        benchmark.benchmark_initialization()
        rag_system = OptimizedPolicyExpertRAG()
        benchmark.benchmark_query_performance(rag_system)
        benchmark.benchmark_caching_performance(rag_system)
        benchmark.generate_report()
        benchmark.save_results("quick_benchmark.json")
        
    elif choice == "2":
        # Full benchmark
        results_file = benchmark.run_full_benchmark()
        print(f"\n✅ Comprehensive benchmark completed!")
        print(f"📁 Results saved to: {results_file}")
        
    elif choice == "3":
        # Custom selection
        print("\nAvailable tests:")
        print("1. Initialization")
        print("2. Query Performance")
        print("3. Caching")
        print("4. Batch Processing")
        print("5. Concurrent Load")
        
        selected = input("Enter test numbers (comma-separated): ").strip().split(',')
        
        if '1' in selected:
            benchmark.benchmark_initialization()
        
        rag_system = OptimizedPolicyExpertRAG()
        
        if '2' in selected:
            benchmark.benchmark_query_performance(rag_system)
        if '3' in selected:
            benchmark.benchmark_caching_performance(rag_system)
        if '4' in selected:
            benchmark.benchmark_batch_processing(rag_system)
        if '5' in selected:
            benchmark.benchmark_concurrent_load(rag_system)
        
        benchmark.generate_report()
        benchmark.save_results("custom_benchmark.json")
    
    else:
        print("Invalid choice. Running quick benchmark...")
        benchmark.benchmark_initialization()
        rag_system = OptimizedPolicyExpertRAG()
        benchmark.benchmark_query_performance(rag_system)
        benchmark.generate_report()