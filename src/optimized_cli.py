from optimized_rag_system import OptimizedPolicyExpertRAG
import sys
import time
import json
from datetime import datetime

class OptimizedPolicyCLI:
    def __init__(self):
        self.rag_system = None
        self.session_queries = []
        self.session_start = datetime.now()
        self.show_json = False  # Toggle for JSON display
        
    def initialize_system(self):
        """Initialize the optimized RAG system"""
        print("🏥 Bajaj Allianz Policy Expert - Optimized Version")
        print("="*60)
        print("🚀 Initializing optimized system...")
        
        # Show device information
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Device: {device.upper()}")
        if torch.cuda.is_available():
            print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
        
        start_time = time.time()
        try:
            self.rag_system = OptimizedPolicyExpertRAG()
            init_time = time.time() - start_time
            print(f"✅ System ready in {init_time:.2f}s!")
            print("🎯 Optimizations active: Fast embeddings, caching, smaller chunks")
            return True
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            return False
    
    def display_help(self):
        """Display help information"""
        print("\n📋 Available Commands:")
        print("  help     - Show this help message")
        print("  stats    - Show session statistics")
        print("  clear    - Clear query cache")
        print("  batch    - Enter batch query mode")
        print("  json     - Toggle JSON output display")
        print("  quit/exit - End session")
        print("\n💡 Sample Questions:")
        print("  • What is covered under mental illness treatment?")
        print("  • How do I make a cashless claim?")
        print("  • What are the exclusions for dental treatment?")
        print("  • What is the waiting period for pre-existing diseases?")
        print("  • What expenses are covered for air ambulance?")
        print()
    
    def show_session_stats(self):
        """Show session statistics"""
        if not self.session_queries:
            print("📊 No queries processed in this session yet.")
            return
        
        total_time = sum(q['time'] for q in self.session_queries)
        avg_time = total_time / len(self.session_queries)
        fastest = min(self.session_queries, key=lambda x: x['time'])
        slowest = max(self.session_queries, key=lambda x: x['time'])
        
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        # Add device information
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"\n📊 Session Statistics:")
        print(f"   Session Duration: {session_duration:.0f}s")
        print(f"   Device Used: {device.upper()}")
        if torch.cuda.is_available():
            print(f"   GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB allocated")
        print(f"   Total Queries: {len(self.session_queries)}")
        print(f"   Average Response Time: {avg_time:.2f}s")
        print(f"   Fastest Query: {fastest['time']:.2f}s")
        print(f"   Slowest Query: {slowest['time']:.2f}s")
        print(f"   Queries per Minute: {len(self.session_queries) / (session_duration / 60):.1f}")
        
        cached_queries = sum(1 for q in self.session_queries if q['time'] < 0.5)
        if cached_queries > 0:
            print(f"   Cached Responses: {cached_queries} ({cached_queries/len(self.session_queries):.1%})")
        print()
    
    def clear_cache(self):
        """Clear the query cache"""
        if self.rag_system:
            self.rag_system.query_cache.clear()
            print("🗑️  Query cache cleared.")
        else:
            print("❌ System not initialized.")
    
    def toggle_json_display(self):
        """Toggle JSON output display"""
        self.show_json = not self.show_json
        status = "ON" if self.show_json else "OFF"
        print(f"📊 JSON output display: {status}")
        if self.show_json:
            print("💡 Responses will now show both formatted text and raw JSON")
        else:
            print("💡 Responses will show formatted text only")
    
    def batch_query_mode(self):
        """Enter batch query mode"""
        print("\n🔄 Batch Query Mode")
        print("Enter multiple queries (one per line), then type 'PROCESS' to execute all:")
        print("Type 'CANCEL' to return to normal mode.")
        
        queries = []
        while True:
            query = input(f"Query {len(queries)+1}: ").strip()
            
            if query.upper() == 'PROCESS':
                if queries:
                    break
                else:
                    print("No queries entered. Type 'CANCEL' to exit.")
                    continue
            elif query.upper() == 'CANCEL':
                print("Batch mode cancelled.")
                return
            elif query:
                queries.append(query)
        
        print(f"\n🔄 Processing {len(queries)} queries in batch...")
        start_time = time.time()
        
        try:
            # Use the new batch JSON system
            batch_result = self.rag_system.batch_query(queries, return_json=True)
            total_time = time.time() - start_time
            
            print(f"\n📋 Batch Results (completed in {total_time:.2f}s):")
            print("="*60)
            
            for result in batch_result['results']:
                query_index = result['query_index']
                question = result['question']
                response_data = result['response']
                
                answer = response_data.get('answer', response_data.get('message', 'No response'))
                status = response_data.get('status', 'unknown')
                confidence = response_data.get('confidence', 0)
                
                print(f"\n{query_index}. Q: {question}")
                print(f"   A: {answer}")
                
                if status == 'partial':
                    print(f"   ⚠️ Partial response (confidence: {confidence:.1f})")
                elif status == 'success':
                    print(f"   ✅ High confidence ({confidence:.1f})")
                
                # Add to session stats
                self.session_queries.append({
                    'query': question,
                    'time': total_time / len(queries),  # Approximate individual time
                    'cached': response_data.get('cached', False)
                })
                
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
    
    def process_query(self, query):
        """Process a single query"""
        print(f"\n🔍 Searching policy documents...")

        start_time = time.time()
        try:
            # Use the new JSON output system
            json_response = self.rag_system.fast_query(query, return_json=True)
            elapsed_time = time.time() - start_time

            # Extract information from JSON response
            status = json_response.get('status', 'unknown')
            answer = json_response.get('answer', json_response.get('message', 'No response'))
            cached = json_response.get('cached', False)
            confidence = json_response.get('confidence', 0)
            sources = json_response.get('sources', [])

            self.session_queries.append({
                'query': query,
                'time': elapsed_time,
                'cached': cached
            })

            print(f"\n💡 **Policy Expert Response** (⏱️ {elapsed_time:.2f}s):")
            print("-" * 50)
            print(answer)

            # Show additional info
            if status == 'partial':
                print(f"\n⚠️ Partial response (confidence: {confidence:.1f})")
            elif status == 'success':
                print(f"\n✅ High confidence response ({confidence:.1f})")
            elif status == 'error':
                print(f"\n❌ Error in response")

            if sources:
                print(f"📄 Sources: {', '.join(sources)}")

            if cached:
                print("\n💾 (Cached response)")

            # Show JSON if enabled
            if self.show_json:
                print(f"\n📊 **Raw JSON Response:**")
                print("-" * 50)
                print(json.dumps(json_response, indent=2, default=str))

        except Exception as e:
            print(f"❌ Error processing query: {e}")

    
    def run(self):
        """Main CLI loop"""
        if not self.initialize_system():
            sys.exit(1)
        
        self.display_help()
        
        print("Type your question or 'help' for commands:")
        
        while True:
            try:
                query = input("\n🤔 Your question: ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() in ['quit', 'exit', 'bye']:
                    self.show_session_stats()
                    print("👋 Thank you for using the Optimized Policy Expert system!")
                    break
                elif query.lower() == 'help':
                    self.display_help()
                elif query.lower() == 'stats':
                    self.show_session_stats()
                elif query.lower() == 'clear':
                    self.clear_cache()
                elif query.lower() == 'json':
                    self.toggle_json_display()
                elif query.lower() == 'batch':
                    self.batch_query_mode()
                else:
                    # Process as a regular query
                    self.process_query(query)
                
                print("\n" + "="*60)
                
            except KeyboardInterrupt:
                print("\n\n⏹️  Session interrupted by user.")
                self.show_session_stats()
                print("👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                print("Type 'quit' to exit or continue with another query.")

if __name__ == "__main__":
    cli = OptimizedPolicyCLI()
    cli.run()