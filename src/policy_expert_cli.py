from rag_system import PolicyExpertRAG
import sys

def main():
    print("🏥 Bajaj Allianz Policy Expert - RAG System")
    print("="*50)
    print("Initializing system...")
    
    try:
        # Initialize RAG system
        rag_system = PolicyExpertRAG()
        rag_system.load_vector_store()
        
        print("✅ System ready! Ask me anything about your insurance policies.")
        print("Type 'quit' or 'exit' to end the session.\n")
        
        while True:
            # Get user input
            query = input("🤔 Your question: ").strip()
            
            # Check for exit conditions
            if query.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thank you for using the Policy Expert system!")
                break
            
            if not query:
                print("Please enter a valid question.")
                continue
            
            # Process query
            print("\n🔍 Searching policy documents...")
            response = rag_system.query(query)
            
            print(f"\n💡 **Policy Expert Response:**")
            print("-" * 40)
            print(response)
            print("\n" + "="*50 + "\n")
            
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        print("Please check your environment variables and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()