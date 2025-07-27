import os
import time
import hashlib
import json
from functools import lru_cache
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class OptimizedPolicyExpertRAG:
    def __init__(self):
        self.query_cache = {}
        self.last_cache_clear = time.time()
        self.setup_optimized_components()
        self.preload_system()
        
    def setup_optimized_components(self):
        """Setup all components with optimizations"""
        print("🔧 Setting up optimized components...")
        
        # Faster, smaller embedding model - FIXED VERSION
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
            model_kwargs={
                'device': 'cpu'
            }
            # Removed encode_kwargs completely to avoid conflicts
        )
        
        # Optimized Gemini setup with faster model
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        self.llm = GoogleGenerativeAI(
            model="gemini-2.5-flash",  # Faster than gemini-2.5-flash
            google_api_key=api_key,
            temperature=0.1,
            max_tokens=150  # Limit response length for speed
        )
        
        self.vector_store = None
        self.documents = []
    
    def preload_system(self):
        """Preload system components for faster queries"""
        print("🚀 Preloading optimized system...")
        
        # Preload embeddings model
        try:
            dummy_text = "test initialization"
            self.embeddings.embed_query(dummy_text)
            print("✅ Embeddings model loaded")
        except Exception as e:
            print(f"⚠️ Embeddings preload warning: {e}")
        
        # Load or create vector store
        self.load_optimized_vector_store()
        
        print("✅ Optimized system ready!")
    
    def load_policy_documents(self, extracted_text_path="extracted_text/"):
        """Load policy documents with optimizations"""
        documents = []
        
        if not os.path.exists(extracted_text_path):
            print(f"Directory {extracted_text_path} not found!")
            return documents
        
        for filename in os.listdir(extracted_text_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(extracted_text_path, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': filename,
                        'policy_type': self.extract_policy_type(content),
                        'uin': self.extract_uin(content)
                    }
                )
                documents.append(doc)
        
        self.documents = documents
        print(f"✅ Loaded {len(documents)} policy documents")
        return documents
    
    def extract_policy_type(self, content):
        """Extract policy type from content"""
        content_lower = content.lower()
        if 'global health care' in content_lower:
            return 'Global Health Care'
        elif 'health' in content_lower:
            return 'Health Insurance'
        return 'General Insurance'
    
    def extract_uin(self, content):
        """Extract UIN from content"""
        import re
        uin_match = re.search(r'UIN-?\s*([A-Z0-9]+)', content)
        return uin_match.group(1) if uin_match else 'Unknown'
    
    def create_optimized_vector_store(self):
        """Create optimized vector store"""
        if not self.documents:
            self.load_policy_documents()
        
        if not self.documents:
            print("❌ No documents found!")
            return
        
        # Smaller chunks for faster processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,    # Reduced from 1000
            chunk_overlap=50,  # Reduced from 200
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(self.documents)
        print(f"📄 Created {len(chunks)} optimized chunks")
        
        # Create db directory
        os.makedirs("db", exist_ok=True)
        
        # Create vector store with batch processing
        print("🔄 Creating optimized vector store...")
        try:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            
            # Save vector store
            self.vector_store.save_local("db/optimized_vector_store")
            print("💾 Optimized vector store created and saved")
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise
    
    def load_optimized_vector_store(self):
        """Load optimized vector store"""
        try:
            self.vector_store = FAISS.load_local(
                "db/optimized_vector_store", 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Optimized vector store loaded")
        except Exception as e:
            print(f"⚠️ Creating new optimized vector store: {e}")
            self.create_optimized_vector_store()
    
    def fast_similarity_search(self, query, k=3):
        """Optimized similarity search with caching"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        # Check if we have cached results
        cache_key = f"search_{query_hash}_{k}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Perform search with fewer results for speed
        relevant_docs = self.vector_store.similarity_search(query, k=k)
        
        # Cache the results
        self.query_cache[cache_key] = relevant_docs
        return relevant_docs
    
    def generate_fast_response(self, query, relevant_docs):
        """Generate optimized response with structured JSON output"""
        if not relevant_docs:
            return {
                "status": "error",
                "message": "No relevant information found in the policy documents.",
                "query": query,
                "sources": [],
                "confidence": 0
            }
        
        # Limit context size for faster processing
        max_context_length = 1500
        
        context_parts = []
        current_length = 0
        sources = []
        
        for doc in relevant_docs:
            doc_text = doc.page_content
            source = doc.metadata.get('source', 'Unknown')
            sources.append(source)
            
            if current_length + len(doc_text) > max_context_length:
                remaining = max_context_length - current_length
                doc_text = doc_text[:remaining]
                context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        context = "\n\n".join(context_parts)
        
        # Enhanced prompt for JSON-structured responses
        prompt = f"""You are an expert insurance advisor. Based on the following Bajaj Allianz policy information, provide a clear and helpful answer.

POLICY INFORMATION:
{context}

USER QUESTION: {query}

INSTRUCTIONS: Provide a direct, informative answer in 1-3 sentences. Use only the information provided above.

Respond with ONLY the answer text, no JSON formatting needed - the system will structure it automatically.

ANSWER:"""

        try:
            print("🔄 Generating JSON response...")
            response = self.llm.invoke(prompt)
            
            # Enhanced response handling
            result = None
            
            # Try different ways to extract the response
            if hasattr(response, 'content') and response.content:
                result = response.content.strip()
            elif hasattr(response, 'text') and response.text:
                result = response.text.strip()
            elif isinstance(response, str) and response:
                result = response.strip()
            elif hasattr(response, '__str__'):
                result = str(response).strip()
            
            # Check if we got a valid response
            if result and len(result) > 10:  # Must be more than 10 characters
                return {
                    "status": "success",
                    "answer": result,
                    "query": query,
                    "sources": list(set(sources)),  # Remove duplicates
                    "confidence": min(0.9, len(relevant_docs) * 0.3),  # Confidence based on doc count
                    "response_time": time.time()
                }
            else:
                # Fallback: Try to extract key information directly from context
                fallback_response = self.extract_direct_answer(query, context)
                return {
                    "status": "partial",
                    "answer": fallback_response,
                    "query": query,
                    "sources": list(set(sources)),
                    "confidence": 0.5,
                    "response_time": time.time(),
                    "note": "Fallback response generated"
                }
                
        except Exception as e:
            print(f"❌ Error in response generation: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing query: {str(e)}",
                "query": query,
                "sources": sources,
                "confidence": 0,
                "response_time": time.time()
            }
    
    def extract_direct_answer(self, query, context):
        """Fallback method to extract direct answer from context - returns text only"""
        # Simple keyword-based extraction for common queries
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Extract relevant sentences based on query
        sentences = context.split('.')
        relevant_sentences = []
        
        # Define keyword mappings
        keyword_map = {
            'mental illness': ['mental', 'psychiatric', 'psychological'],
            'cashless': ['cashless', 'claim', 'hospital'],
            'dental': ['dental', 'teeth', 'oral'],
            'exclusion': ['exclusion', 'not covered', 'excluded'],
            'waiting period': ['waiting', 'period', 'days'],
            'air ambulance': ['air ambulance', 'emergency', 'transport']
        }
        
        # Find relevant keywords
        for topic, keywords in keyword_map.items():
            if any(kw in query_lower for kw in keywords):
                for sentence in sentences:
                    if any(kw in sentence.lower() for kw in keywords):
                        relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            # Return first 2 most relevant sentences
            return '. '.join(relevant_sentences[:2]) + '.'
        else:
            return "Based on the policy document, this information requires more specific details from the policy terms."
    
    def fast_query(self, user_question, return_json=True):
        """Ultra-fast query processing with JSON output option"""
        start_time = time.time()
        
        # Clear cache periodically to prevent memory issues
        current_time = time.time()
        if current_time - self.last_cache_clear > 3600:  # Clear every hour
            self.query_cache.clear()
            self.last_cache_clear = current_time
        
        # Check full query cache first
        query_hash = hashlib.md5(user_question.encode()).hexdigest()
        cache_key = f"json_{query_hash}" if return_json else f"text_{query_hash}"
        
        if cache_key in self.query_cache:
            elapsed = time.time() - start_time
            cached_response = self.query_cache[cache_key]
            if return_json and isinstance(cached_response, dict):
                cached_response["cached"] = True
                cached_response["response_time"] = elapsed
            print(f"⚡ Cached response in {elapsed:.2f}s")
            return cached_response
        
        print(f"🔍 Processing: {user_question}")
        
        # Fast similarity search (only top 2-3 results)
        relevant_docs = self.fast_similarity_search(user_question, k=2)
        
        if not relevant_docs:
            no_results = {
                "status": "error",
                "message": "No relevant information found in the policy documents.",
                "query": user_question,
                "sources": [],
                "confidence": 0,
                "response_time": time.time() - start_time
            }
            return no_results if return_json else no_results["message"]
        
        # Generate response (always returns JSON now)
        response = self.generate_fast_response(user_question, relevant_docs)
        
        # Add timing information
        elapsed = time.time() - start_time
        response["response_time"] = elapsed
        response["cached"] = False
        
        # Cache the complete response
        self.query_cache[cache_key] = response
        
        print(f"⚡ JSON response generated in {elapsed:.2f}s")
        
        # Return JSON or text based on parameter
        if return_json:
            return response
        else:
            # For backward compatibility, return just the answer text
            return response.get("answer", response.get("message", "No answer available"))
    
    def batch_query(self, questions, return_json=True):
        """Process multiple queries efficiently with JSON output"""
        print(f"🔄 Processing {len(questions)} queries in batch...")
        results = []
        
        start_time = time.time()
        for i, question in enumerate(questions, 1):
            print(f"  📝 Query {i}/{len(questions)}")
            response = self.fast_query(question, return_json=return_json)
            
            if return_json:
                results.append({
                    "query_index": i,
                    "question": question,
                    "response": response
                })
            else:
                results.append((question, response))
        
        total_time = time.time() - start_time
        avg_time = total_time / len(questions)
        
        if return_json:
            batch_result = {
                "status": "success",
                "total_queries": len(questions),
                "total_time": total_time,
                "average_time": avg_time,
                "results": results,
                "timestamp": time.time()
            }
            print(f"✅ Batch completed in {total_time:.2f}s (avg: {avg_time:.2f}s per query)")
            return batch_result
        else:
            print(f"✅ Batch completed in {total_time:.2f}s (avg: {avg_time:.2f}s per query)")
            return results

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing Optimized RAG System")
    print("=" * 50)
    
    # Initialize optimized system
    try:
        optimized_rag = OptimizedPolicyExpertRAG()
        
        optimized_rag.create_optimized_vector_store()
        # Test queries
        test_queries = [
            "What is covered under mental illness treatment?",
            "How do I make a cashless claim?",
            "What are the exclusions for dental treatment?"
        ]
        
        # Single query test with JSON output
        print("\n🔍 Testing JSON Output:")
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            json_response = optimized_rag.fast_query(query, return_json=True)
            print(f"📊 JSON Response:")
            print(json.dumps(json_response, indent=2, default=str))
        
        # Test text output for backward compatibility
        print(f"\n🔍 Testing Text Output (Backward Compatibility):")
        for query in test_queries[:1]:  # Just test one
            print(f"\n❓ Query: {query}")
            text_response = optimized_rag.fast_query(query, return_json=False)
            print(f"📝 Text Response: {text_response}")
        
        # Test batch JSON output
        print(f"\n🔄 Testing Batch JSON Output:")
        batch_result = optimized_rag.batch_query(test_queries[:2], return_json=True)
        print(f"📊 Batch JSON Response:")
        print(json.dumps(batch_result, indent=2, default=str))
        
        print(f"\n{'='*50}")
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()