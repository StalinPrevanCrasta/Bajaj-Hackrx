import os
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import logging
import re

# Load environment variables
load_dotenv()

class PolicyExpertRAG:
    def __init__(self):
        self.setup_gemini()
        self.setup_embeddings()
        self.vector_store = None
        self.documents = []
        
    def setup_gemini(self):
        """Initialize Gemini API"""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.llm = GoogleGenerativeAI(
            model="gemini-2.5-flash",  # Updated model name
            google_api_key=api_key,
            temperature=0.1
        )
        
    def setup_embeddings(self):
        """Initialize embeddings model with GPU support"""
        import torch
        
        # Auto-detect best device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Using device: {device.upper()}")
        
        if torch.cuda.is_available():
            print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("💻 Using CPU - for better performance, consider GPU setup")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}  # Will use GPU if available
        )
    
    def load_policy_documents(self, extracted_text_path="extracted_text/"):
        """Load and process policy documents"""
        documents = []
        
        if not os.path.exists(extracted_text_path):
            print(f"Directory {extracted_text_path} not found!")
            return documents
        
        # Load documents from extracted_text directory
        for filename in os.listdir(extracted_text_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(extracted_text_path, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                # Create document with metadata
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
    
    def create_vector_store(self):
        """Create vector store from documents"""
        if not self.documents:
            self.load_policy_documents()
        
        if not self.documents:
            print("❌ No documents found to create vector store!")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(self.documents)
        print(f"📄 Created {len(chunks)} document chunks")
        
        # Create db directory if it doesn't exist
        os.makedirs("db", exist_ok=True)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(
            chunks, 
            self.embeddings
        )
        
        # Save vector store
        self.vector_store.save_local("db/vector_store")
        print("💾 Vector store created and saved")
        
    def load_vector_store(self):
        """Load existing vector store"""
        try:
            self.vector_store = FAISS.load_local(
                "db/vector_store", 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Vector store loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading vector store: {e}")
            print("🔄 Creating new vector store...")
            self.create_vector_store()
    
    def retrieve_relevant_documents(self, query, k=5):
        """Retrieve relevant documents for a query"""
        if not self.vector_store:
            self.load_vector_store()
        
        # Perform similarity search
        relevant_docs = self.vector_store.similarity_search(query, k=k)
        return relevant_docs
    
    def generate_response(self, query, relevant_docs):
        """Generate response using Gemini API"""
        
        # Prepare context from relevant documents
        context = "\n\n".join([
            f"Document: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in relevant_docs
        ])
        
        # Create expert prompt
        prompt = f"""
You are an expert insurance policy advisor specializing in Bajaj Allianz insurance policies.
You have deep knowledge of policy terms, conditions, coverage, exclusions, and claim procedures.

Based on the following policy documents, provide a short, clear, and accurate answer to the user's question.

POLICY DOCUMENTS:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer in one or two concise sentences, using simple and professional language.
2. Base your answer only on the provided policy documents — do NOT assume or use outside information.
3. If the answer is not available in the documents, respond with: "This information is not available in the provided policy documents."
4. Include specific policy terms (e.g., waiting period, sub-limit, UIN) if mentioned in the documents.
5. Explain any technical terms briefly if they appear in the answer.
6. Do NOT return JSON or lists — just a plain text answer.
"""

        
        try:
            # Generate response using Gemini
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"
    
    def query(self, user_question):
        """Main method to handle user queries"""
        print(f"🔍 Processing query: {user_question}")
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(user_question)
        
        if not relevant_docs:
            return "I couldn't find relevant information in the policy documents to answer your question."
        
        # Generate response
        response = self.generate_response(user_question, relevant_docs)
        
        return response

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag_system = PolicyExpertRAG()
    
    # Create/load vector store
    rag_system.create_vector_store()
    
    # Example queries
    test_queries = [
        "What is covered under mental illness treatment?",
        "What are the exclusions for dental treatment?",
        "How do I make a cashless claim?",
        "What is the waiting period for pre-existing diseases?",
        "What expenses are covered for air ambulance?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Q: {query}")
        print(f"{'='*50}")
        response = rag_system.query(query)
        print(f"A: {response}")