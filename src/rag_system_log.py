import os
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import re
import time
import logging
import json
import socket

# Load environment variables
load_dotenv()

class DocumentExpertRAG:
    def __init__(self):
        print("🚀 Initializing Document Expert RAG system...")
        start_time = time.time()
        self.setup_gemini()
        self.setup_embeddings()
        self.vector_store = None
        self.documents = []
        self._setup_query_logger()
        init_time = time.time() - start_time
        print(f"✅ System ready in {init_time:.2f}s!")
        
    def _setup_query_logger(self):
        """Sets up a dedicated logger to write queries to a file."""
        query_logger = logging.getLogger('query_logger')
        query_logger.setLevel(logging.INFO)
        query_logger.propagate = False
        
        if not query_logger.handlers:
            file_handler = logging.FileHandler('rag_queries.log')
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            query_logger.addHandler(file_handler)

    def setup_gemini(self):
        """Initialize Gemini API"""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        self.llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )
        
    def setup_embeddings(self):
        """Initialize embeddings model with GPU support"""
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Using device: {device.upper()}")
        if torch.cuda.is_available():
            print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("💻 Using CPU - for better performance, consider GPU setup")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2",
            model_kwargs={'device': device}
        )
    
    def load_documents(self, extracted_text_path="extracted_text/"):
        """Load and process documents from any domain"""
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
                        'document_type': self.extract_document_type(content),
                        'document_id': self.extract_document_id(content)
                    }
                )
                documents.append(doc)
        
        self.documents = documents
        print(f"✅ Loaded {len(documents)} documents")
        return documents
    
    def extract_document_type(self, content):
        """Extract document type from content dynamically"""
        content_lower = content[:1000].lower()
        if any(word in content_lower for word in ['insurance', 'policy', 'coverage', 'premium']):
            return 'Insurance Document'
        elif any(word in content_lower for word in ['contract', 'agreement', 'terms', 'conditions']):
            return 'Legal Document'
        elif any(word in content_lower for word in ['manual', 'guide', 'instructions', 'procedure']):
            return 'Manual/Guide'
        elif any(word in content_lower for word in ['report', 'analysis', 'findings', 'conclusion']):
            return 'Report'
        else:
            return 'General Document'
    
    def extract_document_id(self, content):
        """Extract document ID from content"""
        id_patterns = [
            r'UIN[-\s]*([A-Z0-9]+)', r'ID[-\s]*([A-Z0-9]+)', r'REF[-\s]*([A-Z0-9]+)',
            r'DOC[-\s]*([A-Z0-9]+)', r'NO[-\s]*([A-Z0-9]+)'
        ]
        for pattern in id_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        return 'Unknown'
    
    def create_vector_store(self):
        """Create vector store from documents"""
        if not self.documents:
            self.load_documents()
        if not self.documents:
            print("❌ No documents found to create vector store!")
            return
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=20, length_function=len,
        )
        chunks = text_splitter.split_documents(self.documents)
        print(f"📄 Created {len(chunks)} document chunks")
        
        os.makedirs("db", exist_ok=True)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local("db/vector_store")
        print("💾 Vector store created and saved")
        
    def load_vector_store(self):
        """Load existing vector store"""
        try:
            self.vector_store = FAISS.load_local(
                "db/vector_store", self.embeddings, allow_dangerous_deserialization=True
            )
            print("✅ Vector store loaded successfully")
        except Exception as e:
            print(f"⚠️ Error loading vector store: {e}")
            print("🔄 Creating new vector store...")
            self.create_vector_store()
    
    def retrieve_relevant_documents(self, query, k=3):
        """Retrieve relevant documents for a query"""
        if not self.vector_store:
            self.load_vector_store()
        return self.vector_store.similarity_search(query, k=k)
    
    def generate_response(self, query, relevant_docs):
        """Generate response using Gemini API"""
        context = "\n\n".join([
            f"Document: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for doc in relevant_docs
        ])
        prompt = f"""
You are a professional assistant designed to extract factual answers from official documents.
TASK: For each user question, provide a precise answer based only on the contents of the provided documents.
DOCUMENTS:
{context}
USER QUESTIONS:
{query}
GUIDELINES:
1. Answer each question in 1–2 clear and professional sentences.
2. Use ONLY the information explicitly found in the documents. Do NOT infer or use external knowledge.
3. If a question cannot be answered from the documents, respond with: "This information is not available in the provided documents."
4. Use specific terms, numbers, or conditions from the documents wherever possible.
5. If a technical term is used in the answer, briefly define it for clarity.
6. Ensure all answers are directly supported by the document and helpful to the user.
"""
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your query: {str(e)}"
    
    def query(self, user_question):
        """Main method to handle user queries and log the interaction"""
        start_time = time.time()
        print(f"🔍 Processing query: {user_question}")
        
        relevant_docs = self.retrieve_relevant_documents(user_question)
        if not relevant_docs:
            return "I couldn't find relevant information in the documents to answer your question."
        
        response = self.generate_response(user_question, relevant_docs)
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Response generated in {elapsed_time:.2f}s")
        
        # --- LOGGING LOGIC ---
        source_documents = [doc.metadata.get('source', 'Unknown') for doc in relevant_docs]
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hostname": socket.gethostname(),
            "query": user_question,
            "used_documents": source_documents,
            "response": response.strip(),
        }
        query_logger = logging.getLogger('query_logger')
        query_logger.info(json.dumps(log_entry))
        # --- END OF LOGGING LOGIC ---
        
        return response

# Example usage
if __name__ == "__main__":
    print("🤖 Document Expert RAG System")
    print("=" * 50)
    
    rag_system = DocumentExpertRAG()
    rag_system.create_vector_store()
    
    print("✅ System ready for queries!")
    print("You can now use the system. For example, in another script or interactive session:")
    print("response = rag_system.query('your question here')")
    print("print(response)")