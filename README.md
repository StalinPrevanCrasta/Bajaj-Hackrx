# 🏥 Bajaj Allianz Policy Expert - RAG System

A Retrieval-Augmented Generation (RAG) system that acts as an intelligent insurance policy advisor for Bajaj Allianz insurance policies. This system uses Google's Gemini API to provide expert-level answers about policy coverage, exclusions, claims, and procedures.

## 🌟 Features

- **📄 PDF Text Extraction**: Extract text from insurance policy PDFs with OCR support
- **🔍 Semantic Search**: Find relevant policy information using vector similarity search
- **🤖 AI-Powered Responses**: Generate expert-level answers using Google Gemini API
- **💾 Vector Storage**: Persistent storage of document embeddings using FAISS
- **🔧 Multiple Interfaces**: CLI and programmatic access to the system
- **📋 Policy Metadata**: Automatic extraction of policy types and UIN numbers

## 🏗️ Project Structure

```
Bajaj-Hackrx/
├── 📁 policy_docs/          # Source PDF files
│   └── BAJHLIP23020V012223.pdf
├── 📁 extracted_text/       # Extracted text from PDFs (auto-generated)
│   └── BAJHLIP23020V012223.txt
├── 📁 scripts/              # Utility scripts
│   └── text_extractor.py    # PDF text extraction with OCR
├── 📁 src/                  # Main application code
│   ├── rag_system.py        # Core RAG implementation
│   └── policy_expert_cli.py # Command-line interface
├── 📁 db/                   # Vector store database (auto-generated)
├── 📁 test_docs/            # Additional test documents
├── .env                     # Environment variables (create this)
├── .gitignore              # Git ignore rules
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/0x5h31d0n/Bajaj-Hackrx.git
cd Bajaj-Hackrx

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

# Additional packages for RAG system
pip install google-generativeai langchain-google-genai faiss-cpu sentence-transformers langchain-huggingface
```

### 3. Setup Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

**📝 How to get Google API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

### 4. Extract Text from PDFs (Optional)

If you have new PDF files to process:

```bash
# Place PDF files in policy_docs/ directory
python scripts/text_extractor.py
```

### 5. Run the RAG System

```bash
# Test the system with example queries
python src/rag_system.py

# Or use the interactive CLI
python src/policy_expert_cli.py
```

## 💡 Usage Examples

### CLI Interface

```bash
python src/policy_expert_cli.py
```

**Sample Questions to Ask:**
- "What is covered under mental illness treatment?"
- "How do I make a cashless claim?"
- "What are the exclusions for dental treatment?"
- "What is the waiting period for pre-existing diseases?"
- "What expenses are covered for air ambulance?"

### Programmatic Usage

```python
from src.rag_system import PolicyExpertRAG

# Initialize the system
rag = PolicyExpertRAG()

# Ask questions
response = rag.query("What is covered under mental illness treatment?")
print(response)
```

## 🛠️ System Components

### 1. Text Extraction (`scripts/text_extractor.py`)
- Extracts text from PDF files using OCR
- Supports scanned documents
- Saves extracted text to `extracted_text/` directory

### 2. RAG System (`src/rag_system.py`)
- **Document Loading**: Processes extracted text files
- **Chunking**: Splits documents into manageable chunks
- **Embeddings**: Creates vector representations using sentence-transformers
- **Vector Store**: Uses FAISS for efficient similarity search
- **AI Generation**: Uses Google Gemini API for responses

### 3. CLI Interface (`src/policy_expert_cli.py`)
- Interactive command-line interface
- Real-time policy question answering
- Session management

## 📊 Supported Policy Types

Currently tested with:
- **Global Health Care Policy** (UIN: BAJHLIP23020V012223)
- General health insurance policies
- Other Bajaj Allianz insurance documents

## 🔧 Configuration

### Model Settings
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Model**: `gemini-1.5-flash`
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Similarity Search**: Top 5 relevant documents

### Environment Variables
```env
GOOGLE_API_KEY=your_api_key_here
```

## 🚨 Troubleshooting

### Common Issues

1. **Missing API Key**
   ```
   Error: GOOGLE_API_KEY not found in environment variables
   ```
   **Solution**: Create `.env` file with your Google API key

2. **Missing Dependencies**
   ```
   ModuleNotFoundError: No module named 'sentence_transformers'
   ```
   **Solution**: Install missing packages:
   ```bash
   pip install sentence-transformers langchain-huggingface faiss-cpu
   ```

3. **Model Not Found Error**
   ```
   404 models/gemini-pro is not found
   ```
   **Solution**: Update model name in `rag_system.py` to `gemini-1.5-flash`

4. **No Documents Found**
   ```
   ❌ No documents found to create vector store!
   ```
   **Solution**: Run text extraction first:
   ```bash
   python scripts/text_extractor.py
   ```

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (for embeddings)
- Internet connection (for Gemini API)

### Dependencies
- `langchain` - LLM framework
- `langchain-community` - Additional LangChain components
- `langchain-google-genai` - Google Gemini integration
- `langchain-huggingface` - HuggingFace embeddings
- `google-generativeai` - Google AI SDK
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Text embeddings
- `python-dotenv` - Environment variable management

## 🔐 Security

- **API Keys**: Never commit `.env` files to version control
- **Generated Content**: Vector stores and extracted text are ignored in git
- **Dependencies**: Regularly update packages for security patches

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is developed for the Bajaj Hackrx hackathon.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the error messages carefully
3. Ensure all dependencies are installed
4. Verify your API key is valid

## 🎯 Future Enhancements

- [ ] Web interface using Streamlit/Flask
- [ ] Support for more policy types
- [ ] Multi-language support
- [ ] Advanced query filtering
- [ ] Export functionality for responses
- [ ] Integration with more LLM providers

---

**Built with ❤️ by Scriptators for Bajaj Hackrx 2024**
