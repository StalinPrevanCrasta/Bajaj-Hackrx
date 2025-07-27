"""
Configuration file for RAG system optimizations
"""

# Embedding Model Configuration
EMBEDDING_CONFIG = {
    'model_name': "sentence-transformers/paraphrase-MiniLM-L3-v2",  # Faster model
    'model_kwargs': {
        'device': 'cpu',
        'normalize_embeddings': True
    },
    'encode_kwargs': {
        'batch_size': 32,
        'show_progress_bar': False
    }
}

# Alternative embedding models (uncomment to try)
# EMBEDDING_CONFIG['model_name'] = "sentence-transformers/all-distilroberta-v1"  # Good balance
# EMBEDDING_CONFIG['model_name'] = "sentence-transformers/all-MiniLM-L6-v2"     # Original model

# Gemini API Configuration
GEMINI_CONFIG = {
    'model': "gemini-1.5-flash",  # Fastest available model
    'temperature': 0.1,
    'max_tokens': 150
}

# Text Chunking Configuration
CHUNKING_CONFIG = {
    'chunk_size': 500,      # Reduced from 1000 for faster processing
    'chunk_overlap': 50,    # Reduced from 200
    'length_function': len
}

# Vector Store Configuration
VECTOR_STORE_CONFIG = {
    'similarity_search_k': 2,  # Reduced from 5 for speed
    'store_path': "db/optimized_vector_store"
}

# Caching Configuration
CACHE_CONFIG = {
    'max_size': 100,           # Maximum cached queries
    'clear_interval': 3600,    # Clear cache every hour (seconds)
    'enable_response_cache': True,
    'enable_search_cache': True
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'excellent': 2.0,    # Response time under 2s = excellent
    'good': 4.0,         # Response time under 4s = good
    'average': 8.0,      # Response time under 8s = average
    'poor': 8.0          # Response time over 8s = needs optimization
}

# Batch Processing Configuration
BATCH_CONFIG = {
    'max_batch_size': 10,      # Maximum queries in a batch
    'batch_timeout': 30,       # Timeout for batch processing (seconds)
    'parallel_processing': False  # Set to True if you want to experiment with threading
}

# Memory Management
MEMORY_CONFIG = {
    'max_memory_mb': 1000,     # Maximum memory usage before cleanup
    'cleanup_threshold': 0.8,   # Cleanup when 80% of max memory is reached
    'monitor_memory': True
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',           # DEBUG, INFO, WARNING, ERROR
    'enable_performance_logs': True,
    'log_file': 'logs/rag_system.log',
    'max_log_size_mb': 10
}

# Development/Debug Configuration
DEBUG_CONFIG = {
    'enable_timing': True,      # Show timing information
    'enable_memory_tracking': True,
    'verbose_errors': True,
    'save_debug_info': False
}