import os

def create_optimization_directories():
    """Create necessary directories for optimization features"""
    
    directories = [
        'config',
        'benchmark_results',
        'logs',
        'db/cache',
        'db/embeddings_cache'
    ]
    
    print("🗂️  Creating optimization directories...")
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ Created: {directory}/")
    
    # Create __init__.py files for Python packages
    init_files = [
        'config/__init__.py'
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('# Optimization configuration package\n')
        print(f"  ✅ Created: {init_file}")
    
    print("\n✅ Optimization directory structure created!")

if __name__ == "__main__":
    create_optimization_directories()