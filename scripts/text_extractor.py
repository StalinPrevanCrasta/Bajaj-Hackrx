# scripts/text_extractor.py

import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader

# Define the path to your source documents relative to the project root
DOCS_PATH = 'policy_docs/'

def extract_text_from_pdfs():
    """
    Loads all PDF documents from a specified directory and extracts their text content.
    """
    if not os.path.exists(DOCS_PATH) or not os.listdir(DOCS_PATH):
        print(f"Error: The '{DOCS_PATH}' directory does not exist or is empty.")
        print("Please create it and add your policy PDF files.")
        return

    print(f"Loading documents from '{DOCS_PATH}'...")
    
    # DirectoryLoader scans the folder for files matching the glob pattern
    # and uses PyMuPDFLoader to process each PDF file found.
    loader = DirectoryLoader(DOCS_PATH, glob='*.pdf', loader_cls=PyMuPDFLoader)
    
    # The .load() method extracts the text and returns a list of Document objects
    documents = loader.load()

    print(f"\n✅ Successfully loaded {len(documents)} document(s).")
    print("--- Sample Extracted Content ---")

    # Loop through each loaded document and print its source and a content snippet
    for i, doc in enumerate(documents):
        # The 'metadata' attribute contains information like the source file path
        source = doc.metadata.get('source', 'Unknown source')
        # The 'page_content' attribute contains the extracted text
        content_snippet = doc.page_content[:500] + "..." # Get the first 500 characters
        
        print(f"\n📄 Document {i+1}: {os.path.basename(source)}")
        print("-------------------------------------------------")
        print(content_snippet)
        print("-------------------------------------------------")


if __name__ == '__main__':
    extract_text_from_pdfs()