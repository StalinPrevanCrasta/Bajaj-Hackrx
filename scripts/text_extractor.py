import os
import time
from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader

# Define paths relative to the project root
DOCS_PATH = 'policy_docs/'
OUTPUT_PATH = 'extracted_text/'

def extract_and_save_text_with_ocr():
    start_time = time.time()
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    if not os.path.exists(DOCS_PATH) or not os.listdir(DOCS_PATH):
        print(f"❌ Error: The '{DOCS_PATH}' directory does not exist or is empty.")
        return

    print(f"📄 Loading documents from '{DOCS_PATH}' with OCR enabled...")

    loader = DirectoryLoader(DOCS_PATH, glob='*.pdf', loader_cls=UnstructuredPDFLoader)
    documents = loader.load()

    print(f"\n✅ Successfully loaded {len(documents)} document(s).")
    print(f"📝 Saving extracted text to '{OUTPUT_PATH}'...\n")

    for doc in documents:
        source_path = doc.metadata.get('source', 'unknown.pdf')
        original_filename = os.path.basename(source_path)
        base_filename, _ = os.path.splitext(original_filename)
        output_filename = f"{base_filename}.txt"
        output_filepath = os.path.join(OUTPUT_PATH, output_filename)

        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(doc.page_content)
            print(f"   ✔️ Saved: {output_filename}")
        except Exception as e:
            print(f"   ❌ Error saving {output_filename}: {e}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n⏱️ Extraction and saving complete in {duration:.2f} seconds.")

if __name__ == '__main__':
    extract_and_save_text_with_ocr()