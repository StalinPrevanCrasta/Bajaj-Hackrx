import os
import time
import torch
from pathlib import Path
import fitz  # PyMuPDF for better PDF processing
import cv2
import numpy as np
from PIL import Image
import pytesseract
from langchain_community.document_loaders import (
    UnstructuredPDFLoader, 
    UnstructuredWordDocumentLoader,
    DirectoryLoader
)
from langchain.schema import Document

# Define paths relative to the project root
DOCS_PATH = 'policy_docs/'
OUTPUT_PATH = 'extracted_text/'

class GPUAcceleratedTextExtractor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.setup_extraction_methods()
        
    def setup_extraction_methods(self):
        """Setup GPU-accelerated extraction methods"""
        print(f"🖥️  Using device: {self.device.upper()}")
        
        if torch.cuda.is_available():
            print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            print("⚡ GPU acceleration enabled for image processing")
            
            # Set OpenCV to use GPU if available
            try:
                cv2.cuda.setDevice(0)
                self.use_gpu_cv = True
                print("✅ OpenCV GPU acceleration enabled")
            except:
                self.use_gpu_cv = False
                print("⚠️ OpenCV GPU not available, using CPU")
        else:
            print("💻 Using CPU - for better performance, install CUDA")
            self.use_gpu_cv = False
    
    def extract_pdf_with_pymupdf(self, pdf_path):
        """Extract text using PyMuPDF with better accuracy"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            print(f"   📄 Processing {len(doc)} pages with PyMuPDF...")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Try text extraction first (for searchable PDFs)
                text = page.get_text()
                
                if len(text.strip()) < 50:  # If text is minimal, use OCR
                    print(f"      🔍 Page {page_num + 1}: Using OCR (scanned content)")
                    text = self.extract_with_ocr(page, page_num)
                else:
                    print(f"      ✅ Page {page_num + 1}: Direct text extraction")
                
                full_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
            
            doc.close()
            return full_text
            
        except Exception as e:
            print(f"   ❌ PyMuPDF extraction failed: {e}")
            return None
    
    def extract_with_ocr(self, page, page_num):
        """GPU-accelerated OCR extraction"""
        try:
            # Convert PDF page to image
            mat = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x zoom for better OCR
            img_data = mat.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data))
            
            # Convert to OpenCV format
            opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # GPU-accelerated image preprocessing
            if self.use_gpu_cv:
                # Upload to GPU
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(opencv_img)
                
                # GPU-accelerated preprocessing
                gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
                gpu_denoised = cv2.cuda.fastNlMeansDenoising(gpu_gray)
                gpu_thresh = cv2.cuda.threshold(gpu_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                
                # Download from GPU
                processed_img = gpu_thresh.download()
            else:
                # CPU preprocessing
                gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
                denoised = cv2.fastNlMeansDenoising(gray)
                _, processed_img = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR with Tesseract
            custom_config = r'--oem 3 --psm 6 -l eng'
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            
            return text
            
        except Exception as e:
            print(f"      ❌ OCR failed for page {page_num + 1}: {e}")
            return ""
    
    def extract_docx_enhanced(self, docx_path):
        """Enhanced DOCX extraction"""
        try:
            from docx import Document as DocxDocument
            
            doc = DocxDocument(docx_path)
            full_text = ""
            
            print(f"   📄 Processing DOCX with enhanced extraction...")
            
            # Extract paragraphs
            for paragraph in doc.paragraphs:
                full_text += paragraph.text + "\n"
            
            # Extract tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        row_text.append(cell.text.strip())
                    full_text += " | ".join(row_text) + "\n"
            
            return full_text
            
        except ImportError:
            print("   ⚠️ python-docx not installed, using fallback method")
            return self.extract_docx_fallback(docx_path)
        except Exception as e:
            print(f"   ❌ Enhanced DOCX extraction failed: {e}")
            return self.extract_docx_fallback(docx_path)
    
    def extract_docx_fallback(self, docx_path):
        """Fallback DOCX extraction using LangChain"""
        try:
            loader = UnstructuredWordDocumentLoader(docx_path)
            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"   ❌ Fallback DOCX extraction failed: {e}")
            return ""
    
    def process_document(self, file_path):
        """Process a single document with the best available method"""
        file_ext = Path(file_path).suffix.lower()
        filename = Path(file_path).name
        
        print(f"\n🔄 Processing: {filename}")
        start_time = time.time()
        
        try:
            if file_ext == '.pdf':
                # Try PyMuPDF first (better accuracy)
                content = self.extract_pdf_with_pymupdf(file_path)
                
                if not content or len(content.strip()) < 100:
                    print("   🔄 Trying fallback PDF extraction...")
                    loader = UnstructuredPDFLoader(file_path)
                    docs = loader.load()
                    content = "\n".join([doc.page_content for doc in docs])
                    
            elif file_ext in ['.docx', '.doc']:
                content = self.extract_docx_enhanced(file_path)
                
            else:
                print(f"   ⚠️ Unsupported file type: {file_ext}")
                return None
            
            processing_time = time.time() - start_time
            
            if content and len(content.strip()) > 50:
                print(f"   ✅ Extracted {len(content)} characters in {processing_time:.2f}s")
                return content
            else:
                print(f"   ⚠️ Minimal content extracted ({len(content) if content else 0} chars)")
                return content
                
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
            return None

def extract_and_save_text_with_gpu():
    """Main extraction function with GPU acceleration"""
    start_time = time.time()
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Check if input directory exists
    if not os.path.exists(DOCS_PATH) or not os.listdir(DOCS_PATH):
        print(f"❌ Error: The '{DOCS_PATH}' directory does not exist or is empty.")
        return
    
    print("🚀 GPU-Accelerated Text Extraction Starting...")
    print("=" * 60)
    print(f"📂 Input directory: {DOCS_PATH}")
    print(f"📁 Output directory: {OUTPUT_PATH}")
    print("📝 Supported formats: PDF, DOCX, DOC")
    
    # Initialize GPU extractor
    extractor = GPUAcceleratedTextExtractor()
    
    # Get all supported files
    supported_extensions = ['.pdf', '.docx', '.doc']
    files_to_process = []
    
    for file_path in Path(DOCS_PATH).iterdir():
        if file_path.suffix.lower() in supported_extensions:
            files_to_process.append(file_path)
    
    if not files_to_process:
        print(f"❌ No supported files found in {DOCS_PATH}")
        return
    
    print(f"\n📋 Found {len(files_to_process)} files to process:")
    for file_path in files_to_process:
        print(f"   📄 {file_path.name}")
    
    # Process each file
    successful_extractions = 0
    total_chars_extracted = 0
    
    for file_path in files_to_process:
        content = extractor.process_document(file_path)
        
        if content:
            # Save extracted content
            output_filename = f"{file_path.stem}.txt"
            output_filepath = Path(OUTPUT_PATH) / output_filename
            
            try:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    # Add metadata header
                    f.write(f"# Extracted from: {file_path.name}\n")
                    f.write(f"# Extraction date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# Extraction method: GPU-Accelerated\n")
                    f.write(f"# File size: {file_path.stat().st_size} bytes\n")
                    f.write("# " + "="*50 + "\n\n")
                    f.write(content)
                
                print(f"   💾 Saved: {output_filename}")
                successful_extractions += 1
                total_chars_extracted += len(content)
                
            except Exception as e:
                print(f"   ❌ Error saving {output_filename}: {e}")
        else:
            print(f"   ⚠️ Skipped {file_path.name} (extraction failed)")
    
    # Final summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("📊 EXTRACTION SUMMARY")
    print("="*60)
    print(f"✅ Successfully processed: {successful_extractions}/{len(files_to_process)} files")
    print(f"📝 Total characters extracted: {total_chars_extracted:,}")
    print(f"⏱️ Total time: {duration:.2f} seconds")
    print(f"🚀 Average speed: {total_chars_extracted/duration:.0f} chars/second")
    
    if torch.cuda.is_available():
        print(f"💾 GPU Memory used: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
    
    print(f"📁 Output saved to: {OUTPUT_PATH}")

# Legacy function for backward compatibility
def extract_and_save_text_with_ocr():
    """Legacy function - redirects to GPU version"""
    extract_and_save_text_with_gpu()

if __name__ == '__main__':
    extract_and_save_text_with_gpu()