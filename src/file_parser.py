"""
file_parser.py — Extracts text from PDF, DOCX, and PPTX files.
Returns a list of document dicts: { text, source, page/slide }
"""

import os
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.chunking.title import chunk_by_title

def parse_pdf(file_path: str) -> list[dict]:
    """Extract text page-by-page from a PDF file. Falls back to OCR for scanned pages."""
    docs = []
    pdf = fitz.open(file_path)
    filename = os.path.basename(file_path)

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text().strip()
        
        # If less than 50 chars natively, assume it's a scanned page / image
        if len(text) < 50:
            pix = page.get_pixmap(dpi=300) # high res for OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            try:
                ocr_text = pytesseract.image_to_string(img)
                text = f"{text}\n{ocr_text}".strip()
            except Exception as e:
                print(f"OCR failed on {filename} page {page_num+1}: {e}")
                
        if text:
            docs.append({
                "text": text,
                "source": filename,
                "page": page_num + 1,
                "type": "pdf"
            })

    pdf.close()
    return docs

def parse_docx(file_path: str) -> list[dict]:
    """Extract text semantically from a DOCX file using Unstructured."""
    filename = os.path.basename(file_path)
    docs = []
    
    try:
        elements = partition_docx(filename=file_path)
        chunks = chunk_by_title(elements) # Group into semantic parent documents
        
        for i, chunk in enumerate(chunks, 1):
            if chunk.text.strip():
                docs.append({
                    "text": chunk.text,
                    "source": filename,
                    "page": i, # treat chunk ID as 'page' conceptually for reference
                    "type": "docx"
                })
    except Exception as e:
        print(f"Failed to parse docx {filename}: {e}")

    return docs

def parse_pptx(file_path: str) -> list[dict]:
    """Extract text semantically from a PPTX file using Unstructured."""
    filename = os.path.basename(file_path)
    docs = []
    
    try:
        elements = partition_pptx(filename=file_path)
        chunks = chunk_by_title(elements) # Group into semantic parent documents
        
        for i, chunk in enumerate(chunks, 1):
            if chunk.text.strip():
                docs.append({
                    "text": chunk.text,
                    "source": filename,
                    "page": chunk.metadata.page_number or i,
                    "type": "pptx"
                })
    except Exception as e:
        print(f"Failed to parse pptx {filename}: {e}")

    return docs

def parse_file(file_path: str) -> list[dict]:
    """Parse a file based on its extension. Returns list of document dicts."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext in (".ppt", ".pptx"):
        return parse_pptx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
