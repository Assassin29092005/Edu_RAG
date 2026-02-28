"""
file_parser.py — Extracts text from PDF, DOCX, and PPTX files.
Returns a list of document dicts: { text, source, page/slide }
"""

import os
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.chunking.title import chunk_by_title

from src.vision_utils import summarize_image

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
        
        # Extract images from PDF and summarize them using Multimodal Vision
        try:
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                summary = summarize_image(image_bytes)
                if summary:
                    text += summary
        except Exception as e:
             print(f"PDF Image extraction failed: {e}")
                
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
    """Extract text semantically from a PPTX file and summarize images."""
    filename = os.path.basename(file_path)
    docs = []
    
    # 1. Extract image summaries per slide using python-pptx
    slide_summaries = {}
    try:
        prs = Presentation(file_path)
        for i, slide in enumerate(prs.slides, 1):
            slide_summaries[i] = ""
            for shape in slide.shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    image_bytes = shape.image.blob
                    summary = summarize_image(image_bytes)
                    if summary:
                        slide_summaries[i] += summary
    except Exception as e:
        print(f"Error parsing PPTX images {filename}: {e}")

    # 2. Extract semantic text using Unstructured
    try:
        from unstructured.partition.pptx import partition_pptx
        from unstructured.chunking.title import chunk_by_title
        from unstructured.documents.elements import Text
        
        elements = partition_pptx(filename=file_path)
        
        # Inject slide summaries as raw Unstructured elements before chunking
        # so they get naturally grouped by title and retain correct page metadata
        injected_elements = []
        
        # Track which slides we've injected from the PPTX to ensure none are missed
        visited_pages = set()
        
        for el in elements:
            # We add the original element
            injected_elements.append(el)
            
            # If this is the first time we see this page number, inject its images immediately after
            page_num = el.metadata.page_number
            if page_num and page_num not in visited_pages:
                visited_pages.add(page_num)
                if page_num in slide_summaries and slide_summaries[page_num]:
                    summary_element = Text(text=slide_summaries[page_num])
                    summary_element.metadata.page_number = page_num
                    injected_elements.append(summary_element)
                    
        # If unstructured completely skipped a slide (e.g. only had an image, no text)
        # We still need to inject that image summary as its own standalone slide/element
        for page_num, summary in slide_summaries.items():
            if page_num not in visited_pages and summary:
                summary_element = Text(text=summary)
                summary_element.metadata.page_number = page_num
                injected_elements.append(summary_element)
                
        # Now chunk the mixed text + image summary elements
        chunks = chunk_by_title(injected_elements)
        
        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.text.strip()
            page_num = chunk.metadata.page_number or i
                
            if chunk_text.strip():
                docs.append({
                    "text": chunk_text,
                    "source": filename,
                    "page": page_num,
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
