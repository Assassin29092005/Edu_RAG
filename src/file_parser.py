"""
file_parser.py — Extracts text from PDF, DOCX, and PPTX files.
Returns a list of document dicts: { text, source, page/slide }
"""

import os
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from pptx import Presentation


def parse_pdf(file_path: str) -> list[dict]:
    """Extract text page-by-page from a PDF file."""
    docs = []
    pdf = fitz.open(file_path)
    filename = os.path.basename(file_path)

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text().strip()
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
    """Extract text from paragraphs and tables in a DOCX file."""
    doc = DocxDocument(file_path)
    filename = os.path.basename(file_path)
    full_text_parts = []

    # Extract paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            full_text_parts.append(para.text.strip())

    # Extract tables
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                full_text_parts.append(row_text)

    full_text = "\n\n".join(full_text_parts)

    if full_text:
        return [{
            "text": full_text,
            "source": filename,
            "page": 1,
            "type": "docx"
        }]
    return []


def parse_pptx(file_path: str) -> list[dict]:
    """Extract text slide-by-slide from a PPTX file, including speaker notes."""
    docs = []
    prs = Presentation(file_path)
    filename = os.path.basename(file_path)

    for slide_num, slide in enumerate(prs.slides, start=1):
        text_parts = []

        # Extract text from shapes
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    para_text = paragraph.text.strip()
                    if para_text:
                        text_parts.append(para_text)

        # Extract speaker notes
        if slide.has_notes_slide:
            notes_frame = slide.notes_slide.notes_text_frame
            if notes_frame:
                notes_text = notes_frame.text.strip()
                if notes_text:
                    text_parts.append(f"[Speaker Notes]: {notes_text}")

        slide_text = "\n".join(text_parts)
        if slide_text:
            docs.append({
                "text": slide_text,
                "source": filename,
                "page": slide_num,
                "type": "pptx"
            })

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
