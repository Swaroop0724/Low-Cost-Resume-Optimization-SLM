"""Extract text from uploaded resume files (PDF, DOCX, TXT)."""
from fastapi import UploadFile
from pathlib import Path
import io

async def extract_text_from_upload(file: UploadFile) -> str:
    content = await file.read()
    ext = Path(file.filename).suffix.lower()
    if ext == ".txt":   return content.decode("utf-8", errors="ignore")
    elif ext == ".pdf": return _pdf(content)
    elif ext in (".docx", ".doc"): return _docx(content)
    return content.decode("utf-8", errors="ignore")

def _pdf(content):
    from PyPDF2 import PdfReader
    r = PdfReader(io.BytesIO(content))
    return "\n".join((p.extract_text() or "") for p in r.pages).strip()

def _docx(content):
    from docx import Document
    return "\n".join(p.text for p in Document(io.BytesIO(content)).paragraphs).strip()
