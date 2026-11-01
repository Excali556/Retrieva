import io
import os
import re
import csv
import json
import time
import zipfile
import hashlib
from typing import List, Dict, Any, Tuple
from datetime import datetime

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

from app.retrieval.hybrid_retriever import DocumentChunk, HybridRetriever
from app.cache import rag_cache

SECTION_HEADING_PATTERN = re.compile(
    r"^(?:(?:[0-9]{1,2}(?:\.[0-9]{1,2})*\s+[A-Z][A-Za-z0-9\s\-]{2,50})|"
    r"(?:Abstract|Introduction|Background|Methodology|Methods|Model\s+Architecture|"
    r"Attention|Experiments|Results|Discussion|Related\s+Work|Conclusion|References))\s*$",
    re.MULTILINE
)

def clean_text(text: str) -> str:
    """Normalize whitespace and strip unprintable characters."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_text_with_overlap(text: str, chunk_size: int = 900, chunk_overlap: int = 120) -> List[str]:
    """
    Splits text by paragraphs, merging them up to chunk_size with overlap.
    Falls back to sentence or sliding window boundary if a paragraph is too large.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)
        if para_len > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if current_length + len(s) + 1 > chunk_size and current_chunk:
                    joined = " ".join(current_chunk)
                    chunks.append(joined)
                    overlap_chars = joined[-chunk_overlap:] if len(joined) > chunk_overlap else joined
                    current_chunk = [overlap_chars, s]
                    current_length = len(overlap_chars) + len(s) + 1
                else:
                    current_chunk.append(s)
                    current_length += len(s) + 1
        elif current_length + para_len + 2 > chunk_size and current_chunk:
            joined = "\n\n".join(current_chunk)
            chunks.append(joined)
            overlap_chars = joined[-chunk_overlap:] if len(joined) > chunk_overlap else joined
            current_chunk = [overlap_chars, para]
            current_length = len(overlap_chars) + len(para) + 2
        else:
            current_chunk.append(para)
            current_length += para_len + 2

    if current_chunk:
        joined = "\n\n".join(current_chunk)
        if joined.strip():
            chunks.append(joined.strip())

    return chunks

def extract_text_from_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract pages from PDF bytes using pypdf."""
    pages_data = []
    if not PYPDF_AVAILABLE:
        # Fallback raw string extraction
        raw_text = extract_printable_strings(file_bytes)
        if raw_text:
            pages_data.append({"text": raw_text, "page": 1, "total_pages": 1})
        return pages_data

    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            cleaned = clean_text(text)
            if cleaned:
                pages_data.append({
                    "text": cleaned,
                    "page": page_idx,
                    "total_pages": total_pages
                })
    except Exception as e:
        print(f"[DocumentProcessor] PDF parse warning: {e}")
        raw_text = extract_printable_strings(file_bytes)
        if raw_text:
            pages_data.append({"text": raw_text, "page": 1, "total_pages": 1})

    return pages_data

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract full text from DOCX bytes without external dependencies."""
    try:
        import xml.etree.ElementTree as ET
        with io.BytesIO(file_bytes) as bio:
            with zipfile.ZipFile(bio) as zf:
                if "word/document.xml" not in zf.namelist():
                    return extract_printable_strings(file_bytes)
                xml_content = zf.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                paragraphs = []
                for p in tree.iterfind(".//w:p", namespaces):
                    p_text = "".join(node.text for node in p.iterfind(".//w:t", namespaces) if node.text)
                    if p_text.strip():
                        paragraphs.append(p_text.strip())
                return "\n\n".join(paragraphs)
    except Exception as e:
        print(f"[DocumentProcessor] DOCX parse warning: {e}")
        return extract_printable_strings(file_bytes)

def extract_text_from_pptx(file_bytes: bytes) -> str:
    """Extract text from PPTX slides."""
    try:
        import xml.etree.ElementTree as ET
        slides_text = []
        with io.BytesIO(file_bytes) as bio:
            with zipfile.ZipFile(bio) as zf:
                slide_files = sorted([f for f in zf.namelist() if f.startswith("ppt/slides/slide") and f.endswith(".xml")])
                namespaces = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
                for s_file in slide_files:
                    xml_content = zf.read(s_file)
                    tree = ET.fromstring(xml_content)
                    texts = [node.text for node in tree.iterfind(".//a:t", namespaces) if node.text]
                    if texts:
                        slides_text.append(" ".join(texts))
        return "\n\n".join(slides_text) if slides_text else extract_printable_strings(file_bytes)
    except Exception:
        return extract_printable_strings(file_bytes)

def extract_text_from_csv(file_bytes: bytes) -> str:
    """Format CSV into informative text rows for retrieval."""
    try:
        decoded = decode_text_bytes(file_bytes)
        reader = csv.reader(io.StringIO(decoded))
        rows = list(reader)
        if not rows:
            return ""
        headers = rows[0]
        formatted = []
        for row_idx, row in enumerate(rows[1:], start=1):
            row_items = []
            for col_idx, val in enumerate(row):
                header = headers[col_idx] if col_idx < len(headers) else f"Col{col_idx+1}"
                if val.strip():
                    row_items.append(f"{header}: {val.strip()}")
            if row_items:
                formatted.append(f"Row {row_idx}: " + " | ".join(row_items))
        return "\n".join(formatted)
    except Exception:
        return decode_text_bytes(file_bytes)

def decode_text_bytes(file_bytes: bytes) -> str:
    """Safely decode text bytes using progressive encoding fallbacks."""
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")

def extract_printable_strings(file_bytes: bytes, min_len: int = 4) -> str:
    """Fallback text extraction for arbitrary binary files."""
    try:
        decoded = file_bytes.decode("utf-8", errors="ignore")
        tokens = re.findall(r"[\x20-\x7E\t\n]{4,}", decoded)
        return "\n".join(t.strip() for t in tokens if t.strip())
    except Exception:
        return ""

def parse_document_to_chunks(
    filename: str,
    file_bytes: bytes,
    chunk_size: int = 900,
    chunk_overlap: int = 120
) -> List[DocumentChunk]:
    """
    Accepts any document type, extracts text, and converts into DocumentChunks.
    Supports PDF, DOCX, PPTX, TXT, MD, CSV, JSON, HTML, and all arbitrary text/code formats.
    """
    ext = os.path.splitext(filename)[1].lower()
    raw_pages: List[Dict[str, Any]] = []

    if ext == ".pdf":
        raw_pages = extract_text_from_pdf(file_bytes)
    elif ext in [".docx", ".doc"]:
        full_text = extract_text_from_docx(file_bytes)
        if full_text:
            raw_pages.append({"text": clean_text(full_text), "page": 1, "total_pages": 1})
    elif ext in [".pptx", ".ppt"]:
        full_text = extract_text_from_pptx(file_bytes)
        if full_text:
            raw_pages.append({"text": clean_text(full_text), "page": 1, "total_pages": 1})
    elif ext in [".csv", ".tsv"]:
        full_text = extract_text_from_csv(file_bytes)
        if full_text:
            raw_pages.append({"text": clean_text(full_text), "page": 1, "total_pages": 1})
    elif ext == ".json":
        try:
            parsed = json.loads(decode_text_bytes(file_bytes))
            full_text = json.dumps(parsed, indent=2)
        except Exception:
            full_text = decode_text_bytes(file_bytes)
        raw_pages.append({"text": clean_text(full_text), "page": 1, "total_pages": 1})
    else:
        # Default text / markdown / code / unknown fallback
        full_text = decode_text_bytes(file_bytes)
        cleaned = clean_text(full_text)
        if len(cleaned) < 20:
            cleaned = clean_text(extract_printable_strings(file_bytes))
        if cleaned:
            raw_pages.append({"text": cleaned, "page": 1, "total_pages": 1})

    chunks: List[DocumentChunk] = []
    upload_timestamp = datetime.now().isoformat()

    chunk_counter = 0
    for page_info in raw_pages:
        text = page_info["text"]
        page_num = page_info.get("page", 1)
        total_pages = page_info.get("total_pages", 1)

        page_chunks = split_text_with_overlap(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for c_text in page_chunks:
            c_text = c_text.strip()
            if len(c_text) < 30:
                continue

            chunk_id = f"upload_{filename}_{chunk_counter}_{hashlib.md5(c_text.encode('utf-8')).hexdigest()[:8]}"
            chunk_counter += 1

            meta = {
                "source": filename,
                "filename": filename,
                "file_type": ext,
                "page": page_num,
                "total_pages": total_pages,
                "chunk_id": chunk_id,
                "uploaded_at": upload_timestamp,
                "char_length": len(c_text)
            }
            chunks.append(DocumentChunk(page_content=c_text, metadata=meta, score=0.0))

    return chunks

def ingest_uploaded_files(
    files: List[Tuple[str, bytes]],
    retriever: HybridRetriever,
    save_dir: str = "./data/uploads",
    corpus_path: str = "./data/bm25_corpus.json"
) -> Dict[str, Any]:
    """
    Ingests multiple uploaded files into the retriever in-memory index and persists to disk.
    Immediately updates BM25Okapi and resets cache.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_new_chunks: List[DocumentChunk] = []
    file_summaries = []

    for filename, content in files:
        safe_filename = os.path.basename(filename)
        dest_path = os.path.join(save_dir, safe_filename)
        
        # Save uploaded file
        try:
            with open(dest_path, "wb") as f:
                f.write(content)
        except Exception as e:
            print(f"[DocumentProcessor] Failed saving {dest_path}: {e}")

        # Parse into chunks
        try:
            chunks = parse_document_to_chunks(safe_filename, content)
            all_new_chunks.extend(chunks)
            file_summaries.append({
                "filename": safe_filename,
                "chunks": len(chunks),
                "size_bytes": len(content),
                "status": "success" if chunks else "no_extractable_text"
            })
        except Exception as e:
            file_summaries.append({
                "filename": safe_filename,
                "chunks": 0,
                "size_bytes": len(content),
                "status": f"error: {str(e)}"
            })

    if all_new_chunks:
        # Incrementally update retriever in memory and persist
        if hasattr(retriever, "add_documents"):
            retriever.add_documents(all_new_chunks, persist=True)
        else:
            updated_corpus = list(retriever.documents) + all_new_chunks
            retriever.set_documents(updated_corpus, persist=True)

        # Invalidate RAG cache to ensure new documents are immediately queryable
        try:
            rag_cache.clear()
        except Exception:
            pass

    return {
        "status": "success",
        "files_processed": len(files),
        "new_chunks_count": len(all_new_chunks),
        "total_corpus_chunks": len(retriever.documents),
        "details": file_summaries
    }
