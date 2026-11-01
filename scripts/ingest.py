import os
import glob
import json
import hashlib
import re
import time
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

DATA_DIR = os.getenv("DATA_DIR", "./data")
CORPUS_PATH = os.path.join(DATA_DIR, "bm25_corpus.json")
MANIFEST_PATH = os.path.join(DATA_DIR, "ingestion_manifest.json")

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
    # Collapse multiple consecutive empty lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def split_text_with_overlap(text: str, chunk_size: int = 900, chunk_overlap: int = 120):
    """
    Splits text by paragraphs, merging them up to chunk_size with 10-15% overlap.
    Falls back to sentence or sliding window boundary if a single paragraph is too large.
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
            # Sub-split long paragraph by sentences or window
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if current_length + len(s) + 1 > chunk_size and current_chunk:
                    joined = " ".join(current_chunk)
                    chunks.append(joined)
                    # Overlap with end of current chunk
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

def extract_section_title(text: str, default: str = "General") -> str:
    match = SECTION_HEADING_PATTERN.search(text)
    if match:
        return match.group(0).strip()
    return default

def run_ingestion(data_dir: str = DATA_DIR):
    print("=" * 70)
    print(f"Starting Retrieva Ingestion Pipeline (Data Directory: {data_dir})")
    print("=" * 70)

    raw_docs = []

    # 1. Native pypdf / PyPDFLoader PDF loading
    pdf_files = glob.glob(os.path.join(data_dir, "**", "*.pdf"), recursive=True)
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF file(s). Extracting pages...")
        for filepath in sorted(pdf_files):
            rel_path = os.path.relpath(filepath, start=os.path.dirname(data_dir)).replace("\\", "/")
            fname = os.path.basename(filepath)
            if PYPDF_AVAILABLE:
                try:
                    reader = pypdf.PdfReader(filepath)
                    total_pages = len(reader.pages)
                    doc_pages = 0
                    for page_idx, page in enumerate(reader.pages, start=1):
                        extracted = page.extract_text() or ""
                        cleaned = clean_text(extracted)
                        if len(cleaned) >= 50:
                            raw_docs.append({
                                "content": cleaned,
                                "meta": {
                                    "source": rel_path,
                                    "filename": fname,
                                    "page": page_idx,
                                    "total_pages": total_pages
                                }
                            })
                            doc_pages += 1
                    print(f"  [PDF] {fname}: {doc_pages}/{total_pages} pages extracted")
                except Exception as e:
                    print(f"  [PDF] Warning: Failed to parse {fname}: {e}")
            elif LANGCHAIN_AVAILABLE:
                try:
                    loader = PyPDFLoader(filepath)
                    pages = loader.load()
                    for p in pages:
                        cleaned = clean_text(p.page_content)
                        if len(cleaned) >= 50:
                            raw_docs.append({
                                "content": cleaned,
                                "meta": {
                                    **p.metadata,
                                    "source": rel_path,
                                    "filename": fname
                                }
                            })
                except Exception as e:
                    print(f"  [PDF] Warning loading {fname}: {e}")

    # 2. Load TXT documents (e.g. sample_retrieva.txt or notes)
    txt_files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
    for filepath in txt_files:
        if os.path.basename(filepath) == "README.txt":
            continue
        rel_path = os.path.relpath(filepath, start=os.path.dirname(data_dir)).replace("\\", "/")
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = clean_text(f.read())
            if content:
                raw_docs.append({
                    "content": content,
                    "meta": {
                        "source": rel_path,
                        "filename": os.path.basename(filepath),
                        "page": 1,
                        "total_pages": 1
                    }
                })
        except Exception as e:
            print(f"  [TXT] Warning reading {filepath}: {e}")

    # If no documents, create sample document
    if not raw_docs:
        print("No documents found in data folder. Generating sample knowledge file...")
        sample_file = os.path.join(data_dir, "sample_retrieva.txt")
        sample_text = (
            "Retrieva (Retrieval-Augmented Generation) combines generative AI with targeted search.\n"
            "Hybrid search integrates dense vector search (embeddings) with sparse lexical search (BM25).\n"
            "Reciprocal Rank Fusion (RRF) merges dense and sparse rankings to produce optimal results.\n"
            "Prometheus monitors API latency, retrieval duration, LLM generation time, and cache hits.\n"
            "Grafana provides a unified visual dashboard for tracking latency, QPS, token usage, and user feedback.\n"
            "MLOps for RAG includes automated evaluation of Hit Rate, Context Relevance, and Groundedness.\n"
        )
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(sample_text)
        print(f"Created sample knowledge file: {sample_file}")
        raw_docs.append({
            "content": sample_text,
            "meta": {"source": "data/sample_retrieva.txt", "filename": "sample_retrieva.txt", "page": 1, "total_pages": 1}
        })

    print(f"\nProcessing {len(raw_docs)} document pages/sources for chunking...")

    # 3. Heading & Section aware chunking with 10-15% overlap
    chunks = []
    chunk_lengths = []
    global_idx = 0
    current_section = "General"

    for doc in raw_docs:
        text = doc["content"]
        meta = doc["meta"]
        
        detected_section = extract_section_title(text, default=current_section)
        current_section = detected_section

        page_chunks = split_text_with_overlap(text, chunk_size=900, chunk_overlap=120)

        for chunk_text in page_chunks:
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 40:
                continue

            chunk_id = f"chunk_{global_idx}_{hashlib.md5(chunk_text.encode('utf-8')).hexdigest()[:8]}"
            chunk_meta = {
                **meta,
                "chunk_id": chunk_id,
                "chunk_index": global_idx,
                "section": current_section,
                "char_length": len(chunk_text)
            }
            chunks.append({
                "page_content": chunk_text,
                "metadata": chunk_meta,
                "score": 0.0
            })
            chunk_lengths.append(len(chunk_text))
            global_idx += 1

    print(f"\nGenerated {len(chunks)} text chunks from {len(pdf_files)} PDFs and {len(txt_files)} text files.")
    if chunk_lengths:
        avg_len = sum(chunk_lengths) / len(chunk_lengths)
        print(f"Chunk statistics: min={min(chunk_lengths)}, max={max(chunk_lengths)}, avg={avg_len:.1f} chars")

    # 4. Persist BM25 Corpus for Hybrid Search
    os.makedirs(data_dir, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to BM25 corpus ({CORPUS_PATH})")

    # 5. Pinecone Vector Store Ingestion (if credentials provided)
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "retrieva")
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if pinecone_api_key and hf_token and LANGCHAIN_AVAILABLE:
        try:
            print("Initializing Hugging Face embeddings...")
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
            from langchain_pinecone import PineconeVectorStore
            from pinecone import Pinecone, ServerlessSpec

            embeddings = HuggingFaceEndpointEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                huggingfacehub_api_token=hf_token
            )
            pc = Pinecone(api_key=pinecone_api_key)
            if index_name not in [idx.name for idx in pc.list_indexes()]:
                print(f"Creating Pinecone serverless index: {index_name} (384 dims, cosine)...")
                pc.create_index(
                    name=index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=os.getenv("PINECONE_CLOUD", "aws"),
                        region=os.getenv("PINECONE_REGION", "us-east-1")
                    )
                )
                time.sleep(2)

            print("Upserting dense vectors to Pinecone...")
            from langchain_core.documents import Document
            lc_chunks = [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in chunks]
            PineconeVectorStore.from_documents(lc_chunks, embeddings, index_name=index_name)
            print("Successfully indexed vectors in Pinecone.")
        except Exception as e:
            print(f"Pinecone vector ingestion skipped or failed: {e}")
    else:
        print("Note: PINECONE_API_KEY / HUGGINGFACEHUB_API_TOKEN not set or local mode. Saved to BM25 corpus.")

    # 6. Save Ingestion Manifest
    manifest = {
        "version": "2.0",
        "ingested_at": datetime.now(datetime.UTC).isoformat() if hasattr(datetime, "UTC") else datetime.utcnow().isoformat(),
        "total_pdf_documents": len(pdf_files),
        "total_documents": len(raw_docs),
        "total_chunks": len(chunks),
        "corpus_file": CORPUS_PATH,
        "pinecone_index": index_name,
        "chunk_statistics": {
            "min_chars": min(chunk_lengths) if chunk_lengths else 0,
            "max_chars": max(chunk_lengths) if chunk_lengths else 0,
            "avg_chars": round(sum(chunk_lengths) / len(chunk_lengths), 1) if chunk_lengths else 0
        }
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Ingestion manifest written to {MANIFEST_PATH}")
    print("Ingestion complete.")
    return len(chunks)

if __name__ == "__main__":
    run_ingestion()
