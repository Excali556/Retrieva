import os
import glob
import json
import hashlib
import time
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

DATA_DIR = os.getenv("DATA_DIR", "./data")
CORPUS_PATH = os.path.join(DATA_DIR, "bm25_corpus.json")
MANIFEST_PATH = os.path.join(DATA_DIR, "ingestion_manifest.json")

def fallback_chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Splits by distinct non-empty knowledge items or sliding window if LangChain is not installed."""
    lines = [line.strip() for line in text.split("\n") if line.strip() and not line.strip().startswith("#")]
    if len(lines) >= 3 and all(len(l) > 25 for l in lines):
        return lines
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += (chunk_size - chunk_overlap)
    return chunks

def run_ingestion(data_dir: str = DATA_DIR):
    print("=" * 60)
    print(f"Starting RAG Ingestion Pipeline (Data Directory: {data_dir})")
    print("=" * 60)

    raw_docs = []

    if LANGCHAIN_AVAILABLE:
        # Load PDFs
        try:
            pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
            pdf_docs = pdf_loader.load()
            raw_docs.extend([{"content": d.page_content, "meta": d.metadata} for d in pdf_docs])
            print(f"Loaded {len(pdf_docs)} PDF document pages.")
        except Exception as e:
            print(f"Warning loading PDFs: {e}")

        # Load TXT files
        try:
            txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
            txt_docs = [d for d in txt_loader.load() if not d.metadata.get("source", "").endswith("README.txt")]
            raw_docs.extend([{"content": d.page_content, "meta": d.metadata} for d in txt_docs])
            if txt_docs:
                print(f"Loaded {len(txt_docs)} text documents.")
        except Exception as e:
            print(f"Warning loading text files: {e}")
    else:
        # Fallback raw loader
        for filepath in glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True):
            if os.path.basename(filepath) == "README.txt":
                continue
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                raw_docs.append({"content": f.read(), "meta": {"source": filepath}})

    # If no documents, create sample document
    if not raw_docs:
        print("No documents found in data folder. Generating sample knowledge file...")
        sample_file = os.path.join(data_dir, "sample_autonomous_rag.txt")
        sample_text = (
            "Autonomous RAG (Retrieval-Augmented Generation) combines generative AI with targeted search.\n"
            "Hybrid search integrates dense vector search (embeddings) with sparse lexical search (BM25).\n"
            "Reciprocal Rank Fusion (RRF) merges dense and sparse rankings to produce optimal results.\n"
            "Prometheus monitors API latency, retrieval duration, LLM generation time, and cache hits.\n"
            "Grafana provides a unified visual dashboard for tracking latency, QPS, token usage, and user feedback.\n"
            "MLOps for RAG includes automated evaluation of Hit Rate, Context Relevance, and Groundedness.\n"
        )
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(sample_text)
        print(f"Created sample knowledge file: {sample_file}")
        raw_docs.append({"content": sample_text, "meta": {"source": sample_file}})

    # Chunk documents
    chunks = []
    chunk_lengths = []

    if LANGCHAIN_AVAILABLE:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        from langchain_core.documents import Document
        lc_docs = [Document(page_content=d["content"], metadata=d["meta"]) for d in raw_docs]
        split_docs = text_splitter.split_documents(lc_docs)
        for idx, doc in enumerate(split_docs):
            content = doc.page_content.strip()
            if not content:
                continue
            chunk_id = f"chunk_{idx}_{hashlib.md5(content.encode('utf-8')).hexdigest()[:8]}"
            meta = {**doc.metadata, "chunk_id": chunk_id, "chunk_index": idx, "char_length": len(content)}
            chunks.append({"page_content": content, "metadata": meta, "score": 0.0})
            chunk_lengths.append(len(content))
    else:
        idx = 0
        for doc in raw_docs:
            for text_chunk in fallback_chunk_text(doc["content"]):
                chunk_id = f"chunk_{idx}_{hashlib.md5(text_chunk.encode('utf-8')).hexdigest()[:8]}"
                meta = {**doc["meta"], "chunk_id": chunk_id, "chunk_index": idx, "char_length": len(text_chunk)}
                chunks.append({"page_content": text_chunk, "metadata": meta, "score": 0.0})
                chunk_lengths.append(len(text_chunk))
                idx += 1

    print(f"Generated {len(chunks)} text chunks.")
    if chunk_lengths:
        avg_len = sum(chunk_lengths) / len(chunk_lengths)
        print(f"Chunk statistics: min={min(chunk_lengths)}, max={max(chunk_lengths)}, avg={avg_len:.1f} chars")

    # Persist BM25 Corpus for Hybrid Search
    os.makedirs(data_dir, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to BM25 corpus ({CORPUS_PATH})")

    # Pinecone Vector Store Ingestion (if credentials provided)
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "autonomous-rag")
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

    # Save Ingestion Manifest
    manifest = {
        "version": "1.0",
        "ingested_at": datetime.now(datetime.UTC).isoformat() if hasattr(datetime, "UTC") else datetime.utcnow().isoformat(),
        "total_documents": len(raw_docs),
        "total_chunks": len(chunks),
        "corpus_file": CORPUS_PATH,
        "pinecone_index": index_name
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Ingestion manifest written to {MANIFEST_PATH}")
    print("Ingestion complete.")

if __name__ == "__main__":
    run_ingestion()
