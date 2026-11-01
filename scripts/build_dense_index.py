import os
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer

def build_dense_embeddings(corpus_path="data/bm25_corpus.json", output_path="data/dense_embeddings.npy"):
    if not os.path.exists(corpus_path):
        print(f"Corpus file not found: {corpus_path}")
        return

    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")

    print("Loading SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [c["page_content"] for c in chunks]
    print(f"Encoding {len(texts)} chunks in batches of 64...")
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    duration = time.time() - t0
    print(f"Encoding completed in {duration:.2f}s ({len(texts)/duration:.1f} chunks/sec).")

    np.save(output_path, embeddings.astype(np.float32))
    print(f"Saved dense embeddings to {output_path} (shape: {embeddings.shape}, size: {os.path.getsize(output_path)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    build_dense_embeddings()
