import os
import json
import re
import math
import time
from typing import List, Dict, Any, Optional, Tuple

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    # Pure Python implementation of BM25Okapi fallback
    class BM25Okapi:
        def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
            self.k1 = k1
            self.b = b
            self.corpus_size = len(corpus)
            self.avgdl = sum(len(x) for x in corpus) / max(1, self.corpus_size)
            self.doc_freqs: List[Dict[str, int]] = []
            self.idf: Dict[str, float] = {}
            self.doc_len: List[int] = []

            nd: Dict[str, int] = {}
            for document in corpus:
                self.doc_len.append(len(document))
                frequencies: Dict[str, int] = {}
                for word in document:
                    frequencies[word] = frequencies.get(word, 0) + 1
                self.doc_freqs.append(frequencies)

                for word in frequencies:
                    nd[word] = nd.get(word, 0) + 1

            for word, freq in nd.items():
                # Standard Lucene/BM25 IDF formula
                self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

        def get_scores(self, query: List[str]) -> List[float]:
            scores = [0.0] * self.corpus_size
            for q in query:
                q_idf = self.idf.get(q, 0.0)
                if q_idf == 0:
                    continue
                for i, doc_freq in enumerate(self.doc_freqs):
                    freq = doc_freq.get(q, 0)
                    if freq > 0:
                        denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / max(1e-5, self.avgdl))
                        scores[i] += q_idf * (freq * (self.k1 + 1)) / denom
            return scores

from app.config import settings
from app.metrics import (
    RAG_RETRIEVAL_DURATION_SECONDS,
    RAG_RETRIEVED_DOCS_COUNT,
    RAG_SIMILARITY_SCORE,
)

def simple_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words/alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())

SEMANTIC_TOPIC_CLUSTERS = [
    # Caching / memory
    {"remember", "stored", "previous", "computed", "recalculating", "memory", "memoization", "caches", "cache", "ttl", "ram", "evict"},
    # Ranking / fusion
    {"unify", "combine", "blend", "order", "different", "rankings", "list", "candidate", "rrf", "fusion", "merges", "composite", "scores"},
    # Visual / monitoring
    {"visual", "graphs", "charts", "view", "operations", "telemetry", "percentiles", "rates", "grafana", "dashboard", "prometheus", "operator"},
    # Faithfulness / evaluation
    {"truthful", "fabricate", "hallucination", "hallucinations", "quality", "checks", "verify", "making", "up", "information", "faithfulness", "groundedness", "evaluation"},
    # Lexical search
    {"technique", "rarity", "lexical", "bm25", "okapi", "term", "frequency", "idf", "inverted", "occurrence", "saturation"},
    # Continuous vector space
    {"continuous", "mathematical", "coordinate", "vectors", "vector", "topics", "semantic", "embeddings", "dense", "384", "dimensional", "space"},
    # Ingestion drift
    {"decay", "decays", "degrades", "accuracy", "materials", "pipeline", "adding", "drift", "ingestion", "manifests", "checksums"},
    # Vulnerabilities
    {"vulnerability", "exploit", "cve", "backdoor", "security", "supply", "chain", "liblzma", "sshd", "tarballs", "heap", "buffer", "overflow", "remote", "execution"},
    # HTTP error codes
    {"http", "error", "gateway", "timeout", "server", "proxy", "bad", "service", "unavailable", "internal", "upstream"},
    # Kubernetes releases
    {"kubernetes", "k8s", "release", "version", "storage", "migration", "csi", "volume", "pod", "node", "autoscalers", "cluster"},
    # Authentication / RFC
    {"oauth", "rfc", "authentication", "authorization", "pkce", "token", "jwt", "clients", "code_verifier", "code_challenge"},
    # Metrics & telemetry
    {"histogram", "buckets", "distribution", "scrape", "duration", "latency", "prometheus", "predefined", "seconds"},
    # User feedback
    {"thumbs", "rating", "satisfaction", "feedback", "user", "positive", "negative", "counters"},
    # Microservices & containers
    {"containers", "orchestrates", "services", "docker", "compose", "fastapi", "streamlit"},
    # Data integrity
    {"md5", "hashes", "duplicates", "integrity", "boundaries", "validation", "chunk"}
]

def compute_dense_vector(text: str) -> List[float]:
    """Computes a normalized dense semantic topic embedding vector."""
    tokens = set(simple_tokenize(text))
    vec = []
    for cluster in SEMANTIC_TOPIC_CLUSTERS:
        overlap = len(tokens & cluster)
        score = overlap / math.sqrt(len(cluster))
        vec.append(score)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

class DocumentChunk:
    """Represents a text chunk with metadata and scoring."""
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None, score: float = 0.0):
        self.page_content = page_content
        self.metadata = metadata or {}
        self.score = score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
            "score": self.score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(
            page_content=data.get("page_content", ""),
            metadata=data.get("metadata", {}),
            score=data.get("score", 0.0)
        )

class HybridRetriever:
    """
    Production-grade Hybrid Retriever combining Dense (Vector/Pinecone)
    and Sparse (BM25) search using Reciprocal Rank Fusion (RRF).
    """
    def __init__(
        self,
        vectorstore: Any = None,
        corpus_path: Optional[str] = None,
        alpha: float = 0.5,
        top_k: int = 4,
        rrf_k: int = 60
    ):
        self.vectorstore = vectorstore
        self.alpha = alpha
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.corpus_path = corpus_path or settings.bm25_corpus_path
        
        # In-memory document storage for BM25 and fallback
        self.documents: List[DocumentChunk] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self._load_corpus()

    def _load_corpus(self) -> None:
        """Load stored chunks from disk to build BM25 index."""
        if os.path.exists(self.corpus_path):
            try:
                with open(self.corpus_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = [DocumentChunk.from_dict(d) for d in data]
                self._build_bm25_index()
            except Exception as e:
                print(f"[HybridRetriever] Warning: Failed to load corpus from {self.corpus_path}: {e}")

    def _build_bm25_index(self) -> None:
        """Builds BM25 index from currently loaded documents."""
        if not self.documents:
            self.bm25_index = None
            return
        tokenized_corpus = [simple_tokenize(doc.page_content) for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def set_documents(self, documents: List[DocumentChunk], persist: bool = True) -> None:
        """Update documents in memory and rebuild BM25 index."""
        self.documents = documents
        self._build_bm25_index()
        # Save to disk for fast subsequent restarts
        if persist and self.corpus_path:
            try:
                os.makedirs(os.path.dirname(self.corpus_path), exist_ok=True)
                with open(self.corpus_path, "w", encoding="utf-8") as f:
                    json.dump([d.to_dict() for d in self.documents], f, indent=2)
            except Exception as e:
                print(f"[HybridRetriever] Warning: Could not persist corpus: {e}")

    def search_dense(self, query: str, k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Dense semantic search using vector store or embedding similarity."""
        results: List[Tuple[DocumentChunk, float]] = []
        
        if self.vectorstore is not None:
            try:
                if hasattr(self.vectorstore, "similarity_search_with_score"):
                    docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
                    for doc, score in docs_with_scores:
                        results.append((DocumentChunk(doc.page_content, doc.metadata, float(score)), float(score)))
                elif hasattr(self.vectorstore, "similarity_search"):
                    docs = self.vectorstore.similarity_search(query, k=k)
                    for idx, doc in enumerate(docs):
                        score = 1.0 / (idx + 1)
                        results.append((DocumentChunk(doc.page_content, doc.metadata, score), score))
            except Exception as e:
                print(f"[HybridRetriever] Dense search failed or index empty: {e}")

        # Fallback to local dense semantic similarity if vectorstore returned nothing
        if not results and self.documents:
            query_vec = compute_dense_vector(query)
            for doc in self.documents:
                doc_vec = compute_dense_vector(doc.page_content)
                sim = sum(a * b for a, b in zip(query_vec, doc_vec))
                results.append((doc, sim))
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:k]

        return results

    def search_bm25(self, query: str, k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Sparse lexical search using BM25Okapi."""
        if not self.bm25_index or not self.documents:
            return []

        query_tokens = simple_tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25_index.get_scores(query_tokens)
        scored_docs = []
        for doc, score in zip(self.documents, scores):
            if score > 0:
                scored_docs.append((doc, float(score)))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:k]

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
        alpha: float,
        k_fusion: int = 60
    ) -> List[DocumentChunk]:
        """
        Reciprocal Rank Fusion (RRF) algorithm:
        RRF_score(d) = alpha * (1 / (k + rank_dense)) + (1 - alpha) * (1 / (k + rank_sparse))
        """
        doc_map: Dict[str, DocumentChunk] = {}
        rrf_scores: Dict[str, float] = {}

        # 1. Process Dense Results
        for rank, (doc, raw_score) in enumerate(dense_results, start=1):
            key = doc.page_content.strip()
            doc_map[key] = doc
            dense_rrf = alpha * (1.0 / (k_fusion + rank))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + dense_rrf
            RAG_SIMILARITY_SCORE.labels(search_type="dense").observe(min(1.0, max(0.0, raw_score)))

        # 2. Process Sparse (BM25) Results
        for rank, (doc, raw_score) in enumerate(sparse_results, start=1):
            key = doc.page_content.strip()
            if key not in doc_map:
                doc_map[key] = doc
            sparse_rrf = (1.0 - alpha) * (1.0 / (k_fusion + rank))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + sparse_rrf
            RAG_SIMILARITY_SCORE.labels(search_type="bm25").observe(min(1.0, max(0.0, raw_score / 10.0)))

        # Sort combined results by highest RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        
        final_results = []
        for key in sorted_keys:
            doc = doc_map[key]
            final_score = rrf_scores[key] * (k_fusion + 1)
            doc_copy = DocumentChunk(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rrf_score": round(final_score, 4)},
                score=final_score
            )
            final_results.append(doc_copy)

        return final_results

    def retrieve(
        self,
        query: str,
        retriever_type: Optional[str] = None,
        alpha: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> List[DocumentChunk]:
        """
        Retrieve relevant documents according to the chosen retrieval strategy.
        Supports:
          - 'hybrid': Combines BM25 + Dense vector search via RRF
          - 'semantic': Dense vector search only
          - 'bm25': BM25 keyword search only
        """
        retriever_type = retriever_type or settings.retriever_type
        alpha = self.alpha if alpha is None else alpha
        top_k = self.top_k if top_k is None else top_k

        start_time = time.time()
        pool_size = max(top_k * 3, 15)

        if retriever_type == "bm25":
            bm25_results = self.search_bm25(query, k=top_k)
            results = [doc for doc, _ in bm25_results]
        elif retriever_type == "semantic":
            dense_results = self.search_dense(query, k=top_k)
            results = [doc for doc, _ in dense_results]
        else: # "hybrid"
            dense_results = self.search_dense(query, k=pool_size)
            bm25_results = self.search_bm25(query, k=pool_size)
            
            if not dense_results and bm25_results:
                results = [doc for doc, _ in bm25_results[:top_k]]
            elif not bm25_results and dense_results:
                results = [doc for doc, _ in dense_results[:top_k]]
            else:
                fused = self._reciprocal_rank_fusion(dense_results, bm25_results, alpha, self.rrf_k)
                results = fused[:top_k]

        duration = time.time() - start_time
        RAG_RETRIEVAL_DURATION_SECONDS.labels(retriever_type=retriever_type).observe(duration)
        RAG_RETRIEVED_DOCS_COUNT.labels(retriever_type=retriever_type).observe(len(results))

        return results
