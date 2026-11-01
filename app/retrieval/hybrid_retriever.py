import os
import json
import re
import math
import time
from typing import List, Dict, Any, Optional, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

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
    {"attention", "transformer", "dmodel", "query", "key", "value", "softmax", "multi-head", "heads", "encoder", "decoder"},
    {"retrieval", "augmented", "generation", "rag", "dpr", "passages", "generator", "seq2seq", "bart"},
    {"bert", "pretraining", "bidirectional", "masked", "mlm", "nsp", "classification"},
    {"lora", "low-rank", "adaptation", "rank", "matrix", "decomposition", "delta", "weights", "finetuning"},
    {"flashattention", "sram", "hbm", "tiling", "io-aware", "bandwidth", "memory", "exact"},
    {"react", "reasoning", "acting", "thought", "action", "observation", "trajectory", "agent"},
    {"consensus", "raft", "leader", "election", "log", "replication", "term", "heartbeat", "rpc"},
    {"caching", "cache", "redis", "ttl", "memory", "memoization", "evict", "ram"},
    {"reranker", "cross-encoder", "ranking", "candidates", "bge", "pairs", "scoring"},
    {"metrics", "telemetry", "prometheus", "grafana", "histogram", "latency", "duration", "dashboard"},
    {"quantization", "qlora", "nf4", "4-bit", "double", "paged", "optimizers"},
    {"dpo", "preference", "optimization", "reward", "policy", "loss", "reference", "beta"},
    {"vulnerability", "cve", "backdoor", "exploit", "security", "supply", "chain", "xz-utils"}
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
    Production-grade Hybrid Retriever combining Dense (Vector/Pinecone/SentenceTransformer)
    and Sparse (BM25) search using Reciprocal Rank Fusion (RRF), with optional
    Cross-Encoder Reranking (Phase 2).
    """
    def __init__(
        self,
        vectorstore: Any = None,
        corpus_path: Optional[str] = None,
        alpha: float = 0.3,
        top_k: int = 4,
        rrf_k: int = 60,
        dense_embeddings_path: Optional[str] = None
    ):
        self.vectorstore = vectorstore
        self.alpha = alpha
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.corpus_path = corpus_path or settings.bm25_corpus_path
        self.dense_embeddings_path = dense_embeddings_path or getattr(settings, "dense_embeddings_path", "./data/dense_embeddings.npy")
        
        # In-memory document storage for BM25 and fallback
        self.documents: List[DocumentChunk] = []
        self.bm25_index: Optional[BM25Okapi] = None

        # Local dense embeddings
        self.dense_embeddings = None
        self.dense_encoder = None

        self._load_corpus()
        self._load_dense_embeddings()

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

    def _load_dense_embeddings(self) -> None:
        """Load precomputed dense embeddings and SentenceTransformer if available."""
        if NUMPY_AVAILABLE and os.path.exists(self.dense_embeddings_path):
            try:
                self.dense_embeddings = np.load(self.dense_embeddings_path)
            except Exception as e:
                print(f"[HybridRetriever] Warning loading dense embeddings from {self.dense_embeddings_path}: {e}")

        if SENTENCE_TRANSFORMERS_AVAILABLE and self.dense_embeddings is not None:
            try:
                self.dense_encoder = SentenceTransformer(settings.embedding_model)
            except Exception as e:
                print(f"[HybridRetriever] Warning loading SentenceTransformer({settings.embedding_model}): {e}")

    def _build_bm25_index(self) -> None:
        """Builds BM25 index from currently loaded documents."""
        if not self.documents:
            self.bm25_index = None
            return
        tokenized_corpus = [simple_tokenize(doc.page_content) for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def add_documents(self, new_chunks: List[DocumentChunk], persist: bool = False) -> None:
        """Incrementally append documents to memory index, updating BM25 and dense embeddings."""
        if not new_chunks:
            return
        self.documents.extend(new_chunks)
        self._build_bm25_index()

        # Incremental dense embedding update (fast: only encodes newly added chunks)
        if self.dense_encoder is not None and self.dense_embeddings is not None and NUMPY_AVAILABLE:
            try:
                new_embs = self.dense_encoder.encode([d.page_content for d in new_chunks], normalize_embeddings=True)
                self.dense_embeddings = np.vstack([self.dense_embeddings, new_embs])
            except Exception as e:
                print(f"[HybridRetriever] Warning updating dense embeddings incrementally: {e}")

        if persist and self.corpus_path:
            try:
                os.makedirs(os.path.dirname(self.corpus_path), exist_ok=True)
                with open(self.corpus_path, "w", encoding="utf-8") as f:
                    json.dump([d.to_dict() for d in self.documents], f, indent=2)
            except Exception as e:
                print(f"[HybridRetriever] Warning: Could not persist corpus: {e}")

    def set_documents(self, documents: List[DocumentChunk], persist: bool = False) -> None:
        """Update documents in memory and rebuild BM25 index & dense embeddings."""
        self.documents = documents
        self._build_bm25_index()
        if self.dense_encoder is not None and self.documents:
            try:
                self.dense_embeddings = self.dense_encoder.encode([d.page_content for d in self.documents], normalize_embeddings=True)
            except Exception:
                self.dense_embeddings = None
        elif self.dense_embeddings is not None and len(self.dense_embeddings) != len(self.documents):
            self.dense_embeddings = None

        if persist and self.corpus_path:
            try:
                os.makedirs(os.path.dirname(self.corpus_path), exist_ok=True)
                with open(self.corpus_path, "w", encoding="utf-8") as f:
                    json.dump([d.to_dict() for d in self.documents], f, indent=2)
            except Exception as e:
                print(f"[HybridRetriever] Warning: Could not persist corpus: {e}")

    def search_dense(self, query: str, k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Dense semantic search using vector store, SentenceTransformer, or topic fallback."""
        results: List[Tuple[DocumentChunk, float]] = []
        
        # 1. Pinecone Vector Store
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
                if results:
                    return results
            except Exception as e:
                print(f"[HybridRetriever] Vectorstore search failed: {e}")

        # 2. Local SentenceTransformer Dense Index
        if (
            self.dense_embeddings is not None
            and self.dense_encoder is not None
            and self.documents
            and len(self.dense_embeddings) == len(self.documents)
        ):
            try:
                q_vec = self.dense_encoder.encode(query, normalize_embeddings=True)
                scores = np.dot(self.dense_embeddings, q_vec)
                top_indices = np.argsort(scores)[::-1][:k]
                for idx in top_indices:
                    if idx < len(self.documents):
                        results.append((self.documents[idx], float(scores[idx])))
                return results
            except Exception as e:
                print(f"[HybridRetriever] Dense encoder search failed: {e}")

        # 3. Fallback heuristic topic embeddings
        if self.documents:
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
            try:
                RAG_SIMILARITY_SCORE.labels(search_type="dense").observe(min(1.0, max(0.0, raw_score)))
            except Exception:
                pass

        # 2. Process Sparse (BM25) Results
        for rank, (doc, raw_score) in enumerate(sparse_results, start=1):
            key = doc.page_content.strip()
            if key not in doc_map:
                doc_map[key] = doc
            sparse_rrf = (1.0 - alpha) * (1.0 / (k_fusion + rank))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + sparse_rrf
            try:
                RAG_SIMILARITY_SCORE.labels(search_type="bm25").observe(min(1.0, max(0.0, raw_score / 10.0)))
            except Exception:
                pass

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

    def _merge_multi_query_results(
        self,
        ranked_lists: List[List[DocumentChunk]],
        weights: Optional[List[float]] = None,
        k_fusion: int = 60
    ) -> List[DocumentChunk]:
        """
        Merge multiple lists of retrieved DocumentChunks from multi-query expansion
        using Reciprocal Rank Fusion (RRF):
        RRF_score(d) = sum_{list_i} weight_i * (1.0 / (k_fusion + rank_i(d)))
        """
        if not ranked_lists:
            return []
        if len(ranked_lists) == 1:
            return ranked_lists[0]

        if weights is None or len(weights) != len(ranked_lists):
            weights = [1.0] * len(ranked_lists)

        doc_map: Dict[str, DocumentChunk] = {}
        rrf_scores: Dict[str, float] = {}

        for list_idx, doc_list in enumerate(ranked_lists):
            w = weights[list_idx]
            for rank, doc in enumerate(doc_list, start=1):
                key = doc.page_content.strip()
                if key not in doc_map:
                    doc_map[key] = doc
                rrf_scores[key] = rrf_scores.get(key, 0.0) + (w / (k_fusion + rank))

        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        merged = []
        for key in sorted_keys:
            doc = doc_map[key]
            final_score = rrf_scores[key] * (k_fusion + 1)
            merged.append(
                DocumentChunk(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "rrf_score": round(final_score, 4)},
                    score=final_score
                )
            )
        return merged

    def _retrieve_single_query(
        self,
        query: str,
        retriever_type: str,
        alpha: float,
        pool_size: int
    ) -> List[DocumentChunk]:
        """Retrieve candidate chunks for a single query using chosen retrieval type."""
        if retriever_type == "bm25":
            bm25_results = self.search_bm25(query, k=pool_size)
            return [doc for doc, _ in bm25_results]
        elif retriever_type == "semantic":
            dense_results = self.search_dense(query, k=pool_size)
            return [doc for doc, _ in dense_results]
        else:  # "hybrid"
            dense_results = self.search_dense(query, k=pool_size)
            bm25_results = self.search_bm25(query, k=pool_size)

            if not dense_results and bm25_results:
                return [
                    DocumentChunk(
                        doc.page_content,
                        metadata={**doc.metadata, "rrf_score": round(doc.score or 0.01, 4)},
                        score=doc.score
                    )
                    for doc, _ in bm25_results[:pool_size]
                ]
            elif not bm25_results and dense_results:
                return [
                    DocumentChunk(
                        doc.page_content,
                        metadata={**doc.metadata, "rrf_score": round(doc.score or 0.01, 4)},
                        score=doc.score
                    )
                    for doc, _ in dense_results[:pool_size]
                ]
            else:
                fused = self._reciprocal_rank_fusion(dense_results, bm25_results, alpha, self.rrf_k)
                return fused[:pool_size]

    def retrieve(
        self,
        query: str,
        retriever_type: Optional[str] = None,
        alpha: Optional[float] = None,
        top_k: Optional[int] = None,
        rerank: Optional[bool] = None,
        rerank_candidates: Optional[int] = None,
        query_mode: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Retrieve relevant documents according to the chosen retrieval strategy.
        Supports:
          - 'hybrid': Combines BM25 + Dense vector search via RRF
          - 'semantic': Dense vector search only
          - 'bm25': BM25 keyword search only
        Optionally performs:
          - query_mode: 'none' (baseline), 'rewrite' (clean query), 'multi' (expand n phrasings)
          - rerank: cross-encoder reranking (Phase 2)
        """
        retriever_type = retriever_type or settings.retriever_type
        alpha = self.alpha if alpha is None else alpha
        top_k = self.top_k if top_k is None else top_k
        rerank = getattr(settings, "rerank_enabled", False) if rerank is None else rerank
        q_mode = query_mode or getattr(settings, "query_mode", "none").lower()

        start_time = time.time()
        
        if rerank:
            cand_count = rerank_candidates if rerank_candidates is not None else getattr(settings, "rerank_candidates", 20)
            pool_size = max(cand_count, top_k * 3, 20)
        else:
            cand_count = top_k
            pool_size = max(top_k * 3, 15)

        # 1. Candidate Retrieval (Single-query vs Query Rewriting vs Multi-Query RRF)
        if q_mode == "rewrite":
            from app.retrieval.query_rewrite import get_query_rewriter
            rewriter = get_query_rewriter()
            rewritten_q = rewriter.rewrite(query)

            if rewritten_q.strip().lower() != query.strip().lower():
                orig_cands = self._retrieve_single_query(query, retriever_type, alpha, pool_size)
                rewritten_cands = self._retrieve_single_query(rewritten_q, retriever_type, alpha, pool_size)
                candidates = self._merge_multi_query_results(
                    [orig_cands, rewritten_cands],
                    weights=[1.0, 1.0],
                    k_fusion=self.rrf_k
                )
            else:
                candidates = self._retrieve_single_query(query, retriever_type, alpha, pool_size)

        elif q_mode == "multi":
            from app.retrieval.query_rewrite import get_query_rewriter
            rewriter = get_query_rewriter()
            n_variants = getattr(settings, "multi_query_variants", 3)
            variants = rewriter.expand(query, n=n_variants)

            query_list = [query]
            for v in variants:
                if v.strip().lower() != query.strip().lower() and v.strip() not in query_list:
                    query_list.append(v.strip())

            candidate_lists = [
                self._retrieve_single_query(q, retriever_type, alpha, pool_size)
                for q in query_list
            ]
            weights = [1.2] + [1.0] * (len(query_list) - 1)
            candidates = self._merge_multi_query_results(
                candidate_lists,
                weights=weights,
                k_fusion=self.rrf_k
            )

        else:
            # Baseline: single query retrieval
            candidates = self._retrieve_single_query(query, retriever_type, alpha, pool_size)

        # Phase 2: Cross-Encoder Reranking
        if rerank and candidates:
            from app.retrieval.reranker import get_reranker
            reranker = get_reranker()
            results = reranker.rerank(query=query, candidates=candidates[:cand_count], top_k=top_k)
        else:
            results = candidates[:top_k]

        duration = time.time() - start_time
        try:
            RAG_RETRIEVAL_DURATION_SECONDS.labels(retriever_type=retriever_type).observe(duration)
            RAG_RETRIEVED_DOCS_COUNT.labels(retriever_type=retriever_type).observe(len(results))
        except Exception:
            pass

        return results
