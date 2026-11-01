import time
from typing import List, Optional, Any

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

from app.config import settings
from app.metrics import RAG_RERANK_DURATION_SECONDS

class Reranker:
    """
    Cross-Encoder Reranker using BAAI/bge-reranker-base to score
    and re-order candidate document chunks retrieved by hybrid search.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.reranker_model
        self.model = None
        self._load_model()

    def _load_model(self):
        if CROSS_ENCODER_AVAILABLE:
            try:
                # Runs on CPU or GPU if available
                self.model = CrossEncoder(self.model_name)
            except Exception as e:
                print(f"[Reranker] Warning: Failed to load CrossEncoder({self.model_name}): {e}")
                self.model = None
        else:
            print("[Reranker] Warning: sentence-transformers not available.")

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: int = 4
    ) -> List[Any]:
        """
        Scores (query, chunk.page_content) pairs and returns top_k re-ranked DocumentChunks.
        """
        if not candidates:
            return []

        if len(candidates) <= 1 or self.model is None:
            return candidates[:top_k]

        start_time = time.perf_counter()

        pairs = [[query, doc.page_content] for doc in candidates]
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            print(f"[Reranker] Error during predict: {e}")
            return candidates[:top_k]

        duration = time.perf_counter() - start_time
        try:
            RAG_RERANK_DURATION_SECONDS.labels(model=self.model_name).observe(duration)
        except Exception:
            pass

        # Pair candidates with their cross-encoder score
        scored_candidates = []
        for doc, score in zip(candidates, scores):
            score_float = float(score)
            # Create a copy or update metadata
            updated_meta = {**doc.metadata, "rerank_score": round(score_float, 4)}
            doc.metadata = updated_meta
            doc.score = score_float
            scored_candidates.append(doc)

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda d: d.score, reverse=True)
        return scored_candidates[:top_k]

_reranker_instance: Optional[Reranker] = None

def get_reranker() -> Reranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance
