try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
except ImportError:
    # Fallback dummy metrics for local environments before pip install
    class DummyMetric:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass

    Counter = Histogram = Gauge = Summary = DummyMetric

# --- Request Metrics ---
RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total count of RAG requests processed",
    ["endpoint", "status"]
)

RAG_REQUEST_DURATION_SECONDS = Histogram(
    "rag_request_duration_seconds",
    "End-to-end request latency in seconds",
    ["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# --- Component Latency Metrics (MLOps Breakdown) ---
RAG_RETRIEVAL_DURATION_SECONDS = Histogram(
    "rag_retrieval_duration_seconds",
    "Time spent in retrieval phase in seconds",
    ["retriever_type"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

RAG_LLM_DURATION_SECONDS = Histogram(
    "rag_llm_duration_seconds",
    "Time spent waiting for LLM completion in seconds",
    ["model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 10.0]
)

# --- Retrieval Quality Metrics ---
RAG_RETRIEVED_DOCS_COUNT = Histogram(
    "rag_retrieved_documents_count",
    "Number of context documents retrieved per query",
    ["retriever_type"],
    buckets=[0, 1, 2, 4, 6, 8, 10, 20]
)

RAG_SIMILARITY_SCORE = Histogram(
    "rag_similarity_score",
    "Relevance/similarity scores of retrieved documents",
    ["search_type"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# --- Cache Efficiency Metrics ---
RAG_CACHE_HITS_TOTAL = Counter(
    "rag_cache_hits_total",
    "Total number of query cache hits"
)

RAG_CACHE_MISSES_TOTAL = Counter(
    "rag_cache_misses_total",
    "Total number of query cache misses"
)

# --- MLOps Human Feedback & Evaluation Metrics ---
RAG_FEEDBACK_TOTAL = Counter(
    "rag_feedback_total",
    "User feedback on generated responses",
    ["rating"]
)

RAG_EVAL_PRECISION = Gauge(
    "rag_retrieval_precision",
    "Latest batch context precision evaluation score"
)

RAG_EVAL_RECALL = Gauge(
    "rag_retrieval_recall",
    "Latest batch context recall evaluation score"
)

RAG_EVAL_FAITHFULNESS = Gauge(
    "rag_generation_faithfulness",
    "Latest batch faithfulness score (groundedness in context)"
)
