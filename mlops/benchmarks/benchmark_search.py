import os
import sys
import time
import statistics

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.config import settings
from app.retrieval import HybridRetriever, DocumentChunk

def run_benchmark(num_queries: int = 100):
    print("=" * 60)
    print(f"Retrieval Performance Benchmark ({num_queries} queries per retriever)")
    print("=" * 60)

    retriever = HybridRetriever(corpus_path=settings.bm25_corpus_path)
    if not retriever.documents:
        # Populate mock documents if empty
        docs = [
            DocumentChunk(
                page_content=f"Document chunk {i}: Hybrid retrieval mechanisms combine dense and sparse index structures for balanced accuracy and low latency.",
                metadata={"chunk_id": f"chunk_{i}"}
            )
            for i in range(50)
        ]
        retriever.set_documents(docs)

    test_queries = [
        "hybrid retrieval mechanisms",
        "low latency and high performance",
        "dense and sparse index structures",
        "retrieva systems"
    ]

    for mode in ["bm25", "semantic", "hybrid"]:
        latencies = []
        for i in range(num_queries):
            query = test_queries[i % len(test_queries)]
            t0 = time.perf_counter()
            retriever.retrieve(query=query, retriever_type=mode, top_k=4)
            latencies.append((time.perf_counter() - t0) * 1000)

        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
        p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
        mean_lat = statistics.mean(latencies)
        qps = 1000.0 / mean_lat if mean_lat > 0 else 0

        print(f"\nMode: [{mode.upper()}]")
        print(f"  Throughput:  {qps:.1f} queries/sec")
        print(f"  Latency p50: {p50:.2f} ms")
        print(f"  Latency p95: {p95:.2f} ms")
        print(f"  Latency p99: {p99:.2f} ms")

if __name__ == "__main__":
    run_benchmark()
