import unittest
from app.retrieval import HybridRetriever, DocumentChunk, simple_tokenize

class TestHybridRetriever(unittest.TestCase):
    def test_simple_tokenize(self):
        text = "Retrieva: Hybrid Search 2026!"
        tokens = simple_tokenize(text)
        self.assertIn("retrieva", tokens)

        self.assertIn("hybrid", tokens)
        self.assertIn("search", tokens)
        self.assertIn("2026", tokens)

    def test_bm25_retrieval(self):
        docs = [
            DocumentChunk("Machine learning systems need automated CI/CD pipelines.", {"id": "1"}),
            DocumentChunk("Hybrid search combines BM25 and vector embeddings.", {"id": "2"}),
            DocumentChunk("FastAPI exposes Prometheus metrics on /metrics endpoint.", {"id": "3"})
        ]

        retriever = HybridRetriever()
        retriever.set_documents(docs, persist=False)

        results = retriever.retrieve("Prometheus metrics endpoint", retriever_type="bm25", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("Prometheus", results[0].page_content)
        self.assertEqual(results[0].metadata["id"], "3")

    def test_hybrid_rrf_ranking(self):
        docs = [
            DocumentChunk("The quick brown fox jumps over the lazy dog.", {"id": "1"}),
            DocumentChunk("Hybrid search with Reciprocal Rank Fusion improves recall.", {"id": "2"}),
            DocumentChunk("Deep learning neural networks and LLMs.", {"id": "3"})
        ]

        retriever = HybridRetriever(alpha=0.5, top_k=2)
        retriever.set_documents(docs, persist=False)

        results = retriever.retrieve("Reciprocal Rank Fusion hybrid", retriever_type="hybrid", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("Reciprocal Rank Fusion", results[0].page_content)
        self.assertIn("rrf_score", results[0].metadata)

if __name__ == "__main__":
    unittest.main()
