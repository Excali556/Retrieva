import unittest
from app.retrieval import HybridRetriever, DocumentChunk, Reranker, get_reranker

class TestReranker(unittest.TestCase):
    def test_reranker_instance(self):
        reranker = get_reranker()
        self.assertIsNotNone(reranker)
        self.assertEqual(reranker.model_name, "BAAI/bge-reranker-base")

    def test_rerank_scoring(self):
        reranker = get_reranker()
        candidates = [
            DocumentChunk("Apples and oranges are common fruits.", {"id": "1"}),
            DocumentChunk("The Transformer uses multi-head self-attention mechanisms.", {"id": "2"}),
            DocumentChunk("Baking recipes for chocolate cake and cookies.", {"id": "3"})
        ]

        reranked = reranker.rerank(
            query="How does multi-head self-attention work in Transformers?",
            candidates=candidates,
            top_k=2
        )

        self.assertEqual(len(reranked), 2)
        # Most relevant document should be ranked #1
        self.assertEqual(reranked[0].metadata["id"], "2")
        self.assertIn("rerank_score", reranked[0].metadata)
        self.assertGreater(reranked[0].score, reranked[1].score)

    def test_hybrid_with_rerank_flag(self):
        docs = [
            DocumentChunk("Unrelated cooking tips and food nutrition.", {"id": "c1"}),
            DocumentChunk("Attention Is All You Need introduced the Transformer with scaled dot-product attention.", {"id": "c2"}),
            DocumentChunk("Weather forecast and atmospheric pressure patterns.", {"id": "c3"})
        ]

        retriever = HybridRetriever(alpha=0.5, top_k=2)
        retriever.set_documents(docs, persist=False)

        results = retriever.retrieve(
            query="What model introduced scaled dot-product attention?",
            retriever_type="hybrid",
            top_k=1,
            rerank=True,
            rerank_candidates=3
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["id"], "c2")
        self.assertIn("rerank_score", results[0].metadata)

if __name__ == "__main__":
    unittest.main()
