import unittest
from app.retrieval import DocumentChunk
from mlops.eval.evaluate_answers import (
    judge_faithfulness,
    judge_relevance,
    judge_correctness,
    generate_answer_with_llm_or_fallback,
)

class TestAnswerEvaluation(unittest.TestCase):
    def setUp(self):
        self.context = [
            DocumentChunk(
                page_content="Low-Rank Adaptation (LoRA) reduces trainable parameters by freezing pretrained model weights and injecting trainable rank decomposition matrices into each Transformer layer.",
                metadata={"source": "lora_paper.pdf", "page": 1}
            )
        ]

    def test_judge_faithfulness_supported(self):
        answer = "LoRA freezes pretrained weights and injects trainable rank decomposition matrices into Transformer layers."
        score, reason = judge_faithfulness(
            query="How does LoRA work?",
            contexts=self.context,
            answer=answer,
            groq_client=None
        )
        self.assertGreaterEqual(score, 0.80)
        self.assertIn("Grounding ratio", reason)

    def test_judge_faithfulness_unsupported(self):
        answer = "LoRA is a recipe for baking sourdough bread using high temperature fermentation and rye flour."
        score, reason = judge_faithfulness(
            query="How does LoRA work?",
            contexts=self.context,
            answer=answer,
            groq_client=None
        )
        self.assertLess(score, 0.50)

    def test_judge_relevance_relevant(self):
        query = "How does LoRA reduce trainable parameters?"
        answer = "LoRA reduces parameters by freezing pretrained model weights and injecting rank decomposition matrices."
        score, reason = judge_relevance(query, answer, groq_client=None)
        self.assertGreaterEqual(score, 0.70)

    def test_judge_correctness_matching(self):
        query = "What does LoRA freeze?"
        answer = "LoRA freezes the pretrained model weights."
        ground_truth = "It freezes the pretrained model weights."
        score, reason = judge_correctness(query, answer, ground_truth, groq_client=None)
        self.assertGreaterEqual(score, 0.80)

    def test_generate_answer_fallback(self):
        query = "What is LoRA?"
        answer = generate_answer_with_llm_or_fallback(query, self.context, groq_client=None)
        self.assertIn("Low-Rank Adaptation", answer)
        self.assertIn("freezing pretrained model weights", answer)
