import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.query_rewrite import QueryRewriter, get_query_rewriter, rewrite, expand
from app.retrieval.hybrid_retriever import HybridRetriever, DocumentChunk


def test_normalize_query():
    rewriter = QueryRewriter()
    raw = "  What is   LoRA, in Fine-Tuning?!  "
    norm = rewriter.normalize_query(raw)
    assert norm == "what is lora in fine tuning"


def test_heuristic_rewrite():
    rewriter = QueryRewriter()
    q = "Could you please explain what is LoRA?"
    cleaned = rewriter._heuristic_rewrite(q)
    assert "lora" in cleaned.lower()
    assert not cleaned.lower().startswith("could you please")


def test_heuristic_expand():
    rewriter = QueryRewriter()
    q = "What is LoRA parameter efficiency?"
    variants = rewriter._heuristic_expand(q, n=3)
    assert isinstance(variants, list)
    assert len(variants) >= 1
    # Check that domain expansion occurs (low-rank adaptation or parameter efficient fine-tuning)
    combined = " ".join(variants).lower()
    assert "low-rank adaptation" in combined or "fine-tuning" in combined or "lora" in combined


def test_query_rewriter_caching():
    rewriter = QueryRewriter()
    rewriter._mem_cache.clear()

    q = "What is BERT pretraining?"
    res1 = rewriter.rewrite(q)
    assert res1

    # Ensure result is now stored in memory cache
    norm_q = rewriter.normalize_query(q)
    cached_val = rewriter._cache_get("rewrite", norm_q)
    assert cached_val == res1

    # Second call should retrieve from cache directly
    res2 = rewriter.rewrite(q)
    assert res2 == res1


def test_expand_caching():
    rewriter = QueryRewriter()
    rewriter._mem_cache.clear()

    q = "What is FlashAttention tiling?"
    vars1 = rewriter.expand(q, n=2)
    assert isinstance(vars1, list)

    vars2 = rewriter.expand(q, n=2)
    assert vars2 == vars1


def test_timeout_fallback_resilience():
    rewriter = QueryRewriter(api_key="mock_key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = TimeoutError("Groq gateway timed out")
    rewriter._client = mock_client

    q = "What is Direct Preference Optimization?"
    # Should not raise exception, should fall back cleanly
    result = rewriter.rewrite(q, timeout=0.1)
    assert result  # Returns fallback string

    exp_result = rewriter.expand(q, n=2, timeout=0.1)
    assert isinstance(exp_result, list)
    assert len(exp_result) >= 1


def test_hybrid_retriever_multi_query_fusion():
    docs = [
        DocumentChunk(
            page_content="LoRA (Low-Rank Adaptation) freezes pretrained model weights and injects trainable rank decomposition matrices.",
            metadata={"source": "lora.pdf", "chunk_id": "chunk_lora_1"}
        ),
        DocumentChunk(
            page_content="BERT pretrains deep bidirectional representations from unlabeled text by masking tokens in the input sequence.",
            metadata={"source": "bert.pdf", "chunk_id": "chunk_bert_1"}
        ),
        DocumentChunk(
            page_content="FlashAttention computes exact attention with IO-awareness, tiling intermediate keys and values into SRAM.",
            metadata={"source": "flashattention.pdf", "chunk_id": "chunk_flash_1"}
        ),
    ]

    retriever = HybridRetriever(alpha=0.3, top_k=2)
    retriever.set_documents(docs, persist=False)

    # 1. Test query_mode="none"
    results_none = retriever.retrieve("What is LoRA?", query_mode="none", top_k=2)
    assert len(results_none) <= 2
    assert "lora" in results_none[0].page_content.lower()

    # 2. Test query_mode="rewrite"
    results_rewrite = retriever.retrieve("Can you tell me what is LoRA?", query_mode="rewrite", top_k=2)
    assert len(results_rewrite) <= 2
    assert "lora" in results_rewrite[0].page_content.lower()

    # 3. Test query_mode="multi"
    results_multi = retriever.retrieve("How does LoRA adapt models?", query_mode="multi", top_k=2)
    assert len(results_multi) <= 2
    assert "lora" in results_multi[0].page_content.lower()


def test_api_chat_with_query_mode():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/chat",
        json={
            "question": "What is LoRA parameter efficiency?",
            "retriever_type": "bm25",
            "top_k": 2,
            "query_mode": "multi",
            "rerank": False
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query_mode"] == "multi"
    assert "answer" in data
    assert isinstance(data["sources"], list)
