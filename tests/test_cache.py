import unittest
from unittest.mock import MagicMock, patch
import json
import time

from app.cache import SimpleRAGCache, RedisRAGCache, create_cache


class TestSimpleRAGCache(unittest.TestCase):
    def test_cache_set_and_get(self):
        cache = SimpleRAGCache(max_size=10, ttl_seconds=5)
        query = "What is hybrid search?"
        payload = {"answer": "Hybrid search combines dense and sparse retrieval."}

        cache.set(query, payload, retriever_type="hybrid", alpha=0.5)
        cached = cache.get(query, retriever_type="hybrid", alpha=0.5)

        self.assertIsNotNone(cached)
        self.assertEqual(cached["answer"], payload["answer"])

    def test_cache_miss_on_different_params(self):
        cache = SimpleRAGCache(max_size=10, ttl_seconds=5)
        query = "Explain Prometheus"
        payload = {"answer": "Monitoring tool."}

        cache.set(query, payload, retriever_type="hybrid", alpha=0.5)

        # Different alpha or retriever_type should miss
        self.assertIsNone(cache.get(query, retriever_type="bm25", alpha=0.5))
        self.assertIsNone(cache.get(query, retriever_type="hybrid", alpha=0.8))

    def test_cache_ttl_expiration(self):
        cache = SimpleRAGCache(max_size=10, ttl_seconds=1)
        query = "Temporary query"
        cache.set(query, {"answer": "Temporary"}, retriever_type="hybrid", alpha=0.5)

        time.sleep(1.1)
        self.assertIsNone(cache.get(query, retriever_type="hybrid", alpha=0.5))

    def test_cache_eviction(self):
        cache = SimpleRAGCache(max_size=2, ttl_seconds=60)
        cache.set("q1", {"answer": "1"})
        time.sleep(0.01)
        cache.set("q2", {"answer": "2"})
        time.sleep(0.01)
        cache.set("q3", {"answer": "3"})

        # q1 should be evicted because max_size=2
        self.assertIsNone(cache.get("q1"))
        self.assertIsNotNone(cache.get("q2"))
        self.assertIsNotNone(cache.get("q3"))


class TestRedisRAGCache(unittest.TestCase):
    def test_redis_fallback_when_unreachable(self):
        # Point to unreachable port/host; should smoothly degrade to fallback memory cache
        cache = RedisRAGCache(host="127.0.0.1", port=9999, ttl_seconds=10, socket_connect_timeout=0.2)
        self.assertFalse(cache.is_healthy())

        query = "What is RAG?"
        payload = {"answer": "Retrieval Augmented Generation"}
        cache.set(query, payload, retriever_type="hybrid", alpha=0.3)

        cached = cache.get(query, retriever_type="hybrid", alpha=0.3)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["answer"], payload["answer"])

    @patch("redis.Redis")
    def test_redis_cache_hits_and_sets(self, mock_redis_class):
        mock_instance = MagicMock()
        mock_redis_class.return_value = mock_instance
        mock_instance.ping.return_value = True

        cache = RedisRAGCache(host="localhost", port=6379, ttl_seconds=3600)
        self.assertTrue(cache.is_healthy())

        # Test set
        query = "How does reciprocal rank fusion work?"
        data = {"answer": "RRF combines rankings from multiple retrieval algorithms."}
        cache.set(query, data, retriever_type="hybrid", alpha=0.3)

        expected_key = "rag:cache:hybrid:0.3:how does reciprocal rank fusion work?"
        mock_instance.set.assert_called_with(expected_key, json.dumps(data), ex=3600)

        # Test get hit
        mock_instance.get.return_value = json.dumps(data)
        result = cache.get(query, retriever_type="hybrid", alpha=0.3)
        self.assertEqual(result, data)

        # Test get miss
        mock_instance.get.return_value = None
        miss_result = cache.get("Unknown query", retriever_type="hybrid", alpha=0.3)
        self.assertIsNone(miss_result)

    def test_create_cache_factory(self):
        with patch("app.cache.settings") as mock_settings:
            mock_settings.cache_backend = "memory"
            mock_settings.cache_max_size = 50
            mock_settings.cache_ttl_seconds = 100
            cache = create_cache()
            self.assertIsInstance(cache, SimpleRAGCache)


if __name__ == "__main__":
    unittest.main()
