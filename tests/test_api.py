import unittest

try:
    from fastapi.testclient import TestClient
    from app.main import app
    HAS_FASTAPI = True
except (ImportError, RuntimeError):
    HAS_FASTAPI = False

class TestAPI(unittest.TestCase):
    def setUp(self):
        if not HAS_FASTAPI:
            self.skipTest("FastAPI / TestClient not installed in this environment.")
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["app"], "Retrieva")
        self.assertIn("metrics_url", data)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["app_name"], "Retrieva")
        self.assertIn("retriever_type", data)

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("rag_requests_total", response.text)
        self.assertIn("rag_retrieval_duration_seconds", response.text)

    def test_chat_endpoint_hybrid(self):
        response = self.client.post(
            "/chat",
            json={"question": "What is hybrid search?", "retriever_type": "hybrid", "alpha": 0.5, "top_k": 2}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("sources", data)
        self.assertIn("latency_seconds", data)
        self.assertEqual(data["retriever_type"], "hybrid")

    def test_chat_endpoint_cached(self):
        payload = {"question": "What is hybrid search?", "retriever_type": "hybrid", "alpha": 0.5, "top_k": 2}
        # First request populates cache
        self.client.post("/chat", json=payload)
        # Second identical request hits cache
        response = self.client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("cached"))

    def test_feedback_endpoint(self):
        response = self.client.post(
            "/feedback",
            json={"query": "test query", "answer": "test answer", "rating": "positive"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "recorded")

if __name__ == "__main__":
    unittest.main()
