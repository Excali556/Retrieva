import io
import os
import json
import zipfile
import unittest
from fastapi.testclient import TestClient

from app.main import app, retriever
from app.document_processor import (
    clean_text,
    split_text_with_overlap,
    extract_text_from_docx,
    extract_text_from_csv,
    parse_document_to_chunks,
    ingest_uploaded_files,
)

def create_in_memory_docx(text: str) -> bytes:
    """Helper to create valid in-memory DOCX bytes."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        xml_content = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body>'
            f'<w:p><w:r><w:t>{text}</w:t></w:r></w:p>'
            '</w:body>'
            '</w:document>'
        )
        zf.writestr("word/document.xml", xml_content)
    return bio.getvalue()

def create_in_memory_pdf(text: str) -> bytes:
    """Helper to create minimal valid PDF bytes with pypdf or fallback."""
    try:
        from pypdf import PdfWriter
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        bio = io.BytesIO()
        writer.write(bio)
        return bio.getvalue()
    except Exception:
        return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF"

class TestDocumentProcessor(unittest.TestCase):
    def test_clean_text(self):
        dirty = "Hello\r\n\x00World!\n\n\n\nEnd"
        cleaned = clean_text(dirty)
        self.assertEqual(cleaned, "Hello\nWorld!\n\nEnd")

    def test_split_text_with_overlap(self):
        text = "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three."
        chunks = split_text_with_overlap(text, chunk_size=50, chunk_overlap=10)
        self.assertGreaterEqual(len(chunks), 1)

    def test_extract_text_from_docx(self):
        sample_text = "Docx retrieval specification test"
        docx_bytes = create_in_memory_docx(sample_text)
        extracted = extract_text_from_docx(docx_bytes)
        self.assertIn("Docx retrieval specification test", extracted)

    def test_extract_text_from_csv(self):
        csv_bytes = b"Model,Latency,Score\nLlama3,120ms,0.92\nBERT,45ms,0.85\n"
        extracted = extract_text_from_csv(csv_bytes)
        self.assertIn("Model: Llama3", extracted)
        self.assertIn("Score: 0.92", extracted)

    def test_parse_document_to_chunks_txt(self):
        content = b"Hybrid search integrates dense vector search with sparse BM25 lexical search."
        chunks = parse_document_to_chunks("readme.txt", content)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Hybrid search", chunks[0].page_content)
        self.assertEqual(chunks[0].metadata["file_type"], ".txt")

    def test_parse_document_to_chunks_docx(self):
        docx_bytes = create_in_memory_docx("Deep learning architectures optimize gradient descent through momentum.")
        chunks = parse_document_to_chunks("spec.docx", docx_bytes)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Deep learning architectures", chunks[0].page_content)
        self.assertEqual(chunks[0].metadata["file_type"], ".docx")

    def test_parse_document_to_chunks_all_types_fallback(self):
        # Arbitrary file format without dedicated parser
        custom_data = b"CustomConfigKey = HighPerformanceClusterNodeAlpha\nServerPort = 9000\n"
        chunks = parse_document_to_chunks("server.conf", custom_data)
        self.assertEqual(len(chunks), 1)
        self.assertIn("HighPerformanceClusterNodeAlpha", chunks[0].page_content)

class TestUploadAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_upload_endpoint_and_listing(self):
        txt_content = b"Retrieval-augmented generation grounds LLM outputs using verifiable factual context documents."
        docx_content = create_in_memory_docx("Prometheus scrapes metrics from instrumented endpoints at regular intervals.")

        files = [
            ("files", ("test_rag_guide.txt", io.BytesIO(txt_content), "text/plain")),
            ("files", ("test_monitoring.docx", io.BytesIO(docx_content), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"))
        ]

        response = self.client.post("/upload", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["files_processed"], 2)
        self.assertGreaterEqual(data["new_chunks_count"], 2)

        # Check /documents endpoint
        docs_resp = self.client.get("/documents")
        self.assertEqual(docs_resp.status_code, 200)
        docs_data = docs_resp.json()
        filenames = [d["filename"] for d in docs_data["documents"]]
        self.assertIn("test_rag_guide.txt", filenames)
        self.assertIn("test_monitoring.docx", filenames)

        # Query newly ingested document to verify searchability
        chat_resp = self.client.post(
            "/chat",
            json={
                "question": "What does Prometheus scrape at regular intervals?",
                "retriever_type": "bm25",
                "top_k": 2
            }
        )
        self.assertEqual(chat_resp.status_code, 200)
        chat_data = chat_resp.json()
        sources_text = " ".join([s["content"] for s in chat_data["sources"]])
        self.assertIn("Prometheus", sources_text)

if __name__ == "__main__":
    unittest.main()
