import os
import sys
import json
from typing import Dict, Any, List

def validate_ingestion_data(corpus_path: str = "./data/bm25_corpus.json") -> Dict[str, Any]:
    """
    Validates data quality, chunk size distributions, duplicate chunks,
    and schema compliance for the retrieval corpus.
    """
    print("=" * 60)
    print("Running Data Integrity and Quality Check")
    print("=" * 60)

    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        return {"status": "failed", "error": "file_not_found"}

    with open(corpus_path, "r", encoding="utf-8") as f:
        try:
            chunks = json.load(f)
        except Exception as e:
            print(f"Error: Failed to parse JSON: {e}")
            return {"status": "failed", "error": "invalid_json"}

    if not isinstance(chunks, list) or len(chunks) == 0:
        print("Error: Corpus is empty or not a list.")
        return {"status": "failed", "error": "empty_corpus"}

    total_chunks = len(chunks)
    empty_content_count = 0
    short_chunks_count = 0
    unique_contents = set()
    lengths = []

    for idx, c in enumerate(chunks):
        content = c.get("page_content", "").strip()
        lengths.append(len(content))

        if not content:
            empty_content_count += 1
        elif len(content) < 30:
            short_chunks_count += 1

        unique_contents.add(content)

    duplicate_count = total_chunks - len(unique_contents)
    avg_len = sum(lengths) / total_chunks if total_chunks else 0

    print(f"- Total Chunks:        {total_chunks}")
    print(f"- Average Length:      {avg_len:.1f} characters")
    print(f"- Empty Chunks:        {empty_content_count}")
    print(f"- Short Chunks (<30c): {short_chunks_count}")
    print(f"- Duplicate Chunks:    {duplicate_count}")

    passed = (empty_content_count == 0) and (duplicate_count / total_chunks < 0.2)

    result = {
        "status": "passed" if passed else "warning",
        "total_chunks": total_chunks,
        "avg_length": avg_len,
        "duplicate_chunks": duplicate_count,
        "empty_chunks": empty_content_count
    }

    if passed:
        print("Data validation passed.")
    else:
        print("Data validation raised warnings.")

    return result

if __name__ == "__main__":
    validate_ingestion_data()
