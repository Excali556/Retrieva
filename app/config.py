import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    
    class Settings(BaseSettings):
        # App Settings
        app_name: str = "Autonomous RAG System"
        app_version: str = "2.0.0"
        environment: str = "production"
        
        # LLM Settings (Groq)
        groq_api_key: Optional[str] = None
        llm_model: str = "llama-3.3-70b-versatile"
        llm_temperature: float = 0.0
        llm_max_tokens: int = 1024

        # Pinecone Settings
        pinecone_api_key: Optional[str] = None
        pinecone_index_name: str = "autonomous-rag"
        pinecone_cloud: str = "aws"
        pinecone_region: str = "us-east-1"

        # Embedding Settings (HuggingFace)
        huggingfacehub_api_token: Optional[str] = None
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
        embedding_dimension: int = 384

        # Retrieval Settings
        retriever_type: str = "hybrid"  # "hybrid", "semantic", or "bm25"
        hybrid_alpha: float = 0.3       # Weight: 0.0 (BM25 only) to 1.0 (Semantic only). 0.3 is empirical optimum.
        top_k: int = 4
        rrf_k: int = 60                 # Smoothing constant for Reciprocal Rank Fusion

        # Paths
        data_dir: str = "./data"
        bm25_corpus_path: str = "./data/bm25_corpus.json"
        bm25_index_path: str = "./data/bm25_index.pkl"

        # Cache Settings
        enable_cache: bool = True
        cache_backend: str = "memory"   # "redis" or "memory" (defaults to memory for tests/local, redis in Docker)
        cache_ttl_seconds: int = 3600
        cache_max_size: int = 1000
        redis_host: str = "localhost"
        redis_port: int = 6379
        redis_password: Optional[str] = None
        redis_db: int = 0

        # Observability
        prometheus_metrics_path: str = "/metrics"

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"
        )

    settings = Settings()

except ImportError:
    # Pure Python fallback configuration if pydantic-settings is not yet installed
    class FallbackSettings:
        def __init__(self):
            self.app_name = os.getenv("APP_NAME", "Autonomous RAG System")
            self.app_version = os.getenv("APP_VERSION", "2.0.0")
            self.environment = os.getenv("ENVIRONMENT", "production")

            self.groq_api_key = os.getenv("GROQ_API_KEY")
            self.llm_model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
            self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
            self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))

            self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
            self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "autonomous-rag")
            self.pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
            self.pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")

            self.huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))

            self.retriever_type = os.getenv("RETRIEVER_TYPE", "hybrid")
            self.hybrid_alpha = float(os.getenv("HYBRID_ALPHA", "0.3"))
            self.top_k = int(os.getenv("TOP_K", "4"))
            self.rrf_k = int(os.getenv("RRF_K", "60"))

            self.data_dir = os.getenv("DATA_DIR", "./data")
            self.bm25_corpus_path = os.getenv("BM25_CORPUS_PATH", "./data/bm25_corpus.json")
            self.bm25_index_path = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")

            self.enable_cache = os.getenv("ENABLE_CACHE", "true").lower() in ("true", "1")
            self.cache_backend = os.getenv("CACHE_BACKEND", "memory").lower()
            self.cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
            self.cache_max_size = int(os.getenv("CACHE_MAX_SIZE", "1000"))
            self.redis_host = os.getenv("REDIS_HOST", "localhost")
            self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
            self.redis_password = os.getenv("REDIS_PASSWORD", None)
            self.redis_db = int(os.getenv("REDIS_DB", "0"))

            self.prometheus_metrics_path = os.getenv("PROMETHEUS_METRICS_PATH", "/metrics")

    settings = FallbackSettings()
