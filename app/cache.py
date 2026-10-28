import json
import logging
import time
from typing import Dict, Any, Optional

from app.config import settings
from app.metrics import RAG_CACHE_HITS_TOTAL, RAG_CACHE_MISSES_TOTAL

logger = logging.getLogger(__name__)


class SimpleRAGCache:
    """
    In-memory LRU-like cache for RAG queries with TTL expiration and Prometheus metrics.
    """
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.backend = "memory"
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}

    def _normalize_key(self, query: str, retriever_type: str, alpha: float) -> str:
        return f"{retriever_type}:{round(alpha, 2)}:{query.strip().lower()}"

    def is_healthy(self) -> bool:
        return True

    def get(self, query: str, retriever_type: str = "hybrid", alpha: float = 0.5) -> Optional[Dict[str, Any]]:
        if not settings.enable_cache:
            return None

        key = self._normalize_key(query, retriever_type, alpha)
        item = self.cache.get(key)

        if item is None:
            RAG_CACHE_MISSES_TOTAL.inc()
            return None

        # Check expiration
        if time.time() - item["timestamp"] > self.ttl_seconds:
            del self.cache[key]
            RAG_CACHE_MISSES_TOTAL.inc()
            return None

        RAG_CACHE_HITS_TOTAL.inc()
        return item["data"]

    def set(self, query: str, data: Dict[str, Any], retriever_type: str = "hybrid", alpha: float = 0.5) -> None:
        if not settings.enable_cache:
            return

        # Evict oldest entry if size limit exceeded
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]

        key = self._normalize_key(query, retriever_type, alpha)
        self.cache[key] = {
            "timestamp": time.time(),
            "data": data
        }

    def clear(self) -> None:
        self.cache.clear()


class RedisRAGCache:
    """
    Redis-backed distributed cache for RAG queries with native TTL expiration.
    Automatically degrades and falls back to SimpleRAGCache if Redis is unreachable.
    """
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        ttl_seconds: int = 3600,
        max_memory_size: int = 1000,
        key_prefix: str = "rag:cache:",
        socket_timeout: float = 1.0,
        socket_connect_timeout: float = 1.0
    ):
        self.backend = "redis"
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.fallback_cache = SimpleRAGCache(max_size=max_memory_size, ttl_seconds=ttl_seconds)
        self.client = None
        self._connected = False

        try:
            import redis
            self.client = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db,
                decode_responses=True,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout
            )
            # Test connectivity
            self.client.ping()
            self._connected = True
            logger.info(f"Connected to Redis cache at {host}:{port}/{db}")
        except Exception as e:
            logger.warning(f"Could not connect to Redis at {host}:{port}: {e}. Falling back to in-memory cache.")
            self._connected = False

    def is_healthy(self) -> bool:
        if not self._connected or not self.client:
            return False
        try:
            return bool(self.client.ping())
        except Exception:
            self._connected = False
            return False

    def _normalize_key(self, query: str, retriever_type: str, alpha: float) -> str:
        return f"{self.key_prefix}{retriever_type}:{round(alpha, 2)}:{query.strip().lower()}"

    def get(self, query: str, retriever_type: str = "hybrid", alpha: float = 0.5) -> Optional[Dict[str, Any]]:
        if not settings.enable_cache:
            return None

        if self.is_healthy():
            key = self._normalize_key(query, retriever_type, alpha)
            try:
                raw_data = self.client.get(key)
                if raw_data is None:
                    RAG_CACHE_MISSES_TOTAL.inc()
                    return None
                RAG_CACHE_HITS_TOTAL.inc()
                return json.loads(raw_data)
            except Exception as e:
                logger.warning(f"Redis get failed ({e}), falling back to memory cache.")
                self._connected = False

        # Fallback to in-memory cache
        return self.fallback_cache.get(query, retriever_type=retriever_type, alpha=alpha)

    def set(self, query: str, data: Dict[str, Any], retriever_type: str = "hybrid", alpha: float = 0.5) -> None:
        if not settings.enable_cache:
            return

        if self.is_healthy():
            key = self._normalize_key(query, retriever_type, alpha)
            try:
                serialized = json.dumps(data)
                self.client.set(key, serialized, ex=self.ttl_seconds)
                return
            except Exception as e:
                logger.warning(f"Redis set failed ({e}), falling back to memory cache.")
                self._connected = False

        # Fallback to in-memory cache
        self.fallback_cache.set(query, data, retriever_type=retriever_type, alpha=alpha)

    def clear(self) -> None:
        if self.is_healthy():
            try:
                cursor = 0
                while True:
                    cursor, keys = self.client.scan(cursor=cursor, match=f"{self.key_prefix}*", count=100)
                    if keys:
                        self.client.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"Redis clear failed ({e})")
        self.fallback_cache.clear()


def create_cache():
    """Cache factory based on configuration."""
    backend = getattr(settings, "cache_backend", "memory").lower()
    if backend == "redis":
        return RedisRAGCache(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password,
            db=settings.redis_db,
            ttl_seconds=settings.cache_ttl_seconds,
            max_memory_size=settings.cache_max_size
        )
    return SimpleRAGCache(
        max_size=settings.cache_max_size,
        ttl_seconds=settings.cache_ttl_seconds
    )


# Global singleton cache instance
rag_cache = create_cache()
