import os
import re
import json
import time
import hashlib
import logging
from typing import List, Optional, Any, Dict

from app.config import settings
from app.metrics import (
    RAG_QUERY_REWRITE_DURATION_SECONDS,
    RAG_QUERY_REWRITE_TOTAL,
)

logger = logging.getLogger(__name__)

# Acronym and technical synonym map for domain-aware heuristic expansion/fallback
TECH_SYNONYMS = {
    "lora": ["low-rank adaptation", "parameter efficient fine-tuning", "rank decomposition matrices"],
    "rag": ["retrieval augmented generation", "dense passage retrieval", "external knowledge retrieval"],
    "bert": ["bidirectional encoder representations", "masked language model", "transformer encoder"],
    "flashattention": ["io-aware attention", "fast exact attention", "sram tiling memory hierarchy"],
    "react": ["reasoning and acting", "interleaved thought action observation", "synergizing reasoning"],
    "raft": ["consensus algorithm", "replicated log state machine", "leader election heartbeat"],
    "dpo": ["direct preference optimization", "reinforcement learning from human feedback", "implicit reward function"],
    "qlora": ["4-bit quantized lora", "nf4 double quantization", "paged optimizers"],
    "mrr": ["mean reciprocal rank", "reciprocal rank evaluation", "first relevant document rank"],
    "rrf": ["reciprocal rank fusion", "rank aggregation score", "interleaving ranked lists"],
    "bge": ["baai general embedding", "reranker cross-encoder", "relevance scoring"],
}

FILLER_PATTERNS = [
    r"^(can you\s+|could you\s+)?(please\s+)?(tell me\s+)?(what is|what are|what was|explain|describe|how does|how do|how is)\s+",
    r"\s+(in detail|briefly|please|exactly)\??$",
]


class QueryRewriter:
    """
    Query Rewriting and Multi-Query Expansion module with Groq LLM,
    Redis caching, strict timeouts, and fallback resilience.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key or getattr(settings, "groq_api_key", None) or os.getenv("GROQ_API_KEY")
        self.model = model or getattr(settings, "query_rewrite_model", "llama-3.3-70b-versatile")
        self.timeout = timeout or getattr(settings, "query_rewrite_timeout", 2.5)

        # In-memory LRU-like cache fallback
        self._mem_cache = {}
        self._redis_client = None
        self._init_redis()

        self._client = None
        if self.api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key, timeout=self.timeout)
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}")

    def _init_redis(self):
        """Initialize Redis connection if enabled and configured."""
        backend = getattr(settings, "cache_backend", "memory").lower()
        if getattr(settings, "enable_cache", True) and backend == "redis":
            try:
                import redis
                host = getattr(settings, "redis_host", "localhost")
                port = getattr(settings, "redis_port", 6379)
                pwd = getattr(settings, "redis_password", None)
                db = getattr(settings, "redis_db", 0)
                client = redis.Redis(
                    host=host,
                    port=port,
                    password=pwd,
                    db=db,
                    decode_responses=True,
                    socket_timeout=1.0,
                    socket_connect_timeout=1.0
                )
                client.ping()
                self._redis_client = client
                logger.info("QueryRewriter successfully connected to Redis cache.")
            except Exception:
                self._redis_client = None

    def normalize_query(self, query: str) -> str:
        """Normalize query string for consistent cache key generation."""
        cleaned = re.sub(r"[-?!.,;:\"'’‘()\[\]{}_]", " ", query.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _cache_get(self, prefix: str, normalized_query: str) -> Optional[Any]:
        """Fetch item from Redis cache or memory cache fallback."""
        key_hash = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:16]
        full_key = f"rag:{prefix}:{key_hash}"

        if self._redis_client:
            try:
                val = self._redis_client.get(full_key)
                if val is not None:
                    return json.loads(val)
            except Exception:
                pass

        # Memory cache fallback
        entry = self._mem_cache.get(full_key)
        if entry:
            if time.time() - entry["ts"] < getattr(settings, "cache_ttl_seconds", 3600):
                return entry["data"]
            del self._mem_cache[full_key]

        return None

    def _cache_set(self, prefix: str, normalized_query: str, data: Any):
        """Store item in Redis cache and memory cache with TTL."""
        key_hash = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:16]
        full_key = f"rag:{prefix}:{key_hash}"
        ttl = getattr(settings, "cache_ttl_seconds", 3600)

        if self._redis_client:
            try:
                self._redis_client.set(full_key, json.dumps(data), ex=ttl)
            except Exception:
                pass

        if len(self._mem_cache) > 2000:
            oldest = min(self._mem_cache.keys(), key=lambda k: self._mem_cache[k]["ts"])
            del self._mem_cache[oldest]

        self._mem_cache[full_key] = {"data": data, "ts": time.time()}

    def _heuristic_rewrite(self, question: str) -> str:
        """Heuristic rule-based query cleaning for offline or fallback operation."""
        cleaned = question.strip()
        for pat in FILLER_PATTERNS:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip("?!.,;: ")
        if cleaned:
            return cleaned
        return question.strip()

    def _heuristic_expand(self, question: str, n: int = 3) -> List[str]:
        """Heuristic domain synonym query expansion for offline or fallback operation."""
        norm = self.normalize_query(question)
        variants = []
        tokens = norm.split()

        # 1. Expand detected acronyms
        for token in tokens:
            if token in TECH_SYNONYMS:
                for syn in TECH_SYNONYMS[token]:
                    cand = question.lower().replace(token, syn).strip()
                    if cand not in variants and cand != norm:
                        variants.append(cand)
                    if len(variants) >= n:
                        break
            if len(variants) >= n:
                break

        # 2. Add cleaned version if distinct
        cleaned = self._heuristic_rewrite(question)
        if cleaned != question and cleaned not in variants:
            variants.append(cleaned)

        # 3. Add focused keyword query
        kw_query = " ".join([t for t in tokens if len(t) > 3 and t not in {"what", "which", "where", "does", "from", "with", "into", "than"}])
        if kw_query and kw_query not in variants and kw_query != norm:
            variants.append(kw_query)

        return variants[:n]

    def rewrite(self, question: str, timeout: Optional[float] = None) -> str:
        """
        Rewrite question into a single cleaner, more specific search query.
        Uses Redis cache, Groq LLM with strict timeout, and fallback.
        """
        if not question or not question.strip():
            return question

        norm_q = self.normalize_query(question)
        cached = self._cache_get("rewrite", norm_q)
        if cached:
            RAG_QUERY_REWRITE_TOTAL.labels(mode="rewrite", status="cached").inc()
            return str(cached)

        if not self._client:
            # Fallback when Groq client is not configured
            rewritten = self._heuristic_rewrite(question)
            self._cache_set("rewrite", norm_q, rewritten)
            RAG_QUERY_REWRITE_TOTAL.labels(mode="rewrite", status="heuristic").inc()
            return rewritten

        prompt = (
            "You are an expert search query optimizer for AI and computer science research documents.\n"
            "Rewrite the user's question into one concise, cleaner, and more specific search query optimized for retrieval.\n"
            "Preserve technical terms, acronyms, and names. Output ONLY the rewritten search query with no quotes or filler.\n\n"
            f"User Question: {question}\n"
            "Search Query:"
        )

        start_t = time.time()
        call_timeout = timeout or self.timeout
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,
                timeout=call_timeout,
            )
            duration = time.time() - start_t
            RAG_QUERY_REWRITE_DURATION_SECONDS.labels(mode="rewrite").observe(duration)

            raw_text = response.choices[0].message.content.strip()
            # Clean possible markdown or quotation wrapping
            cleaned = raw_text.strip("\"'`\n").split("\n")[0].strip()
            result = cleaned if cleaned else question

            self._cache_set("rewrite", norm_q, result)
            RAG_QUERY_REWRITE_TOTAL.labels(mode="rewrite", status="success").inc()
            return result

        except Exception as e:
            duration = time.time() - start_t
            logger.warning(f"Query rewrite LLM call failed or timed out ({e}). Falling back to original.")
            RAG_QUERY_REWRITE_TOTAL.labels(mode="rewrite", status="fallback").inc()
            fallback = self._heuristic_rewrite(question)
            return fallback

    def expand(self, question: str, n: int = 3, timeout: Optional[float] = None) -> List[str]:
        """
        Generate n diverse alternative phrasings for the query.
        Uses Redis cache, Groq LLM with strict timeout, and fallback.
        """
        if not question or not question.strip():
            return [question]

        norm_q = self.normalize_query(question)
        cached = self._cache_get(f"expand:{n}", norm_q)
        if cached and isinstance(cached, list):
            RAG_QUERY_REWRITE_TOTAL.labels(mode="multi", status="cached").inc()
            return cached

        if not self._client:
            # Fallback when Groq client is not configured
            variants = self._heuristic_expand(question, n=n)
            if not variants:
                variants = [question]
            self._cache_set(f"expand:{n}", norm_q, variants)
            RAG_QUERY_REWRITE_TOTAL.labels(mode="multi", status="heuristic").inc()
            return variants

        prompt = (
            "You are an expert query expansion engine for academic computer science and AI research papers.\n"
            f"Generate exactly {n} diverse alternative search queries or phrasings that capture the same intent or technical terminology.\n"
            "Rules:\n"
            "- Output one query per line.\n"
            "- Do NOT include numbers, bullet points, hyphens, or markdown.\n"
            "- Keep queries technical, precise, and search-ready.\n\n"
            f"Question: {question}\n"
            "Alternative Queries:"
        )

        start_t = time.time()
        call_timeout = timeout or self.timeout
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
                timeout=call_timeout,
            )
            duration = time.time() - start_t
            RAG_QUERY_REWRITE_DURATION_SECONDS.labels(mode="multi").observe(duration)

            raw_text = response.choices[0].message.content.strip()
            lines = [
                re.sub(r"^(\d+[\.\)]|\-|\*)\s*", "", line).strip("\"'` ")
                for line in raw_text.split("\n")
                if line.strip()
            ]

            # Filter duplicates and empty lines
            unique_variants = []
            for line in lines:
                if line and line.lower() != norm_q and line not in unique_variants:
                    unique_variants.append(line)
                if len(unique_variants) >= n:
                    break

            if not unique_variants:
                unique_variants = self._heuristic_expand(question, n=n)
                if not unique_variants:
                    unique_variants = [question]

            self._cache_set(f"expand:{n}", norm_q, unique_variants)
            RAG_QUERY_REWRITE_TOTAL.labels(mode="multi", status="success").inc()
            return unique_variants

        except Exception as e:
            duration = time.time() - start_t
            logger.warning(f"Multi-query expansion LLM call failed or timed out ({e}). Falling back to heuristic expansion.")
            RAG_QUERY_REWRITE_TOTAL.labels(mode="multi", status="fallback").inc()
            fallback_variants = self._heuristic_expand(question, n=n)
            return fallback_variants if fallback_variants else [question]


# Singleton instance
_GLOBAL_REWRITER: Optional[QueryRewriter] = None


def get_query_rewriter() -> QueryRewriter:
    """Singleton getter for QueryRewriter."""
    global _GLOBAL_REWRITER
    if _GLOBAL_REWRITER is None:
        _GLOBAL_REWRITER = QueryRewriter()
    return _GLOBAL_REWRITER


def rewrite(question: str) -> str:
    """Standalone module function: rewrite question into cleaner, more specific search query."""
    return get_query_rewriter().rewrite(question)


def expand(question: str, n: int = 3) -> List[str]:
    """Standalone module function: expand question into n alternative query phrasings."""
    return get_query_rewriter().expand(question, n=n)
