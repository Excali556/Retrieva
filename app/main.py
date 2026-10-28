import os
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app.cache import rag_cache
from app.metrics import (
    RAG_REQUESTS_TOTAL,
    RAG_REQUEST_DURATION_SECONDS,
    RAG_LLM_DURATION_SECONDS,
    RAG_FEEDBACK_TOTAL
)
from app.retrieval import HybridRetriever, DocumentChunk

# 1. Initialize FastAPI Application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Hybrid search RAG API with Pinecone, BM25, Redis caching, and Prometheus metrics."
)

# 2. Expose Prometheus Metrics via Instrumentator
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"]
)
instrumentator.instrument(app).expose(app, endpoint="/metrics")

# 3. Model & Retrieval Components Setup
vectorstore = None
embeddings = None
llm = None

def init_components():
    global vectorstore, embeddings, llm
    
    # Initialize Embeddings
    if settings.huggingfacehub_api_token:
        try:
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
            embeddings = HuggingFaceEndpointEmbeddings(
                model=settings.embedding_model,
                huggingfacehub_api_token=settings.huggingfacehub_api_token
            )
        except Exception as e:
            print(f"[Init] Warning: Could not initialize HuggingFace embeddings: {e}")

    # Initialize Pinecone Vector Store
    if settings.pinecone_api_key and embeddings:
        try:
            from langchain_pinecone import PineconeVectorStore
            vectorstore = PineconeVectorStore(
                index_name=settings.pinecone_index_name,
                embedding=embeddings
            )
        except Exception as e:
            print(f"[Init] Warning: Could not initialize PineconeVectorStore: {e}")

    # Initialize Groq LLM
    if settings.groq_api_key:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                temperature=settings.llm_temperature,
                model_name=settings.llm_model,
                groq_api_key=settings.groq_api_key
            )
        except Exception as e:
            print(f"[Init] Warning: Could not initialize ChatGroq: {e}")

init_components()

# Initialize Hybrid Retriever
retriever = HybridRetriever(
    vectorstore=vectorstore,
    corpus_path=settings.bm25_corpus_path,
    alpha=settings.hybrid_alpha,
    top_k=settings.top_k,
    rrf_k=settings.rrf_k
)

# 4. Request / Response Pydantic Schemas
class ChatRequest(BaseModel):
    question: str = Field(..., description="The query or question to ask the RAG system")
    retriever_type: Optional[str] = Field(None, description="Retrieval mode: 'hybrid', 'semantic', or 'bm25'")
    alpha: Optional[float] = Field(None, ge=0.0, le=1.0, description="Hybrid weight: 0.0=BM25 only, 1.0=Semantic only")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of context chunks to retrieve")

class SourceCitation(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0

class ChatResponse(BaseModel):
    answer: str
    retriever_type: str
    alpha: float
    sources: List[SourceCitation]
    latency_seconds: float
    cached: bool = False

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: str = Field(..., description="'positive' or 'negative'")
    notes: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    vectorstore_connected: bool
    bm25_corpus_size: int
    llm_connected: bool
    retriever_type: str
    cache_backend: str = "memory"
    cache_connected: bool = True

# 5. Core RAG Generation Helper
def generate_rag_response(query: str, contexts: List[DocumentChunk]) -> str:
    """Generate grounded answer using Groq LLM or fallback simulation."""
    if not contexts:
        return "I could not find any relevant information in the documents to answer your question."

    context_str = "\n\n".join([
        f"[Context {i+1} | Source: {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 1)})]:\n{doc.page_content}"
        for i, doc in enumerate(contexts)
    ])

    system_prompt = (
        "You are an assistant answering questions based on the provided document excerpts.\n"
        "Guidelines:\n"
        "1. Base your answer only on the facts directly stated in the context.\n"
        "2. If the context does not contain enough information to answer, state that clearly.\n"
        "3. Cite the relevant context numbers or sources where applicable.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    if llm is not None:
        start_llm = time.time()
        try:
            response = llm.invoke(system_prompt)
            llm_duration = time.time() - start_llm
            RAG_LLM_DURATION_SECONDS.labels(model=settings.llm_model).observe(llm_duration)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            return f"Error contacting LLM: {str(e)}\n\nExtracted Context:\n{context_str[:500]}..."
    else:
        # Fallback simulation when LLM API key is not configured
        return (
            f"[Local mode - LLM key not configured]: Context found for '{query}'.\n\n"
            f"Top context:\n{contexts[0].page_content[:300]}..."
        )

# 6. API Endpoints
@app.get("/", tags=["General"])
def root():
    return {
        "status": "online",
        "app": settings.app_name,
        "version": settings.app_version,
        "retrieval": settings.retriever_type,
        "docs_url": "/docs",
        "metrics_url": "/metrics"
    }

@app.get("/health", response_model=HealthResponse, tags=["Observability"])
def health_check():
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        vectorstore_connected=vectorstore is not None,
        bm25_corpus_size=len(retriever.documents),
        llm_connected=llm is not None,
        retriever_type=settings.retriever_type,
        cache_backend=getattr(rag_cache, "backend", "memory"),
        cache_connected=rag_cache.is_healthy() if hasattr(rag_cache, "is_healthy") else True
    )

@app.post("/chat", response_model=ChatResponse, tags=["RAG"])
async def chat_endpoint(request: ChatRequest):
    req_start = time.time()
    endpoint = "/chat"
    retriever_mode = request.retriever_type or settings.retriever_type
    alpha = request.alpha if request.alpha is not None else settings.hybrid_alpha
    top_k = request.top_k or settings.top_k

    try:
        # Check Cache
        cached_result = rag_cache.get(request.question, retriever_type=retriever_mode, alpha=alpha)
        if cached_result:
            RAG_REQUESTS_TOTAL.labels(endpoint=endpoint, status="success_cached").inc()
            latency = time.time() - req_start
            RAG_REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(latency)
            result = dict(cached_result)
            result["latency_seconds"] = round(latency, 4)
            result["cached"] = True
            return ChatResponse(**result)

        # 1. Retrieve Contexts using Hybrid / Dense / BM25
        retrieved_chunks = retriever.retrieve(
            query=request.question,
            retriever_type=retriever_mode,
            alpha=alpha,
            top_k=top_k
        )

        # 2. Generate Grounded LLM Response
        answer = generate_rag_response(request.question, retrieved_chunks)

        citations = [
            SourceCitation(
                content=chunk.page_content,
                metadata=chunk.metadata,
                score=chunk.score
            )
            for chunk in retrieved_chunks
        ]

        total_latency = time.time() - req_start
        RAG_REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(total_latency)
        RAG_REQUESTS_TOTAL.labels(endpoint=endpoint, status="success").inc()

        response_data = {
            "answer": answer,
            "retriever_type": retriever_mode,
            "alpha": alpha,
            "sources": citations,
            "latency_seconds": round(total_latency, 4),
            "cached": False
        }

        # Store in Cache
        rag_cache.set(
            request.question,
            {**response_data, "sources": [s.model_dump() for s in citations]},
            retriever_type=retriever_mode,
            alpha=alpha
        )

        return ChatResponse(**response_data)

    except Exception as e:
        RAG_REQUESTS_TOTAL.labels(endpoint=endpoint, status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG Execution Error: {str(e)}"
        )

@app.post("/feedback", tags=["MLOps"])
def user_feedback(feedback: FeedbackRequest):
    """
    MLOps Feedback Loop: Log user thumbs up/down to Prometheus metrics
    for continuous monitoring of retrieval quality and answer satisfaction.
    """
    rating = feedback.rating.lower()
    if rating not in ["positive", "negative"]:
        raise HTTPException(status_code=400, detail="Rating must be 'positive' or 'negative'")
    
    RAG_FEEDBACK_TOTAL.labels(rating=rating).inc()
    return {"status": "recorded", "rating": rating}

@app.post("/corpus/reload", tags=["Admin"])
def reload_corpus():
    """Reload BM25 corpus from disk without restarting."""
    retriever._load_corpus()
    return {
        "status": "reloaded",
        "documents_loaded": len(retriever.documents)
    }
