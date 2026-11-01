import os
import streamlit as st
import requests
import time
from streamlit_lottie import st_lottie

# --- API CONFIG ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://chatbot-api:8000").rstrip("/")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

# --- UI CONFIG ---
st.set_page_config(page_title="Retrieva", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .stChatMessage {
        animation: fadeIn 0.4s ease-in-out;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        background-color: #f0f2f6;
        font-size: 0.85rem;
        margin-right: 8px;
        color: #31333F;
    }
    .source-box {
        padding: 10px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        background-color: #f9f9f9;
        margin-top: 8px;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(8px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# --- UTILITIES ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

lottie_ai = load_lottieurl("https://lottie.host/78229c29-399d-4340-bb53-3932731804c8/G2Q9fTzFzI.json")

# --- SIDEBAR: RETRIEVAL & OBSERVABILITY SETTINGS ---
with st.sidebar:
    if lottie_ai:
        st_lottie(lottie_ai, height=130, key="sidebar_robot")
    st.title("Retrieva")
    st.caption("Hybrid Search with Prometheus and Grafana Observability")

    st.subheader("Retrieval Strategy")
    retrieval_mode = st.selectbox(
        "Search Algorithm",
        options=["hybrid", "semantic", "bm25"],
        index=0,
        help="Hybrid combines Dense Vector search with Sparse BM25 lexical search using Reciprocal Rank Fusion."
    )

    alpha = 0.5
    if retrieval_mode == "hybrid":
        alpha = st.slider(
            "Hybrid Weight (Alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="0.0 = BM25 Only | 0.3 = Production Default | 0.5 = Balanced | 1.0 = Semantic Only"
        )

    top_k = st.slider("Top Documents (K)", min_value=1, max_value=8, value=4)

    # Phase 2: Cross-Encoder Reranking Toggle
    rerank_enabled = st.checkbox(
        "Enable Cross-Encoder Reranking",
        value=False,
        help="Retrieves top 20 candidate chunks and scores each with BAAI/bge-reranker-base to fix hybrid recall dips."
    )

    # Phase 3: Query Rewriting & Multi-Query Expansion
    query_mode = st.selectbox(
        "Query Expansion Mode (Phase 3)",
        options=["none", "rewrite", "multi"],
        index=0,
        help="'none': original query | 'rewrite': LLM rewrites into cleaner, specific query | 'multi': expands into 3 diverse phrasings merged with RRF"
    )

    st.divider()

    # --- DOCUMENT INGESTION ---
    st.subheader("Document Ingestion")
    st.caption("Upload PDFs or any documents (DOCX, TXT, MD, CSV, JSON, code, etc.) to index them into Retrieva.")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=None,
        accept_multiple_files=True,
        help="Upload PDF, DOCX, TXT, MD, CSV, JSON, or any document format.",
        key="sidebar_doc_uploader"
    )

    if uploaded_files:
        st.write(f"Selected: **{len(uploaded_files)} file(s)**")
        if st.button("Upload & Index", type="primary", use_container_width=True, key="btn_sidebar_upload"):
            with st.spinner(f"Ingesting and indexing {len(uploaded_files)} document(s)..."):
                try:
                    files_payload = [
                        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                        for f in uploaded_files
                    ]
                    upload_resp = requests.post(f"{API_BASE_URL}/upload", files=files_payload, timeout=120)
                    if upload_resp.status_code == 200:
                        res = upload_resp.json()
                        st.success(f"Indexed **{res.get('new_chunks_count', 0)} new chunks** across {res.get('files_processed', 0)} files!")
                        st.toast(f"Documents indexed into Retrieva! Total chunks: {res.get('total_corpus_chunks', 0)}", icon="📄")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Upload failed ({upload_resp.status_code}): {upload_resp.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend upload: {e}")

    # Indexed Library inventory
    try:
        docs_resp = requests.get(f"{API_BASE_URL}/documents", timeout=2)
        if docs_resp.status_code == 200:
            docs_info = docs_resp.json()
            total_docs = docs_info.get("total_documents", 0)
            total_chunks = docs_info.get("total_chunks", 0)
            with st.expander(f"Indexed Library ({total_docs} files, {total_chunks} chunks)"):
                for d in docs_info.get("documents", []):
                    st.markdown(f"- **{d['filename']}** (`{d['chunks']} chunks`)")
    except Exception:
        pass

    st.divider()

    st.subheader("Observability")
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            health = health_resp.json()
            st.success(f"Backend: Online (v{health.get('version', '2.0')})")
            cache_info = f"Cache: {health.get('cache_backend', 'memory')} ({'Connected' if health.get('cache_connected') else 'Offline'})"
            st.caption(f"Corpus Chunks: {health.get('bm25_corpus_size', 0)} | {cache_info}")
            if health.get("reranker_model"):
                st.caption(f"Reranker: {health.get('reranker_model').split('/')[-1]}")
        else:
            st.warning("Backend reporting warning")
    except Exception:
        st.error("Backend offline (verify Docker services)")

    col1, col2 = st.columns(2)
    with col1:
        st.link_button("Grafana", GRAFANA_URL, use_container_width=True)
    with col2:
        st.link_button("Prometheus", PROMETHEUS_URL, use_container_width=True)

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT AREA ---
st.title("Retrieva: Document Search and QA")
st.markdown("Query ingested documents with hybrid retrieval (BM25 keyword matching and dense embeddings) and telemetry tracking.")

with st.expander("Upload Documents to Retrieva (PDF, DOCX, TXT, MD, CSV, etc.)", expanded=False):
    st.markdown("Add documents to the hybrid index. They will be segmented and immediately indexed for dense vector and BM25 retrieval.")
    main_uploaded_files = st.file_uploader(
        "Select documents to upload",
        type=None,
        accept_multiple_files=True,
        help="Supports PDF, DOCX, TXT, Markdown, CSV, JSON, code, or any document format.",
        key="main_doc_uploader"
    )
    if main_uploaded_files:
        st.write(f"Selected: **{len(main_uploaded_files)} file(s)**")
        if st.button("Index Documents Now", type="primary", key="btn_main_upload"):
            with st.spinner(f"Ingesting and indexing {len(main_uploaded_files)} document(s)..."):
                try:
                    files_payload = [
                        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                        for f in main_uploaded_files
                    ]
                    upload_resp = requests.post(f"{API_BASE_URL}/upload", files=files_payload, timeout=120)
                    if upload_resp.status_code == 200:
                        res = upload_resp.json()
                        st.success(f"Successfully indexed **{res.get('new_chunks_count', 0)} chunks** across {res.get('files_processed', 0)} files! Total corpus chunks: {res.get('total_corpus_chunks', 0)}")
                        st.toast("Documents indexed into Retrieva!", icon="📚")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {upload_resp.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display sources if assistant message
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander(f"Retrieved Context Sources ({len(message['sources'])})"):
                for s_idx, src in enumerate(message["sources"]):
                    st.markdown(f"**Chunk {s_idx+1}** (Relevance Score: `{src.get('score', 0):.4f}`)")
                    st.markdown(f"*{src.get('content', '')}*")
                    if src.get("metadata"):
                        st.caption(f"Source: {src['metadata'].get('source', 'N/A')} | Page: {src['metadata'].get('page', 1)}")

if prompt := st.chat_input("Ask a question about the documents..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with placeholder.container():
            if lottie_ai:
                st_lottie(lottie_ai, height=90, key="loading_ai")
            status_desc = f"Executing {retrieval_mode.upper()} search" + (" + Cross-Encoder Reranking" if rerank_enabled else "")
            st.caption(f"{status_desc} and generating response...")

        try:
            payload = {
                "question": prompt,
                "retriever_type": retrieval_mode,
                "alpha": alpha,
                "top_k": top_k,
                "rerank": rerank_enabled,
                "query_mode": query_mode
            }

            resp = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=30)
            placeholder.empty()

            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "No response generated.")
                sources = data.get("sources", [])
                latency = data.get("latency_seconds", 0.0)
                cached = data.get("cached", False)
                reranked_flag = data.get("reranked", False)
                query_mode_used = data.get("query_mode", "none")
                rewritten_queries = data.get("rewritten_queries")

                # Render Answer
                st.markdown(answer)

                # Metadata & Latency Badges
                cache_text = "Cached" if cached else "Live Query"
                rerank_badge = "<span class='metric-badge' style='background-color:#E8F5E9; color:#2E7D32;'>Rerank: <b>ON</b></span>" if reranked_flag else ""
                qmode_badge = f"<span class='metric-badge' style='background-color:#EDE7F6; color:#4A148C;'>Query Mode: <b>{query_mode_used.upper()}</b></span>" if query_mode_used != "none" else ""
                st.markdown(
                    f"<span class='metric-badge'>Search: <b>{retrieval_mode}</b></span>"
                    f"<span class='metric-badge'>Latency: <b>{latency:.3f}s</b></span>"
                    f"<span class='metric-badge'>{cache_text}</span>"
                    f"{rerank_badge}"
                    f"{qmode_badge}",
                    unsafe_allow_html=True
                )

                # Show rewritten / expanded queries if present
                if rewritten_queries:
                    with st.expander(f"Query Variants Generated ({len(rewritten_queries)})"):
                        for q_idx, q_var in enumerate(rewritten_queries, 1):
                            st.markdown(f"**Variant {q_idx}:** `{q_var}`")

                # Source Citations
                if sources:
                    with st.expander(f"Retrieved Context ({len(sources)} chunks)"):
                        for s_idx, src in enumerate(sources):
                            score_label = "Rerank Score" if reranked_flag else "Score"
                            st.markdown(f"**Chunk {s_idx+1}** - {score_label}: `{src.get('score', 0):.4f}`")
                            st.markdown(f"> {src.get('content')}")
                            if src.get("metadata"):
                                st.caption(f"File: {src['metadata'].get('source', 'N/A')} | Page: {src['metadata'].get('page', 1)} | ID: {src['metadata'].get('chunk_id', 'N/A')}")

                # Feedback Buttons
                col_f1, col_f2, _ = st.columns([1.2, 1.2, 7.6])
                with col_f1:
                    if st.button("Helpful", key=f"up_{len(st.session_state.messages)}"):
                        requests.post(f"{API_BASE_URL}/feedback", json={"query": prompt, "answer": answer, "rating": "positive"})
                        st.toast("Feedback recorded.")
                with col_f2:
                    if st.button("Unhelpful", key=f"down_{len(st.session_state.messages)}"):
                        requests.post(f"{API_BASE_URL}/feedback", json={"query": prompt, "answer": answer, "rating": "negative"})
                        st.toast("Feedback recorded.")

                # Store in session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            else:
                st.error(f"Error {resp.status_code}: {resp.text}")

        except Exception as e:
            placeholder.empty()
            st.error(f"Failed to communicate with API: {e}")
