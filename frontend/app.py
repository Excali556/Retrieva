import os
import streamlit as st
import requests
import time
from streamlit_lottie import st_lottie

# --- API CONFIG ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://chatbot-api:8000")
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

    st.divider()

    st.subheader("Observability")
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if health_resp.status_code == 200:
            health = health_resp.json()
            st.success(f"Backend: Online (v{health.get('version', '2.0')})")
            cache_info = f"Cache: {health.get('cache_backend', 'memory')} ({'Connected' if health.get('cache_connected') else 'Offline'})"
            st.caption(f"Corpus Chunks: {health.get('bm25_corpus_size', 0)} | {cache_info}")
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
            st.caption(f"Executing {retrieval_mode.upper()} search and generating response...")

        try:
            payload = {
                "question": prompt,
                "retriever_type": retrieval_mode,
                "alpha": alpha,
                "top_k": top_k
            }

            resp = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=30)
            placeholder.empty()

            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "No response generated.")
                sources = data.get("sources", [])
                latency = data.get("latency_seconds", 0.0)
                cached = data.get("cached", False)

                # Render Answer
                st.markdown(answer)

                # Metadata & Latency Badges
                cache_text = "Cached" if cached else "Live Query"
                st.markdown(
                    f"<span class='metric-badge'>Mode: <b>{retrieval_mode}</b></span>"
                    f"<span class='metric-badge'>Latency: <b>{latency:.3f}s</b></span>"
                    f"<span class='metric-badge'>{cache_text}</span>",
                    unsafe_allow_html=True
                )

                # Source Citations
                if sources:
                    with st.expander(f"Retrieved Context ({len(sources)} chunks)"):
                        for s_idx, src in enumerate(sources):
                            st.markdown(f"**Chunk {s_idx+1}** - Score: `{src.get('score', 0):.4f}`")
                            st.markdown(f"> {src.get('content')}")
                            if src.get("metadata"):
                                st.caption(f"File: {src['metadata'].get('source', 'N/A')} | ID: {src['metadata'].get('chunk_id', 'N/A')}")

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
