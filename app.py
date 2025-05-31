import sys
import types
import os
import time
from rag_core import load_faiss_resources, generate_answer
import re
import streamlit as st

# === Load FAISS and metadata once ===
if "faiss_index" not in st.session_state:
    index, chunk_ids, chunk_id_to_info = load_faiss_resources()
    st.session_state.faiss_index = index
    st.session_state.chunk_ids = chunk_ids
    st.session_state.chunk_id_to_info = chunk_id_to_info

# --- Set page configuration ---
st.set_page_config(page_title="IPCC Assistant", page_icon="💬", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .chat-bubble {
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        max-width: 70%;
        display: inline-block;
        word-wrap: break-word;
    }
    .user-bubble {
        background-color: #f6f6f6;
        border-right: 6px solid black;
        margin-left: auto;
    }
    .bot-bubble {
        background-color: #eaf7e0;
        border-left: 6px solid #81C244;
        margin-right: auto;
    }
    .chat-row {
        display: flex;
        margin-bottom: 0.5rem;
    }
    .user-row {
        justify-content: flex-end;
    }
    .bot-row {
        justify-content: flex-start;
    }
    input[type="text"] {
        border: 1px solid #000 !important;
        border-radius: 999px !important;
    }
    input[type="text"]:focus {
        border: 2px solid #81C244 !important;
        outline: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# --- Header ---
st.markdown(
    """<h1 style="text-align: center; font-size: 50px; margin-bottom: 2rem;">IPCC Assistant</h1>""",
    unsafe_allow_html=True
)

# --- Chat input ---
question = st.chat_input("Ask something about climate reports...")

# === Submit new question ===
if question and question.strip():
    st.session_state.chat_history.append((question, ""))  # Add blank answer
    st.rerun()

# === Generate response ===
if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "":
    question = st.session_state.chat_history[-1][0]

    # Memory: up to 2 last Q&A pairs
    memory = []
    for user_msg, bot_msg in st.session_state.chat_history[-3:-1]:
        memory.append({"user": user_msg, "assistant": bot_msg})

    with st.spinner("Thinking..."):
        response = generate_answer(
            query=question,
            index=st.session_state.faiss_index,
            chunk_ids=st.session_state.chunk_ids,
            chunk_id_to_info=st.session_state.chunk_id_to_info,
            k=4,
            window=1,
            rerank_top_n=6,
            chat_history=memory  # <= enables contextual follow-ups
        )

    st.session_state.chat_history[-1] = (question, response["answer"])
    st.session_state.last_sources = response["chunks"]
    st.rerun()

# === Layout: Chat on left, sources on right ===
col1, col2 = st.columns([2, 1], gap="large")

# --- Left: Chat display ---
with col1:
    for user_msg, bot_msg in st.session_state.chat_history:
        def linkify_refs(text):
            return re.sub(r"\((\d+)\)", r"([\1](#ref\1))", text)

        bot_msg_linked = linkify_refs(bot_msg)

        st.markdown(f"""
            <div class="chat-row user-row">
                <div class="chat-bubble user-bubble">{user_msg}</div>
            </div>
            <div class="chat-row bot-row">
                <div class="chat-bubble bot-bubble">{bot_msg_linked}</div>
            </div>
        """, unsafe_allow_html=True)

# --- Right: Source display ---
with col2:
    st.markdown("### 🔎 Sources")
    for i, chunk in enumerate(st.session_state.last_sources):
        meta = chunk.get("metadata", {})
        page = meta.get("source", "")
        report = meta.get("report_name", "Unknown Report")
        page_number = int(page.replace("page_", "")) if "page_" in page else "?"
        score = chunk.get("reranker_score", 0.0)
        rel = round((score + 10) * 5, 1)
        ref = chunk.get("reference", f"({i+1})")

        pdf_url = f"https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_FullReport.pdf#page={page_number}"

        st.markdown(f"""
        <a id="ref{i+1}"></a>
        <div style="background-color: #f9f9f9; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 10px;">
            <b>{ref} – {report} – Page {page_number}</b><br>
            <p style="font-size: 14px;">{chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}</p>
            <p><i>Relevancy score: {rel}%</i></p>
            🔗 <a href="{pdf_url}" target="_blank">Open PDF</a>
        </div>
        """, unsafe_allow_html=True)
