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
    .sources-container {
        max-height: 250px;
        overflow-x: auto;
        overflow-y: hidden;
        white-space: nowrap;
        padding: 1rem;
        background-color: #f0f0f0;
        border-top: 2px solid #ddd;
    }
    .source-card {
        display: inline-block;
        width: 320px;
        height: 100%;
        vertical-align: top;
        background: #fff;
        margin-right: 12px;
        padding: 12px;
        border: 1px solid #ccc;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# --- Header ---
st.markdown("""<h1 style="text-align: center; font-size: 50px; margin-bottom: 2rem;">IPCC Assistant</h1>""", unsafe_allow_html=True)

# --- Chat input ---
question = st.chat_input("Ask something about climate reports...")

# === Submit new question ===
if question and question.strip():
    st.session_state.chat_history.append((question, "..."))  # Placeholder

# === Chat display ===
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

# === Generate response ===
if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "...":
    question = st.session_state.chat_history[-1][0]

    memory = []
    for user_msg, bot_msg in st.session_state.chat_history[-3:-1]:
        memory.append({"user": user_msg, "assistant": bot_msg})

    progress_placeholder = st.empty()
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed < 30:
            pct = int((elapsed / 30) * 80)
        elif elapsed < 40:
            pct = 80 + int(((elapsed - 30) / 10) * 10)
        elif elapsed < 50:
            pct = 90 + int(((elapsed - 40) / 10) * 5)
        elif elapsed < 60:
            pct = 95 + int(((elapsed - 50) / 10) * 3)
        elif elapsed < 75:
            pct = 98 + int(((elapsed - 60) / 15) * 1)
        else:
            pct = 99

        progress_bar_html = f"""
            <div class="loader-bar">
                <div class="loader-fill" style="width: {pct}%;"></div>
            </div>
            <div style="text-align:center; font-style: italic; font-size: 14px;">Thinking... {pct}%</div>
            <style>
                .loader-bar {{
                    width: 60%;
                    margin: auto;
                    height: 8px;
                    background-color: #e0e0e0;
                    border-radius: 8px;
                    overflow: hidden;
                    margin-top: 8px;
                    margin-bottom: 12px;
                }}
                .loader-fill {{
                    height: 100%;
                    background-color: #81C244;
                    transition: width 0.3s ease-in-out;
                }}
            </style>
        """
        progress_placeholder.markdown(progress_bar_html, unsafe_allow_html=True)

        if elapsed >= 75:
            break
        time.sleep(0.25)

    response = generate_answer(
        query=question,
        index=st.session_state.faiss_index,
        chunk_ids=st.session_state.chunk_ids,
        chunk_id_to_info=st.session_state.chunk_id_to_info,
        k=4,
        window=1,
        rerank_top_n=6,
        chat_history=memory
    )

    answer_full = response["answer"]
    progress_placeholder.markdown(progress_bar_html.replace(f"width: {pct}%", "width: 100%").replace(f"{pct}%", "100%"), unsafe_allow_html=True)
    time.sleep(0.3)
    progress_placeholder.empty()

    placeholder = st.empty()
    streamed = ""
    for word in answer_full.split():
        streamed += word + " "
        placeholder.markdown(f"""
            <div class="chat-row bot-row">
                <div class="chat-bubble bot-bubble">{streamed}</div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(0.02)

    st.session_state.chat_history[-1] = (question, answer_full)
    st.session_state.last_sources = response["chunks"]
    st.rerun()

# === Independent Scrollable Sources Section ===
if st.session_state.last_sources:
    if st.toggle("Show Sources", value=True):
        st.markdown("<div class='sources-container'>", unsafe_allow_html=True)
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
            <div class='source-card'>
                <b>{ref} – {report} – Page {page_number}</b><br>
                <p style='font-size: 14px;'>{chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}</p>
                <p><i>Relevancy score: {rel}%</i></p>
                🔗 <a href="{pdf_url}" target="_blank">Open PDF</a>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
