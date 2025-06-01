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

# --- Initialize storage for timings (if not already set) ---
if "last_timings" not in st.session_state:
    st.session_state.last_timings = None

# --- Set page configuration ---
st.set_page_config(page_title="IPCC Assistant", page_icon="💬", layout="wide")

# --- Custom CSS for chat bubbles and source‐cards ---
st.markdown(
    """
    <style>
    /* Chat‐bubble styling */
    .chat-bubble {
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        max-width: 70%;
        display: inline-block;
        word-wrap: break-word;
        font-family: inherit;
        font-size: 14px;
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
        font-family: inherit;
    }
    input[type="text"]:focus {
        border: 2px solid #81C244 !important;
        outline: none !important;
    }

    /* Timing‐bubble styling */
    .timing-bubble {
        border-radius: 12px;
        background-color: #f0f0f0;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.5rem;
        max-width: 60%;
        font-size: 12px;
        color: #555;
        font-family: inherit;
    }

    /* Sources strip styling */
    .sources-container {
        display: flex;
        flex-direction: row;
        overflow-x: auto;
        overflow-y: hidden;
        white-space: nowrap;
        padding: 1rem;
        background-color: #f0f0f0;
        border-top: 2px solid #ddd;
    }
    .source-card {
        display: inline-flex;
        flex-direction: column;
        justify-content: space-between;
        width: 320px;
        height: 180px;               /* Fixed height */
        background: #fff;
        margin-right: 12px;
        padding: 12px;
        border: 1px solid #ccc;
        border-radius: 10px;
        flex-shrink: 0;
        font-family: inherit;
        font-size: 14px;
        line-height: 1.3;
        overflow: hidden;            /* Ensure nothing can escape the box */
    }
    .source-card-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .source-card-snippet {
        flex-grow: 1;
        overflow: hidden;            /* Clip excess lines */
        margin-bottom: 0.5rem;
    }
    .source-card-footer {
        font-size: 13px;
    }
    .source-card a {
        text-decoration: none;
        color: #0066cc;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Initialize session state for chat history & sources ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# --- Header ---
st.markdown(
    """<h1 style="text-align: center; font-size: 50px; margin-bottom: 2rem; font-family: inherit;">IPCC Assistant</h1>""",
    unsafe_allow_html=True,
)

# --- Chat input (appears at bottom) ---
question = st.chat_input("Ask something about climate reports...")

# === Submit new question ===
if question and question.strip():
    # Reset last_timings so we don’t accidentally show stale timings
    st.session_state.last_timings = None
    st.session_state.chat_history.append((question, "..."))  # Placeholder for streaming

# === Chat display ===
for user_msg, bot_msg in st.session_state.chat_history:

    def linkify_refs(text):
        # Turn (1), (2), etc. into clickable anchors
        return re.sub(
            r"\((\d+)\)",
            r'<a href="#ref\1" style="text-decoration:none;color:#0077cc;">(\1)</a>',
            text,
        )

    bot_msg_html = linkify_refs(bot_msg)

    # Display user’s message
    st.markdown(
        f"""
        <div class="chat-row user-row">
            <div class="chat-bubble user-bubble">{user_msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display bot’s response
    st.markdown(
        f"""
        <div class="chat-row bot-row">
            <div class="chat-bubble bot-bubble">{bot_msg_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# If we have a stored timing dict, render it once as a “timing‐bubble”
if st.session_state.last_timings is not None:
    t = st.session_state.last_timings
    st.markdown(
        f"""
        <div class="chat-row bot-row">
            <div class="timing-bubble">
                ⏱ Contextualization: {t.get('contextualization', 0):.2f}s<br>
                ⏱ Paraphrase Gen: {t.get('paraphrase_generation', 0):.2f}s<br>
                ⏱ Retrieval+Rerank: {t.get('retrieval_and_rerank', 0):.2f}s<br>
                ⏱ LLM Generation: {t.get('generation', 0):.2f}s
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Clear it so we only show once
    st.session_state.last_timings = None

# === Generate response logic ===
if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "...":
    question = st.session_state.chat_history[-1][0]

    # Build a short “memory” of the last couple of turns
    memory = []
    for user_msg, bot_msg in st.session_state.chat_history[-3:-1]:
        memory.append({"user": user_msg, "assistant": bot_msg})

    # Show a fake “thinking...” progress bar
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

    # Call the rag_core generate_answer
    response = generate_answer(
        query=question,
        index=st.session_state.faiss_index,
        chunk_ids=st.session_state.chunk_ids,
        chunk_id_to_info=st.session_state.chunk_id_to_info,
        k=4,
        window=1,
        rerank_top_n=6,
        chat_history=memory,
    )

    # Store timings in session_state so we can render them in the UI
    st.session_state.last_timings = response.get("timings", {})

    answer_full = response["answer"]
    # Force the progress bar to 100%
    progress_placeholder.markdown(
        progress_bar_html.replace(f"width: {pct}%", "width: 100%").replace(f"{pct}%", "100%"),
        unsafe_allow_html=True,
    )
    time.sleep(0.3)
    progress_placeholder.empty()

    # Stream the assistant’s answer word by word
    placeholder = st.empty()
    streamed = ""
    for word in answer_full.split():
        streamed += word + " "
        placeholder.markdown(
            f"""
            <div class="chat-row bot-row">
                <div class="chat-bubble bot-bubble">{streamed}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.02)

    # Replace the placeholder with the full answer
    st.session_state.chat_history[-1] = (question, answer_full)
    st.session_state.last_sources = response["chunks"]
    st.rerun()

# === Collapsible, horizontally scrollable “Sources” strip ===
if st.session_state.last_sources:
    with st.expander("Show Sources", expanded=True):
        # Build ONE big HTML block (including CSS + cards with anchors)
        html = """
        <style>
          .sources-container {
            display: flex;
            flex-direction: row;
            overflow-x: auto;
            overflow-y: hidden;
            white-space: nowrap;
            padding: 1rem;
            background-color: #f0f0f0;
            border-top: 2px solid #ddd;
          }
          .source-card {
            display: inline-flex;
            flex-direction: column;
            justify-content: space-between;
            width: 320px;
            height: 180px;             /* Fixed height */
            background: #fff;
            margin-right: 12px;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 10px;
            flex-shrink: 0;
            font-family: inherit;
            font-size: 14px;
            line-height: 1.3;
            overflow: hidden;          /* Clip anything outside */
          }
          .source-card-header {
            font-weight: bold;
            margin-bottom: 0.5rem;
          }
          .source-card-snippet {
            flex-grow: 1;
            overflow: hidden;          /* Clip excess lines */
            margin-bottom: 0.5rem;
          }
          .source-card-footer {
            font-size: 13px;
          }
          .source-card a {
            text-decoration: none;
            color: #0066cc;
            font-size: 13px;
          }
        </style>
        <div class="sources-container">
        """

        for i, chunk in enumerate(st.session_state.last_sources):
            meta = chunk.get("metadata", {})
            page = meta.get("source", "")
            report = meta.get("report_name", "Unknown Report")
            page_number = int(page.replace("page_", "")) if "page_" in page else "?"
            score = chunk.get("reranker_score", 0.0)
            rel = round((score + 10) * 5, 1)
            ref = chunk.get("reference", f"({i+1})")

            # Create an anchor target called "ref{i+1}"
            html += f'<div class="source-card" id="ref{i+1}">'

            # Header line
            html += f'<div class="source-card-header">{ref} – {report} – Page {page_number}</div>'

            # Snippet (max 250 chars), clipped if too long
            raw_text = chunk["text"]
            snippet = raw_text[:250].replace("\n", " ")
            if len(raw_text) > 250:
                snippet = snippet.rstrip() + "…"
            html += f'<div class="source-card-snippet">{snippet}</div>'

            # Footer: relevancy + link
            pdf_url = (
                "https://www.ipcc.ch/report/ar6/wg3/downloads/report/"
                "IPCC_AR6_WGIII_FullReport.pdf"
                f"#page={page_number}"
            )
            html += (
                f'<div class="source-card-footer">'
                f'<i>Relevancy: {rel}%</i><br>'
                f'<a href="{pdf_url}" target="_blank">Open PDF</a>'
                f'</div>'
            )

            html += "</div>"

        html += "</div>"

        # Render it via st.markdown so that anchor links (#refN) work on the same page
        st.markdown(html, unsafe_allow_html=True)
