import sys
import types
import os
import time
from rag_core import load_faiss_resources, generate_answer
import re
import streamlit as st

# ==============================================================================
# === CONSTANTS ===
# ==============================================================================
# We assume `user_guide.pdf` is placed at the root of this repo.
# On Streamlit Cloud, files in the repo root are served at "/<filename>".
USER_GUIDE_URL = "/user_guide.pdf"

# ==============================================================================
# === LOAD FAISS + METADATA ONCE ===
# ==============================================================================
if "faiss_index" not in st.session_state:
    index, chunk_ids, chunk_id_to_info = load_faiss_resources()
    st.session_state.faiss_index = index
    st.session_state.chunk_ids = chunk_ids
    st.session_state.chunk_id_to_info = chunk_id_to_info

# ==============================================================================
# === INITIALIZE STORAGE FOR TIMINGS ===
# ==============================================================================
if "last_timings" not in st.session_state:
    st.session_state.last_timings = None

# ==============================================================================
# === STREAMLIT PAGE CONFIGURATION ===
# ==============================================================================
st.set_page_config(page_title="Climate Impact Assistant", page_icon="💬", layout="wide")

# ==============================================================================
# === GLOBAL CSS STYLING ===
# ==============================================================================
st.markdown(
    """
    <style>
    /* ================= Welcome‐banner styling ================= */
    .welcome-banner {
        max-width: 700px;
        color: #777777;        /* greyed out for discrete look */
        font-family: inherit;
        font-size: 16px;
        margin: auto;
        margin-bottom: 2rem;
    }

    /* ================= Chat‐bubble styling ================= */
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

    /* ================= Timing‐bubble styling ================= */
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

    /* ================= Sources strip styling ================= */
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
        justify-content: flex-start;
        width: 320px;
        height: 190px;            /* Fixed overall card height */
        background: #fff;
        margin-right: 12px;
        padding: 12px;
        border: 1px solid #ccc;
        border-radius: 10px;
        flex-shrink: 0;
        font-family: inherit;
        font-size: 14px;
        line-height: 1.3;
    }
    .source-card-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .source-card-snippet {
        flex-grow: 1;
        overflow-y: auto;         /* Enables vertical scrolling inside the card */
        margin-bottom: 0.5rem;
        white-space: normal;      /* Allow wrapping */
    }
    .source-card-footer {
        font-size: 13px;
        margin-top: auto;         /* Stick to bottom */
    }
    .source-card a {
        text-decoration: none;
        color: #0066cc;
        font-size: 13px;
    }

    /* ================= User‐Guide Button styling ================= */
    .user-guide-btn {
        background-color: #81C244;
        color: white !important;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        text-align: center;
        font-size: 14px;
        font-family: inherit;
        text-decoration: none;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .user-guide-btn:hover {
        background-color: #6DAF38;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# === INITIALIZE SESSION STATE FOR HISTORY & SOURCES ===
# ==============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# ==============================================================================
# === HEADER & WELCOME BANNER ===
# ==============================================================================
st.markdown(
    """<h1 style="text-align: center; font-size: 50px; margin-bottom: 1rem; font-family: inherit;">
    Climate Impact Assistant</h1>""",
    unsafe_allow_html=True,
)

st.markdown(
    """<div class="welcome-banner">
    <strong>Welcome to the Climate Impact Assistant!</strong><br>
    Ask me how different climate scenarios affect sectors like energy, transport, or agriculture.
    I provide clear, sourced answers on risks, strategies, and impacts across industries.
    You can explore anything from technologies to policy options.
    Need help? Check the user guide in the bottom right corner.
    </div>""",
    unsafe_allow_html=True,
)

# ==============================================================================
# === CHAT DISPLAY (SHOW HISTORY) ===
# ==============================================================================
for user_msg, bot_msg in st.session_state.chat_history:

    def linkify_refs(text):
        """
        Find any '(6)', '(11)', or '(6, 11)' etc., 
        and turn each number into its own clickable anchor.
        E.g. '(6, 11)' → '<a href="#ref6">(6)</a> <a href="#ref11">(11)</a>'.
        """
        def _replace(match):
            nums = match.group(1).split(",")
            anchors = []
            for num in nums:
                num = num.strip()
                anchors.append(
                    f'<a href="#ref{num}" style="text-decoration:none;color:#0077cc;">({num})</a>'
                )
            return " ".join(anchors)

        return re.sub(r"\((\d+(?:\s*,\s*\d+)*)\)", _replace, text)

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

    # Display bot’s response (with each reference now its own anchor)
    st.markdown(
        f"""
        <div class="chat-row bot-row">
            <div class="chat-bubble bot-bubble">{bot_msg_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# === TIMING BUBBLE (uncomment if wanted) ===
# ==============================================================================
# if st.session_state.last_timings is not None:
#     t = st.session_state.last_timings
#     st.markdown(
#         f"""
#         <div class="chat-row bot-row">
#             <div class="timing-bubble">
#                 ⏱ Contextualization: {t.get('contextualization', 0):.2f}s<br>
#                 ⏱ Paraphrase Gen: {t.get('paraphrase_generation', 0):.2f}s<br>
#                 ⏱ Retrieval+Rerank: {t.get('retrieval_and_rerank', 0):.2f}s<br>
#                 ⏱ LLM Generation: {t.get('generation', 0):.2f}s
#             </div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )
#     # Clear so it only shows once
#     st.session_state.last_timings = None

# ==============================================================================
# === GENERATE RESPONSE LOGIC (WITH SPINNER) ===
# ==============================================================================
if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "...":
    question = st.session_state.chat_history[-1][0]

    # Build a short “memory” of the last couple of turns
    memory = []
    for user_msg, bot_msg in st.session_state.chat_history[-3:-1]:
        memory.append({"user": user_msg, "assistant": bot_msg})

    # Call generate_answer() inside a spinner so the user sees “Thinking…”
    with st.spinner("Thinking…"):
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

    # Store the returned timing breakdown so we can render it above
    st.session_state.last_timings = response.get("timings", {})

    answer_full = response["answer"]

    # Stream the assistant’s answer paragraph by paragraph
    placeholder = st.empty()
    streamed = ""
    for para in answer_full.split("\n"):
        # Append this entire paragraph (with its trailing newline) at once
        streamed += para + "\n"

        # Render the partial text inside a chat bubble
        placeholder.markdown(
            f"""
            <div class="chat-row bot-row">
                <div class="chat-bubble bot-bubble">{streamed}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Pause briefly so that the user sees each paragraph appear in turn
        time.sleep(0.1)

    # Replace the placeholder with the full answer (so chat_history is updated)
    st.session_state.chat_history[-1] = (question, answer_full)
    st.session_state.last_sources = response["chunks"]
    st.rerun()

# ==============================================================================
# === COLLAPSIBLE, HORIZONTALLY SCROLLABLE “SOURCES” STRIP ===
# ==============================================================================
if st.session_state.last_sources:
    # Expander collapsed by default (expanded=False)
    with st.expander("Show Sources", expanded=False):
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
            justify-content: flex-start;
            width: 320px;
            height: 190px;            /* Fixed overall card height */
            background: #fff;
            margin-right: 12px;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 10px;
            flex-shrink: 0;
            font-family: inherit;
            font-size: 14px;
            line-height: 1.3;
          }
          .source-card-header {
            font-weight: bold;
            margin-bottom: 0.5rem;
          }
          .source-card-snippet {
            flex-grow: 1;
            overflow-y: auto;         /* Enables vertical scrolling inside the card */
            margin-bottom: 0.5rem;
            white-space: normal;      /* Allow wrapping */
          }
          .source-card-footer {
            font-size: 13px;
            margin-top: auto;         /* Stick to bottom */
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

            # Each card has an anchor ID "ref{i+1}" so the inline link can jump here
            html += f'<div class="source-card" id="ref{i+1}">'

            # Header
            html += f'<div class="source-card-header">{ref} – {report} – Page {page_number}</div>'

            # FULL chunk text (no truncation), scrollable
            raw_text = chunk["text"].replace("\n", " ")
            html += f'<div class="source-card-snippet">{raw_text}</div>'

            # Footer: relevancy & PDF link
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

# ==============================================================================
# === INPUT ROW: CHAT_INPUT + “User Guide” BUTTON (AT BOTTOM) ===
# ==============================================================================
col1, col2 = st.columns([0.85, 0.15])

with col1:
    question = st.chat_input("Ask something about climate reports…", key="bottom_chat_input")

with col2:
    st.markdown(
        f'<a href="{USER_GUIDE_URL}" target="_blank" class="user-guide-btn">User Guide</a>',
        unsafe_allow_html=True,
    )

# ==============================================================================
# === SUBMIT NEW QUESTION (BOTTOM INPUT) ===
# ==============================================================================
if question and question.strip():
    # Reset last_timings so we don’t show stale data
    st.session_state.last_timings = None
    st.session_state.chat_history.append((question, "..."))
