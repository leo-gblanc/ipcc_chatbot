import streamlit as st
import time
from rag_core import load_faiss_resources, generate_answer  # Your RAG backend logic

# === Load FAISS and metadata once ===
if "faiss_index" not in st.session_state:
    index, chunk_ids, chunk_id_to_info = load_faiss_resources()
    st.session_state.faiss_index = index
    st.session_state.chunk_ids = chunk_ids
    st.session_state.chunk_id_to_info = chunk_id_to_info

# --- Set page configuration ---
st.set_page_config(page_title="IPCC Assistant", page_icon="💬", layout="wide")

# --- Custom CSS ---
st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# --- Header ---
st.markdown(
    """
    <h1 style="text-align: center; font-size: 50px; margin-bottom: 2rem;">IPCC Assistant</h1>
    """,
    unsafe_allow_html=True
)

# --- Chat input ---
question = st.chat_input("Ask something about climate reports...")

# === On question submit ===
if question and question.strip():
    st.session_state.chat_history.append((question, ""))  # Save question with empty answer
    st.rerun()

# === Handle answer generation ===
if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "":
    question = st.session_state.chat_history[-1][0]

    with st.spinner("Thinking..."):
        response = generate_answer(
            question,
            st.session_state.faiss_index,
            st.session_state.chunk_ids,
            st.session_state.chunk_id_to_info,
            k=5,
            window=1  # Retrieve neighbors
        )

    st.session_state.chat_history[-1] = (question, response["answer"])
    st.session_state.last_sources = response["chunks"]
    st.rerun()

# === Layout with chat on the left and sources on the right ===
col1, col2 = st.columns([2, 1], gap="large")

# --- Left: Chat bubbles ---
with col1:
    for user_msg, assistant_msg in st.session_state.chat_history:
        st.markdown(f"""
            <div class="chat-row user-row">
                <div class="chat-bubble user-bubble">{user_msg}</div>
            </div>
            <div class="chat-row bot-row">
                <div class="chat-bubble bot-bubble">{assistant_msg}</div>
            </div>
        """, unsafe_allow_html=True)

# --- Right: Sources used ---
with col2:
    st.markdown("### 🔎 Sources")
    for i, chunk in enumerate(st.session_state.last_sources):
        meta = chunk.get("metadata", {})
        page = meta.get("source", "")
        report = meta.get("report_name", "Unknown Report")
        page_number = int(page.replace("page_", "")) if "page_" in page else "?"
        similarity = round(chunk["distance"] * 100, 1)  # IndexFlatIP with normalized vectors

        pdf_url = f"https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_FullReport.pdf#page={page_number}"

        st.markdown(f"""
        <div style="background-color: #f9f9f9; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 10px;">
            <b>Doc {i+1} – {report} – Page {page_number}</b><br>
            <p style="font-size: 14px;">{chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}</p>
            <p><i>Similarity score: {similarity}%</i></p>
            🔗 <a href="{pdf_url}" target="_blank">Open PDF</a>
        </div>
        """, unsafe_allow_html=True)
