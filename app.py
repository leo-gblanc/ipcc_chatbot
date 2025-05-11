import streamlit as st
import time
from rag_core import load_faiss_resources, generate_answer  # Your backend code

# === Load FAISS and metadata once ===
if "faiss_index" not in st.session_state:
    index, chunk_ids, chunk_id_to_info = load_faiss_resources()
    st.session_state.faiss_index = index
    st.session_state.chunk_ids = chunk_ids
    st.session_state.chunk_id_to_info = chunk_id_to_info

# --- Set page configuration ---
st.set_page_config(page_title="IPCC Assistant", page_icon="💬")

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

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Header ---
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 2rem;">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4mGN0EMyYSpmOiimWh8AwFitk5pUFAcKVLA&s" 
             width="80" style="margin-right: 15px;">
        <h1 style="margin: 0; font-size: 50px;">IPCC Assistant</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Render chat history ---
for user_msg, assistant_msg in st.session_state.chat_history:
    st.markdown(f"""
        <div class="chat-row user-row">
            <div class="chat-bubble user-bubble">{user_msg}</div>
        </div>
        <div class="chat-row bot-row">
            <div class="chat-bubble bot-bubble">{assistant_msg}</div>
        </div>
    """, unsafe_allow_html=True)

# --- Chat input ---
question = st.chat_input("Ask something about climate reports...")

# --- On message sent ---
if question and question.strip():
    # Save user message with empty bot placeholder
    st.session_state.chat_history.append((question, ""))
    st.rerun()

# --- Generate answer if needed ---
if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "":
    question = st.session_state.chat_history[-1][0]

    # Show thinking animation
    with st.spinner("Thinking..."):
        response = generate_answer(
            question,
            st.session_state.faiss_index,
            st.session_state.chunk_ids,
            st.session_state.chunk_id_to_info
        )

    # Format the final response
    answer_text = response["answer"]
    st.session_state.chat_history[-1] = (question, answer_text)
    st.rerun()
