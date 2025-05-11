import streamlit as st
from rag_core import load_faiss_resources, generate_answer

# === Load FAISS and metadata once ===
if "faiss_index" not in st.session_state:
    index, chunk_ids, chunk_id_to_info = load_faiss_resources()
    st.session_state.faiss_index = index
    st.session_state.chunk_ids = chunk_ids
    st.session_state.chunk_id_to_info = chunk_id_to_info

# --- Set page configuration ---
st.set_page_config(page_title="IPCC Assistant", page_icon="💬", layout="wide")

# --- Init session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "source_chunks" not in st.session_state:
    st.session_state.source_chunks = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# --- Custom CSS ---
st.markdown("""
<style>
/* Fixed header */
#main > div:first-child {
    position: fixed;
    top: 0;
    left: 0;
    z-index: 100;
    background-color: white;
    width: 100%;
    padding: 10px 25px;
    border-bottom: 1px solid #e0e0e0;
}
.fixed-header {
    display: flex;
    align-items: center;
}
.fixed-header img {
    height: 40px;
    margin-right: 10px;
}
.fixed-header h1 {
    font-size: 24px;
    margin: 0;
}

/* Scrollable sections */
.scrollable-left, .scrollable-right {
    height: calc(100vh - 100px);
    overflow-y: auto;
    padding-top: 80px;
}
.chat-bubble {
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    max-width: 90%;
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
.source-bubble {
    background-color: #f0f0f0;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
}
.fixed-input {
    position: fixed;
    bottom: 15px;
    left: 30px;
    width: 60%;
    background: white;
    z-index: 99;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="fixed-header">
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ4mGN0EMyYSpmOiimWh8AwFitk5pUFAcKVLA&s" />
    <h1>IPCC Assistant</h1>
</div>
""", unsafe_allow_html=True)

# --- Layout: 2 columns ---
col1, col2 = st.columns([2.5, 1], gap="large")

with col1:
    with st.container():
        st.markdown('<div class="scrollable-left">', unsafe_allow_html=True)

        # Render chat history
        for user_msg, assistant_msg in st.session_state.chat_history:
            st.markdown(f"""
                <div class="chat-row user-row">
                    <div class="chat-bubble user-bubble">{user_msg}</div>
                </div>
                <div class="chat-row bot-row">
                    <div class="chat-bubble bot-bubble">{assistant_msg}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Fixed input at bottom
        with st.container():
            st.markdown('<div class="fixed-input">', unsafe_allow_html=True)
            user_input = st.text_input("Ask something about climate reports...", key="user_input", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

        if user_input and user_input.strip():
            st.session_state.current_question = user_input.strip()
            st.session_state.chat_history.append((user_input.strip(), ""))
            st.rerun()

        # Generate answer if needed
        if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "":
            question = st.session_state.chat_history[-1][0]

            with st.spinner("Thinking..."):
                response = generate_answer(
                    question,
                    st.session_state.faiss_index,
                    st.session_state.chunk_ids,
                    st.session_state.chunk_id_to_info
                )

            answer_text = response["answer"]
            st.session_state.chat_history[-1] = (question, answer_text)

            # Accumulate sources
            existing_ids = {c["chunk_id"] for c in st.session_state.source_chunks}
            for chunk in response["chunks"]:
                if chunk["chunk_id"] not in existing_ids:
                    st.session_state.source_chunks.append(chunk)

            st.rerun()

with col2:
    st.markdown('<div class="scrollable-right">', unsafe_allow_html=True)
    st.markdown("## 📚 Sources")
    if st.session_state.source_chunks:
        for i, chunk in enumerate(sorted(st.session_state.source_chunks, key=lambda x: x["distance"], reverse=True), start=1):
            page_info = chunk["metadata"].get("source", "")
            page_num = int(page_info.replace("page_", "")) if "page_" in page_info else "?"
            similarity = round(chunk["distance"] * 100, 1)
            st.markdown(f"""
            <div class="source-bubble">
                <strong>Doc {i} – IPCC_AR6_WGIII_FullReport – Page {page_num}</strong><br>
                <small>Relevance: {similarity}%</small><br>
                <a href="https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_FullReport.pdf#page={page_num}" target="_blank">
                    Open original page
                </a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("Ask a question to view source documents.")
    st.markdown('</div>', unsafe_allow_html=True)
