import streamlit as st
from rag_core import load_faiss_resources, generate_answer

st.set_page_config(page_title="RAG IPCC Chatbot", layout="wide")

st.title("🌍 IPCC Report Q&A Assistant")

query = st.text_input("Enter your question:", placeholder="e.g., What is the impact of 2°C warming on agriculture?")

if query:
    with st.spinner("Processing your question..."):
        index, chunk_ids, chunk_id_to_info = load_faiss_resources()
        result = generate_answer(query, index, chunk_ids, chunk_id_to_info)
    
    st.subheader("🧠 Answer:")
    st.markdown(result["answer"])

    with st.expander("📄 Context chunks used"):
        for chunk in result["chunks"]:
            st.markdown(f"**Page:** {chunk['metadata'].get('source', 'N/A')}")
            st.write(chunk["text"])
            st.markdown("---")