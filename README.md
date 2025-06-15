# IPCC RAG Chatbot

This is a Streamlit app that uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions from the IPCC AR6 WGIII report using a FAISS vector index and Databricks-hosted LLMs and embeddings.

## Setup

1. Clone this repository
2. Add your Databricks credentials in Streamlit Cloud under **Secrets**:

```
DATABRICKS_HOST = "https://<your-workspace>.databricks.com"
DATABRICKS_TOKEN = "dapi-xxxxxxxxxxxxxxxx"
```

3. Deploy to [Streamlit Cloud](https://streamlit.io/cloud)

## Files

- `app.py`: Main Streamlit interface
- `rag_core.py`: Core logic for vector search and answer generation
- `requirements.txt`: Python dependencies
- `Notebooks` folder: Databricks notebooks used to create embeddings vectors and development (must be run on a Databricks cluster having required libraries installed)
