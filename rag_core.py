import pickle
import faiss
import numpy as np
from databricks_langchain import DatabricksEmbeddings, ChatDatabricks
from langchain.schema import SystemMessage, HumanMessage

# === Configuration ===
FAISS_INDEX_PATH = "/dbfs/FileStore/rag/ipcc_faiss.index"
METADATA_PATH    = "/dbfs/FileStore/rag/ipcc_faiss_metadata.pkl"

# === Load FAISS index and metadata ===
def load_faiss_resources():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        meta = pickle.load(f)
    chunk_ids = meta["chunk_ids"]
    chunk_id_to_info = meta["chunk_id_to_info"]
    return index, chunk_ids, chunk_id_to_info

# === Initialize models ===
embedder = DatabricksEmbeddings(
    endpoint="databricks-gte-large-en",
    batch_size=1
)

chat_model = ChatDatabricks(
    endpoint="databricks-claude-3-7-sonnet",
    max_tokens=1024,
    temperature=0.1
)

# === Search ===
def faiss_similarity_search(query: str, index, chunk_ids, chunk_id_to_info, k: int = 5):
    query_vector = embedder.embed_query(query)
    distances, indices = index.search(np.array([query_vector]).astype("float32"), k)

    results = []
    for rank, idx in enumerate(indices[0]):
        dist = distances[0][rank]
        chunk_id = chunk_ids[idx]
        info = chunk_id_to_info[chunk_id]
        results.append({
            "rank": rank,
            "distance": dist,
            "chunk_id": chunk_id,
            "text": info["text"],
            "metadata": info.get("metadata", {})
        })
    return results

# === LLM Answer ===
def generate_answer(question: str, index, chunk_ids, chunk_id_to_info, k: int = 10):
    hits = faiss_similarity_search(question, index, chunk_ids, chunk_id_to_info, k=k)
    context = "\n\n".join(h["text"] for h in hits)

    prompt = (
        "Use the context below to answer the question precisely.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Provide a detailed answer:"
    )

    msgs = [
        SystemMessage(content="""You are an expert assistant specialized in climate change impacts, with a focus on economic consequences such as demand shifts, productivity variations, resource dependencies, and regulatory dynamics.

Use only the provided context from official IPCC reports and related documents to answer the user's question. Do not invent information or cite external knowledge.

When answering:
- Provide clear, concise, and fact-based explanations.
- Focus especially on economic, regulatory, and resource-related aspects if relevant.
- Cite specific data points, scenarios, or mechanisms if available in the context.
- If the context does not contain a direct answer, say "The provided documents do not contain enough information to answer this question precisely."

Never assume facts outside the given documents, and do not speculate.

Be factual, structured, and neutral."""),
        HumanMessage(content=prompt),
    ]
    ai_msg = chat_model.invoke(msgs)
    return {"answer": ai_msg.content, "chunks": hits}