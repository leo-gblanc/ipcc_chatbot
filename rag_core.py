import pickle
import faiss
import numpy as np
from databricks_langchain import DatabricksEmbeddings, ChatDatabricks
from langchain.schema import SystemMessage, HumanMessage
import urllib.request
import os
from sklearn.preprocessing import normalize

# === Configuration ===

# Direct links to Google Drive
FAISS_URL = "https://drive.google.com/uc?export=download&id=1xWcHgAKqUdHug5Eqec0MEE2MfyTVBoId"
META_URL  = "https://drive.google.com/uc?export=download&id=1DcG89F5hRGRs0Oe6YO4kB83Lq4rOsF3D"

FAISS_INDEX_PATH = "/tmp/ipcc_faiss.index"
METADATA_PATH    = "/tmp/ipcc_faiss_metadata.pkl"

# Download once if not already here
if not os.path.exists(FAISS_INDEX_PATH):
    urllib.request.urlretrieve(FAISS_URL, FAISS_INDEX_PATH)
    print("✅ FAISS index downloaded.")

if not os.path.exists(METADATA_PATH):
    urllib.request.urlretrieve(META_URL, METADATA_PATH)
    print("✅ Metadata downloaded.")

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
    max_tokens=2048,
    temperature=0.1
)

# === Search with adjacent-chunk retrieval ===
def faiss_similarity_search_with_neighbors(query: str, index, chunk_ids, chunk_id_to_info, k: int = 5, window: int = 1):
    query_vector = embedder.embed_query(query)
    query_vector = np.array([query_vector], dtype="float32")
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, k)

    selected_indices = set()
    index_to_distance = {}

    for rank, anchor_idx in enumerate(indices[0]):
        if 0 <= anchor_idx < len(chunk_ids):
            dist = distances[0][rank]
            for offset in range(-window, window + 1):
                neighbor_idx = anchor_idx + offset
                if 0 <= neighbor_idx < len(chunk_ids):
                    selected_indices.add(neighbor_idx)
                    if neighbor_idx not in index_to_distance or dist < index_to_distance[neighbor_idx]:
                        index_to_distance[neighbor_idx] = dist

    sorted_indices = sorted(selected_indices)

    results = []
    for rank, idx in enumerate(sorted_indices):
        chunk_id = chunk_ids[idx]
        info = chunk_id_to_info[chunk_id]
        results.append({
            "rank": rank,
            "distance": index_to_distance.get(idx),
            "chunk_id": chunk_id,
            "text": info["text"],
            "metadata": info.get("metadata", {})
        })

    return results

# === LLM Answer ===
def generate_answer(question: str, index, chunk_ids, chunk_id_to_info, k: int = 5, window: int = 1):
    hits = faiss_similarity_search_with_neighbors(question, index, chunk_ids, chunk_id_to_info, k=k, window=window)
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
- Provide clear and fact-based explanations.
- Focus especially on economic, regulatory, and resource-related aspects if relevant.
- Cite specific data points, scenarios, or mechanisms if available in the context.
- If the context does not contain a direct answer, say \"The report does not contain enough information to answer this question precisely.\"

Never assume facts outside the given documents, and do not speculate.

Be factual, structured, and neutral."""),
        HumanMessage(content=prompt),
    ]
    ai_msg = chat_model.invoke(msgs)
    return {"answer": ai_msg.content, "chunks": hits}