import pickle
import faiss
import numpy as np
from databricks_langchain import DatabricksEmbeddings, ChatDatabricks
from langchain.schema import SystemMessage, HumanMessage
import urllib.request
import os
from sklearn.preprocessing import normalize
import time  # ← We still need timing

# ------------------------------------------------------------------------------
# === EARLY IMPORTS & PATCH FOR TORCH.CLASSES ===
# ------------------------------------------------------------------------------

# Import torch and the Hugging Face reranker at the top so we can patch the problematic attribute.
# This ensures Streamlit’s watcher never sees an un‐patched torch.classes.__path__.
try:
    import torch
    # Attempt to access torch.classes.__path__. This often raises the “__path__._path” error.
    # We catch it and then forcibly override __path__ to an empty list.
    try:
        _ = torch.classes.__path__
    except Exception:
        pass
    # Override to prevent further attempts by any watcher to inspect it.
    torch.classes.__path__ = []
except ImportError:
    # In case torch isn't installed, we still want the rest of the file to load.
    torch = None

# Now import transformers (we can do this unconditionally if torch is available):
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # Instantiate the reranker model & tokenizer at module load time.
    rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    rerank_model     = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
except Exception:
    # If anything goes wrong (e.g. torch not installed), set them to None.
    rerank_tokenizer = None
    rerank_model     = None

# ------------------------------------------------------------------------------
# === CONFIGURATION & FAISS LOADING ===
# ------------------------------------------------------------------------------

# Direct links to Google Drive
FAISS_URL = "https://drive.google.com/uc?export=download&id=1xWcHgAKqUdHug5Eqec0MEE2MfyTVBoId"
META_URL  = "https://drive.google.com/uc?export=download&id=1DcG89F5hRGRs0Oe6YO4kB83Lq4rOsF3D"

FAISS_INDEX_PATH = "/tmp/ipcc_faiss.index"
METADATA_PATH    = "/tmp/ipcc_faiss_metadata.pkl"

# Download once if not already present
if not os.path.exists(FAISS_INDEX_PATH):
    urllib.request.urlretrieve(FAISS_URL, FAISS_INDEX_PATH)
    print("✅ FAISS index downloaded.")

if not os.path.exists(METADATA_PATH):
    urllib.request.urlretrieve(META_URL, METADATA_PATH)
    print("✅ Metadata downloaded.")

# Load FAISS index and metadata
def load_faiss_resources():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta["chunk_ids"], meta["chunk_id_to_info"]

# ------------------------------------------------------------------------------
# === EMBEDDING & CHAT MODEL INITIALIZATION ===
# ------------------------------------------------------------------------------

embedder   = DatabricksEmbeddings(endpoint="databricks-gte-large-en", batch_size=1)
chat_model = ChatDatabricks(endpoint="databricks-claude-3-7-sonnet", max_tokens=2048, temperature=0.1)

# ------------------------------------------------------------------------------
# === CONTEXTUALIZATION & PARAPHRASE FUNCTIONS ===
# ------------------------------------------------------------------------------

def contextualize_query_with_background(query: str) -> str:
    prompt = """
    You are a domain expert in climate change mitigation (IPCC WGIII).
    Given a user question, briefly explain any relevant background knowledge:
    - What the mentioned industry relies on (inputs, feedstocks, energy)
    - Associated sectors or emissions sources
    - Relevant technical terms used in IPCC reports
    Return 2-3 sentences.
    """
    response = chat_model.invoke([SystemMessage(content=prompt), HumanMessage(content=query)])
    return response.content.strip()

def generate_paraphrases(query: str) -> list:
    prompt = """
    Rewrite the user's question in 2 semantically equivalent but differently phrased versions,
    using terminology common in IPCC WGIII reports.
    Return one paraphrase per line.
    """
    response = chat_model.invoke([SystemMessage(content=prompt), HumanMessage(content=query)])
    return [line.strip() for line in response.content.strip().split("\n") if line.strip()]

# ------------------------------------------------------------------------------
# === FAISS RETRIEVAL ===
# ------------------------------------------------------------------------------

def faiss_similarity_search_groups(query: str, index, chunk_ids, chunk_id_to_info, k: int = 5, window: int = 1):
    query_vector = embedder.embed_query(query)
    query_vector = np.array([query_vector], dtype="float32")
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, k)

    chunk_groups = []
    for anchor_idx, dist in zip(indices[0], distances[0]):
        if anchor_idx < 0 or anchor_idx >= len(chunk_ids):
            continue
        group = []
        for offset in range(-window, window + 1):
            neighbor_idx = anchor_idx + offset
            if 0 <= neighbor_idx < len(chunk_ids):
                chunk_id = chunk_ids[neighbor_idx]
                info = chunk_id_to_info[chunk_id]
                group.append({
                    "chunk_id": chunk_id,
                    "text": info["text"],
                    "metadata": info.get("metadata", {})
                })
        if group:
            chunk_groups.append(group)
    return chunk_groups

# ------------------------------------------------------------------------------
# === RERANK FUNCTION (now uses top‐level rerank_model & rerank_tokenizer) ===
# ------------------------------------------------------------------------------

def rerank_chunk_groups(query, chunk_groups, top_n=5):
    """
    Uses the pre‐loaded `rerank_model` and `rerank_tokenizer` from the module‐load phase.
    Since `torch.classes.__path__` was patched above, the watcher will not error out.
    """
    if rerank_model is None or rerank_tokenizer is None:
        # If for some reason loading failed, just return the unranked groups:
        return chunk_groups

    # Build inputs for reranking
    inputs = [
        f"{query} [SEP] " + " ".join(chunk["text"] for chunk in group)
        for group in chunk_groups
    ]
    tokens = rerank_tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")

    # Compute scores
    with torch.no_grad():
        scores = rerank_model(**tokens).logits.squeeze(-1)

    # Select top_n groups by score
    top_indices = torch.topk(scores, k=min(top_n, len(chunk_groups))).indices.tolist()
    top_groups  = [chunk_groups[i] for i in top_indices]

    # Attach reranker_score to each chunk
    for i, idx in enumerate(top_indices):
        for chunk in top_groups[i]:
            chunk["reranker_score"] = round(scores[idx].item(), 4)

    return top_groups

# ------------------------------------------------------------------------------
# === GENERATE_ANSWER WITH TIMING ===
# ------------------------------------------------------------------------------

def generate_answer(query: str,
                    index,
                    chunk_ids,
                    chunk_id_to_info,
                    k: int = 5,
                    window: int = 1,
                    rerank_top_n: int = 6,
                    chat_history: list = []):
    timings = {}

    # 1) Contextualization
    t0 = time.time()
    context = contextualize_query_with_background(query)
    timings["contextualization"] = time.time() - t0

    # 2) Paraphrase generation
    t1 = time.time()
    paraphrases = generate_paraphrases(query)
    timings["paraphrase_generation"] = time.time() - t1

    # Build enriched queries (original + paraphrases)
    queries = [query] + paraphrases
    enriched_queries = [f"{q}\n\n{context}" for q in queries]

    # 3) FAISS retrieval + reranking
    t2 = time.time()
    all_groups = []
    seen_ids = set()
    for q in enriched_queries:
        groups = faiss_similarity_search_groups(q, index, chunk_ids, chunk_id_to_info, k=k, window=window)
        for group in groups:
            ids = tuple(chunk["chunk_id"] for chunk in group)
            if ids not in seen_ids:
                seen_ids.add(ids)
                all_groups.append(group)

    top_groups = rerank_chunk_groups(query, all_groups, top_n=rerank_top_n)
    timings["retrieval_and_rerank"] = time.time() - t2

    selected_chunks = [chunk for group in top_groups for chunk in group]

    # Tag chunks with inline references (1), (2), ...
    context_text = ""
    for i, chunk in enumerate(selected_chunks):
        ref = f"({i+1})"
        chunk["reference"] = ref
        context_text += f"{ref} {chunk['text']}\n\n"

    # Include memory: up to 1 previous Q&A
    memory_msgs = []
    for pair in chat_history[-1:]:
        memory_msgs.extend([
            HumanMessage(content=pair["user"]),
            SystemMessage(content=pair["assistant"])
        ])

    prompt = (
        "Use the numbered references in the context to cite where each fact comes from. "
        "Ignore any citations like [914] or [6813] that may appear inside the text.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Provide a detailed answer using inline references like (1), (2), etc. to indicate sources."
    )

    # 4) Final LLM generation
    t3 = time.time()
    msgs = memory_msgs + [
        SystemMessage(content="""
        You are an expert assistant specialized in climate change impacts, with a focus on economic consequences 
        such as demand shifts, productivity variations, resource dependencies, and regulatory dynamics.

        Use only the provided context from official IPCC reports and related documents to answer the user's question. 
        Do not invent information or cite external knowledge.

        When answering:
        - Provide clear and fact-based explanations.
        - Focus especially on economic, regulatory, and resource-related aspects if relevant.
        - Cite specific data points, scenarios, or mechanisms if available in the context.
        - If the context does not contain a direct answer, briefly mention that a precise answer may not be available.

        Never assume facts outside the given documents, and do not speculate. Be factual, structured, and neutral.
        """),
        HumanMessage(content=prompt)
    ]
    ai_msg = chat_model.invoke(msgs)
    timings["generation"] = time.time() - t3

    return {
        "answer": ai_msg.content,
        "chunks": selected_chunks,
        "timings": timings
    }
