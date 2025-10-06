# src/eval/evaluate_rag.py
import os
import sys
import re
import faiss
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---- Fix module path for Kaggle + local runs ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.common.paths import KG_DIR, OUT_DIR


# ---------- Paths ----------
GRAPH_DIR = KG_DIR
RESULT_PATH = os.path.join(OUT_DIR, "evaluation_report.csv")

# ---------- Embedding model ----------
EMB_MODEL = "nomic-ai/nomic-embed-text-v1.5"


# ============================================================
# Utility Functions
# ============================================================

def load_index_and_meta(dir_path):
    """Load FAISS index and metadata file."""
    faiss_path = os.path.join(dir_path, "community_faiss.index")
    meta_path = os.path.join(dir_path, "community_meta.pkl")
    assert os.path.exists(faiss_path), f"Missing FAISS index at {faiss_path}"
    assert os.path.exists(meta_path), f"Missing metadata file at {meta_path}"

    index = faiss.read_index(faiss_path)
    meta = pd.read_pickle(meta_path)
    return index, meta


def recall_at_k(index, embeddings, k=5):
    """Self-retrieval Recall@K (each item should retrieve itself)."""
    if k == 0:
        return 0.0
    D, I = index.search(embeddings, k)
    correct = sum(i in I[i] for i in range(len(I)))
    return round(correct / len(I), 4)


def precision_at_k(index, embeddings, k=5, threshold=0.7):
    """Approximate Precision@K using cosine similarity."""
    if k == 0:
        return 0.0
    D, I = index.search(embeddings, k)
    precision_scores = []
    for sims in D:
        relevant = np.sum(sims > threshold)
        precision_scores.append(relevant / k)
    return round(float(np.mean(precision_scores)), 4)


def semantic_faithfulness(answer, contexts, model):
    """Embedding-based faithfulness proxy."""
    if not answer.strip():
        return 0.0
    ans_emb = model.encode([answer], normalize_embeddings=True)
    ctx_embs = model.encode(contexts, normalize_embeddings=True)
    sims = cosine_similarity(ans_emb, ctx_embs)
    return round(float(np.mean(sims)), 4)


def context_coverage(answer, contexts):
    """Word-overlap based context grounding metric."""
    answer_words = set(re.findall(r"\w+", answer.lower()))
    context_words = set(re.findall(r"\w+", " ".join(contexts).lower()))
    overlap = len(answer_words & context_words)
    return round(overlap / max(1, len(answer_words)), 4)


# ============================================================
# Main Evaluation
# ============================================================

if __name__ == "__main__":
    print("[Info] Loading FAISS index and metadata...")
    index, meta = load_index_and_meta(GRAPH_DIR)

    if index.ntotal == 0:
        raise ValueError("FAISS index is empty — cannot evaluate retrieval metrics.")

    print("[Info] Loading embedding model for semantic similarity...")
    model = SentenceTransformer(EMB_MODEL, trust_remote_code=True)

    print("[Info] Reconstructing embeddings from FAISS index...")
    try:
        embs = np.vstack([index.reconstruct(i) for i in range(index.ntotal)])
    except Exception as e:
        print(f"[Warning] Could not reconstruct embeddings: {e}")
        print("[Info] Falling back to recomputing embeddings from summaries...")
        embs = model.encode(meta["summary"].tolist(), normalize_embeddings=True).astype("float32")

    ans_path = os.path.join(OUT_DIR, "answer.txt")
    assert os.path.exists(ans_path), f"Missing generated answer at {ans_path}"
    with open(ans_path, "r", encoding="utf-8") as f:
        answer = f.read().strip()

    contexts = meta["summary"].tolist()[:5]

    print("[Info] Computing metrics...")
    rec_at_5 = recall_at_k(index, embs, k=5)
    prec_at_5 = precision_at_k(index, embs, k=5)
    faithfulness_score = semantic_faithfulness(answer, contexts, model)
    coverage_score = context_coverage(answer, contexts)
    hallucination_score = round(max(0.0, min(1.0, 1 - faithfulness_score)), 4)

    results = pd.DataFrame([{
        "Recall@5": rec_at_5,
        "Precision@5": prec_at_5,
        "Faithfulness": faithfulness_score,
        "Context_Coverage": coverage_score,
        "Hallucination": hallucination_score,
    }])

    os.makedirs(OUT_DIR, exist_ok=True)
    results.to_csv(RESULT_PATH, index=False)

    print("\n✅ Evaluation Complete — Results:")
    print(results)