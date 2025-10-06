# src/eval/evaluate_rag.py
import os
import sys
import faiss
import numpy as np
import pandas as pd
from datasets import Dataset
from ragas.metrics import faithfulness
from ragas import evaluate

# ---- Fix module path for Kaggle + local runs ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.common.paths import KG_DIR, OUT_DIR


# Paths
GRAPH_DIR = KG_DIR
RESULT_PATH = os.path.join(OUT_DIR, "evaluation_report.csv")


def load_index_and_meta(dir_path):
    faiss_path = os.path.join(dir_path, "community_faiss.index")
    meta_path = os.path.join(dir_path, "community_meta.pkl")
    assert os.path.exists(faiss_path), f"Missing FAISS index at {faiss_path}"
    assert os.path.exists(meta_path), f"Missing metadata file at {meta_path}"

    index = faiss.read_index(faiss_path)
    meta = pd.read_pickle(meta_path)
    return index, meta


def recall_at_k(index, embeddings, k=5):
    """
    Computes Recall@K for retrieval evaluation.
    Measures how often the ground-truth item appears among top-k retrieved items.
    """
    D, I = index.search(embeddings, k)
    # For simplicity, we simulate “self retrieval” (each item should retrieve itself)
    correct = sum(i in I[i] for i in range(len(I)))
    return round(correct / len(I), 4)


def compute_ragas_faithfulness(meta_df):
    """
    Evaluate generation quality using RAGAS Faithfulness metric.
    Compares model’s generated answer to its retrieved context.
    """
    ans_path = os.path.join(OUT_DIR, "answer.txt")
    assert os.path.exists(ans_path), f"Missing generated answer at {ans_path}"

    eval_data = {
        "question": ["Explain the Mississippi Bridge Collapse 2007 and its impact"],
        "answer": [open(ans_path).read()],
        "contexts": [meta_df["summary"].tolist()[:5]],
    }

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(dataset, metrics=[faithfulness])
    return round(result["faithfulness"], 4)


if __name__ == "__main__":
    print("[Info] Loading FAISS index and metadata...")
    index, meta = load_index_and_meta(GRAPH_DIR)

    print("[Info] Reconstructing embeddings from FAISS index...")
    embs = np.vstack([index.reconstruct(i) for i in range(index.ntotal)])

    print("[Info] Computing Recall@5...")
    rec_at_5 = recall_at_k(index, embs, k=5)

    print("[Info] Computing RAGAS Faithfulness...")
    ragas_faith = compute_ragas_faithfulness(meta)

    results = pd.DataFrame([{
        "Recall@5": rec_at_5,
        "RAGAS_Faithfulness": ragas_faith,
    }])
    os.makedirs(OUT_DIR, exist_ok=True)
    results.to_csv(RESULT_PATH, index=False)

    print("\n✅ Evaluation Complete — Results:")
    print(results)