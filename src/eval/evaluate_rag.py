# src/eval/evaluate_rag.py
import os
import faiss
import numpy as np
import pandas as pd
from datasets import Dataset
from ragas.metrics import faithfulness
from ragas import evaluate
from src.common.paths import KG_DIR, OUT_DIR

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
    D, I = index.search(embeddings, k)
    correct = sum(i in I[i] for i in range(len(I)))
    return correct / len(I)


def compute_ragas_faithfulness(meta_df):
    """Evaluate generation quality using RAGAS Faithfulness metric."""
    # Load your QA pairs
    eval_data = {
        "question": ["Explain the Mississippi Bridge Collapse 2007 and its impact"],
        "answer": [open(os.path.join(OUT_DIR, "answer.txt")).read()],
        "contexts": [meta_df["summary"].tolist()[:5]],
    }
    dataset = Dataset.from_dict(eval_data)
    score = evaluate(dataset, metrics=[faithfulness])["faithfulness"]
    return score


if __name__ == "__main__":
    index, meta = load_index_and_meta(GRAPH_DIR)
    embs = np.array(index.reconstruct_n(0, index.ntotal))
    rec_at_5 = recall_at_k(index, embs, k=5)
    ragas_faith = compute_ragas_faithfulness(meta)

    results = pd.DataFrame([{
        "Recall@5": rec_at_5,
        "RAGAS_Faithfulness": ragas_faith,
    }])
    os.makedirs(OUT_DIR, exist_ok=True)
    results.to_csv(RESULT_PATH, index=False)
    print("✅ Evaluation Results:")
    print(results)