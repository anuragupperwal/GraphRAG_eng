import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Load FAISS and meta
def load_index_and_meta(dir_path):
    index = faiss.read_index(os.path.join(dir_path, "community_faiss.index"))
    meta = pd.read_pickle(os.path.join(dir_path, "community_meta.pkl"))
    return index, meta

def embed_texts(texts, model_name="nomic-ai/nomic-embed-text-v1.5"):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

def retrieve_contexts(queries, index, meta, top_k=5):
    model_name = "nomic-ai/nomic-embed-text-v1.5"
    model = SentenceTransformer(model_name, trust_remote_code=True)
    q_embs = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_embs, top_k)
    contexts = []
    for row in idxs:
        contexts.append([meta.iloc[i]["summary"] for i in row])
    return contexts


def compute_retrieval_metrics(test_df, retrieved_contexts, top_k=5):
    from sklearn.metrics import precision_score, recall_score

    gold_contexts = test_df["gold_context"].tolist()
    recall_scores, precision_scores = [], []

    for i, retrieved in enumerate(retrieved_contexts):
        gold = gold_contexts[i].lower()
        hits = sum(1 for ctx in retrieved if gold[:50] in ctx.lower())  # overlap heuristic
        recall = hits / 1.0  # one gold per query
        precision = hits / top_k
        recall_scores.append(recall)
        precision_scores.append(precision)

    print(f"🔹 Recall@{top_k}: {np.mean(recall_scores):.3f}")
    print(f"🔹 Precision@{top_k}: {np.mean(precision_scores):.3f}")
    return np.mean(recall_scores), np.mean(precision_scores)


def compute_generation_metrics(test_df):
    dataset = Dataset.from_dict({
        "question": test_df["question"].tolist(),
        "contexts": test_df["contexts"].tolist(),
        "answer": test_df["answer"].tolist(),
        "ground_truth": test_df["gold_answer"].tolist(),
    })

    results = evaluate(dataset, metrics=[
        faithfulness, answer_relevancy, context_precision, context_recall
    ])
    print("\n🧠 Generation Evaluation Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.3f}")
    return results


if __name__ == "__main__":
    GRAPH_DIR = "data/kg"
    TEST_FILE = "data/eval/test_queries.csv"
    OUTPUTS_FILE = "data/out/answer.txt"

    index, meta = load_index_and_meta(GRAPH_DIR)
    test_df = pd.read_csv(TEST_FILE)

    # 1️⃣ Retrieve contexts
    contexts = retrieve_contexts(test_df["question"].tolist(), index, meta, top_k=5)
    test_df["contexts"] = contexts

    # 2️⃣ Compute retrieval metrics
    recall, precision = compute_retrieval_metrics(test_df, contexts, top_k=5)

    # 3️⃣ Read generated answers
    if os.path.exists(OUTPUTS_FILE):
        with open(OUTPUTS_FILE, "r", encoding="utf-8") as f:
            gen_ans = [f.read()]
        test_df["answer"] = gen_ans * len(test_df)
    else:
        test_df["answer"] = [""] * len(test_df)

    # 4️⃣ Compute generation metrics
    gen_results = compute_generation_metrics(test_df)