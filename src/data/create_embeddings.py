import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.common.paths import PROC_DIR

SUMMARY_MODEL = "nomic-ai/nomic-embed-text-v1.5" 

def generate_embeddings(INPUT_PATH, OUTPUT_PATH=None, TEXT_COL="text", nrows=None):
    df = pd.read_csv(INPUT_PATH, nrows=nrows)
    assert text_col in df.columns, f"Column '{text_col}' not found. Available: {list(df.columns)}"
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    print(f"[Info] Encoding {len(texts)} documents using {EMBED_MODEL} ...")
    model = SentenceTransformer(SUMMARY_MODEL, trust_remote_code=True)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    if OUTPUT_PATH is None:
        OUTPUT_PATH = os.path.join(PROC_DIR, "summarized_embeddings.npy")
    np.save(OUTPUT_PATH, embs.astype("float32"))
    print(f"Saved embeddings: {OUTPUT_PATH}, shape={embs.shape}")


