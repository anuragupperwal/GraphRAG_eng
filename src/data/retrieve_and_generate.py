import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

# --- Load Gemini API key ---
load_dotenv()
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")
except Exception:
    pass
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Long-context embedding model ---
_EMB_MODEL = "nomic-ai/nomic-embed-text-v1.5"
_emb_model = SentenceTransformer(_EMB_MODEL, trust_remote_code=True)


def _load_faiss_and_meta(dir_path):
    index = faiss.read_index(os.path.join(dir_path, "community_faiss.index"))
    meta = pd.read_pickle(os.path.join(dir_path, "community_meta.pkl"))
    return index, meta


def _embed(texts):
    return _emb_model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")


def _retrieve(index, meta_df, query, top_k=5):
    """Search FAISS index for top-k similar community summaries."""
    q = _embed([query])
    scores, idxs = index.search(q, top_k)
    out = meta_df.iloc[idxs[0]].copy()
    out["score"] = scores[0]
    return out


def _gen_answer(context, query, model_name="gemini-2.5-flash"):
    """Generate long, detailed answer using Gemini 2.5 Flash."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[Warn] GOOGLE_API_KEY not found — using local model fallback.")
        return _gen_answer_local(context, query)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = (
        "You are an expert assistant working within a retrieval-augmented system.\n"
        "Use *only* the provided context to answer the question.\n"
        "Write a long, well-structured answer that integrates all relevant facts.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini error: {e}] Falling back to local model.")
        return _gen_answer_local(context, query)


def _gen_answer_local(context, query, model_name="google/flan-t5-base", max_in=2048, max_out=256):
    """Fallback generator using FLAN-T5 (if Gemini unavailable)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    prompt = (
        "Answer the question based *only* on the given context.\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    inputs = tok([prompt], return_tensors="pt", truncation=True, max_length=max_in, padding=True).to(mod.device)
    with torch.no_grad():
        out = mod.generate(**inputs, max_length=max_out, num_beams=3)
    return tok.batch_decode(out, skip_special_tokens=True)[0]


def generate_output(top_k, query, model_name, project_root, summary_graph_path, embedding_path, output_path):
    """Retrieve top-k summaries from FAISS and generate final answer."""
    index, meta = _load_faiss_and_meta(summary_graph_path)
    hits = _retrieve(index, meta, query, top_k=top_k)

    if hits.empty:
        print("⚠️ No retrieved summaries found.")
        return

    context = "\n\n".join(hits["summary"].tolist())
    ans = _gen_answer(context, query, model_name=model_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ans)

    print("\n✅ ANSWER GENERATED SUCCESSFULLY:\n")
    print(ans)