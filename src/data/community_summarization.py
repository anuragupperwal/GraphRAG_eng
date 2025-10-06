import os
import faiss
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from src.common.paths import KG_DIR

load_dotenv()
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ["GOOGLE_API_KEY"] = user_secrets.get_secret("GOOGLE_API_KEY")
except Exception:
    pass
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


_EMB_MODEL = "nomic-ai/nomic-embed-text-v1.5"
_emb_model = SentenceTransformer(_EMB_MODEL, trust_remote_code=True)
_gemini_model = genai.GenerativeModel("gemini-2.5-flash")


def _summarize_text_gemini(long_text):
    """Gemini-based long-context community summarizer."""
    prompt = (
        "You are summarizing a community of related documents for a Graph-based RAG system.\n"
        "Write a detailed, information-dense synthesis preserving every distinct fact and event.\n\n"
        f"Documents:\n{long_text}\n\nCommunity Summary:"
    )
    try:
        response = _gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Warning] Gemini error: {e}")
        return long_text[:8000]

def summarize_communities(G: nx.Graph, output_path_directory: str):
    comm2nodes = defaultdict(list)
    for nid, data in G.nodes(data=True):
        comm2nodes[data.get("community", -1)].append(nid)

    rows = []
    for cid, nodes in tqdm(comm2nodes.items(), desc="Community summarization (Gemini)"):
        deg = nx.degree_centrality(G.subgraph(nodes))
        top_nodes = sorted(nodes, key=lambda n: deg.get(n, 0.0), reverse=True)[:40]
        texts = [G.nodes[n].get("content", "") for n in top_nodes]
        joined = " ".join(texts)[:12000]
        summary = _summarize_text_gemini(joined)
        rows.append({"community": cid, "summary": summary})

    com_df = pd.DataFrame(rows).sort_values("community")
    csv_path = os.path.join(output_path_directory, "community_summary.csv")
    com_df.to_csv(csv_path, index=False)
    print(f"Summarized {len(rows)} communities.")

    # Build FAISS index
    embs = _emb_model.encode(com_df["summary"].tolist(),
                             convert_to_numpy=True,
                             normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, os.path.join(output_path_directory, "community_faiss.index"))
    com_df.to_pickle(os.path.join(output_path_directory, "community_meta.pkl"))
    print(f"Saved community summaries + FAISS index to {output_path_directory}")