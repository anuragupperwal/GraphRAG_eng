import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import community.community_louvain as community_louvain  
from src.common.paths import KG_DIR


def build_knowledge_graph(
    summary_path,
    embedding_path,
    graph_path,
    max_rows=3000,
    top_k=7,
    content_col="text",
    sim_threshold=0.4
):
    """
    Build a weighted knowledge graph from document embeddings.
    
    Args:
        summary_path (str): CSV path containing text content (cleaned/summarized documents)
        embedding_path (str): Path to .npy file with embeddings
        graph_path (str): Output path to save .graphml
        max_rows (int): Maximum rows to load from dataset
        top_k (int): Number of nearest neighbors per node
        content_col (str): Column name containing text
        sim_threshold (float): Minimum cosine similarity to add an edge
    """

    # --- Load data ---
    df = pd.read_csv(summary_path, nrows=max_rows)
    assert content_col in df.columns, f"Missing column '{content_col}' in {summary_path}"
    texts = df[content_col].fillna("").astype(str).tolist()
    embs = np.load(embedding_path)[:len(df)]
    print(f"[Info] Building graph with {len(df)} nodes and top_k={top_k} ...")

    # --- Compute pairwise similarities ---
    sim = cosine_similarity(embs)
    np.fill_diagonal(sim, 0.0)

    # --- Create graph ---
    G = nx.Graph()
    for i, content in enumerate(texts):
        G.add_node(i, content=content)

    # --- Connect top-k neighbors based on cosine similarity ---
    edge_count = 0
    for i in range(len(df)):
        nbr_ids = np.argpartition(-sim[i], top_k)[:top_k]
        for j in nbr_ids:
            w = float(sim[i, j])
            if w >= sim_threshold:
                G.add_edge(i, j, weight=w)
                edge_count += 1

    # --- Community detection using Louvain method ---
    print("[Info] Running Louvain community detection ...")
    part = community_louvain.best_partition(G, weight="weight")
    nx.set_node_attributes(G, part, "community")

    # --- Save graph ---
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    nx.write_graphml(G, graph_path)

    # --- Save community labels ---
    labels = pd.DataFrame({"node": list(part.keys()), "community": list(part.values())})
    labels.to_csv(os.path.join(KG_DIR, "community_labels.csv"), index=False)

    print(f"Graph saved: {graph_path}")
    print(f"   Nodes: {G.number_of_nodes()}  Edges: {edge_count}  Communities: {len(set(part.values()))}")
    print(f"   Threshold: {sim_threshold}  Top-k per node: {top_k}")