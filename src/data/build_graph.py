import os
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain  # from python-louvain
from src.common.paths import KG_DIR

def build_knowledge_graph(summary_path, embedding_path, graph_path, max_rows=3000, top_k=5):
    df = pd.read_csv(summary_path, nrows=max_rows)
    embs = np.load(embedding_path)[:len(df)]  # guard

    sim = cosine_similarity(embs)  # (N,N)
    np.fill_diagonal(sim, 0.0)

    # Create graph
    G = nx.Graph()
    for i in range(len(df)):
        G.add_node(i, summary=df["summary"].iloc[i])

    # Connect top-k nearest neighbors above threshold
    THRESHOLD = 0.4
    for i in range(len(df)):
        nbr_ids = np.argpartition(-sim[i], top_k)[:top_k]
        for j in nbr_ids:
            w = float(sim[i, j])
            if w >= THRESHOLD:
                G.add_edge(i, j, weight=w)

    # Louvain communities
    part = community_louvain.best_partition(G, weight="weight")
    nx.set_node_attributes(G, part, "community")

    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    nx.write_graphml(G, graph_path)

    # Save a tiny sidecar (labels)
    labels = pd.DataFrame({"node": list(part.keys()), "community": list(part.values())})
    labels.to_csv(os.path.join(KG_DIR, "community_labels.csv"), index=False)