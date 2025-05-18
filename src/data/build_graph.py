import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from collections import defaultdict
import community.community_louvain as community_louvain

from .community_summarization import summarize_communities



def build_knowledge_graph(summary_path="summarized_IITB.csv",
                          embedding_path="summarized_embeddings.npy",
                          graph_path="summary_graph.graphml",
                          max_rows=None,
                          top_k=5):


    print(f"Loading summarized data from: {summary_path}")
    df = pd.read_csv(summary_path, nrows=max_rows)
    print(f"Loading embeddings from: {embedding_path}")
    embeddings = np.load(embedding_path)

    if max_rows is not None:
        embeddings = embeddings[:len(df)]

    embeddings = normalize(embeddings, norm='l2')

    #sanity check - if embedding and summary idx matches
    assert len(df) == len(embeddings), "Mismatch between number of summaries and embeddings!"

    # Initialize graph and add nodes
    G = nx.Graph()
    #Storing idx and summary as nodes and not embeddings as GraphML file doesn't support high-dimensional arrays well
    #also we can access embedding using the idx of the node. The i-th row in the summary CSV → is related to the i-th embedding in the .npy array.

    valid_node_map = {}
    for idx, row in df.iterrows():
        # Clean summary text further
        summary_text = str(row["summary"]).strip()
        summary_text = summary_text.replace("\n", " ")         
        summary_text = summary_text.replace(",", " ")          
        summary_text = summary_text.replace(".", " ")          
        summary_text = " ".join(summary_text.split())           # remove extra spaces
        if not summary_text:
            continue
        G.add_node(idx, text=summary_text)
        valid_node_map[len(valid_node_map)] = idx

    similarity_matrix = cosine_similarity(embeddings)
    print("\nSample Similarity Scores:")
    num_nodes = len(embeddings)
    for i in range(min(num_nodes, 5)):
        for j in range(i + 1, min(num_nodes, 5)):
            print(f"Similarity between {i} and {j}: {similarity_matrix[i][j]:.4f}")


    print("\nAdding edges based on top-k similarity...")
    for i in range(num_nodes):
        sims = similarity_matrix[i]
        top_indices = np.argsort(sims)[-top_k-1:-1][::-1]  # Take top_k excluding self
        for j in top_indices:
            if i != j:
                G.add_edge(i, j, weight=sims[j])


    print(f"🔗 Total edges added: {G.number_of_edges()}")


    #Louvain's community detection algorithm
    print("\nRunning Louvain Community Detection...")
    partition = community_louvain.best_partition(G, weight='weight')
    for node, community_id in partition.items():
        G.nodes[int(node)]["community"] = community_id
            


    # Save graph
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    nx.write_graphml(G, graph_path)
    print(f"Knowledge Graph saved to: {graph_path}")

    # community_labels_path = os.path.join(os.path.dirname(graph_path), "community_labels.csv")
    community_labels_path = os.path.join(os.path.dirname(graph_path), "community_labels.csv")
    os.makedirs(os.path.dirname(community_labels_path), exist_ok=True)
    # print("PATH: ", os.path.dirname(community_labels_path)) 
    community_map = {node: G.nodes[node]['community'] for node in G.nodes()}
    pd.DataFrame.from_dict(community_map, orient='index', columns=["community"]).to_csv(community_labels_path)
    print(f"Community labels saved to: {community_labels_path}")

    print(f"\nFinal Stats — Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    community_groups = defaultdict(list)
    for node, community_id in community_map.items():
        community_groups[community_id].append(G.nodes[node]["text"])




if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
    # PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # SUMMARY_PATH = os.path.join(PROJECT_ROOT, "data/processed/summarized_IITB.csv")
    # EMBEDDING_PATH = os.path.join(PROJECT_ROOT, "data/processed/summarized_embeddings.npy")
    # GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/knowledge_graph/summary_graph.graphml")

    SUMMARY_PATH = os.path.join(PROJECT_ROOT, "processed/tokenized_summarized.csv")
    EMBEDDING_PATH = os.path.join(PROJECT_ROOT, "processed/summarized_embeddings.npy")
    GRAPH_PATH = os.path.join(PROJECT_ROOT, "knowledge_graph/summary_graph.graphml")

    build_knowledge_graph(summary_path=SUMMARY_PATH, embedding_path=EMBEDDING_PATH, graph_path=GRAPH_PATH, max_rows=100, top_k=5)
    