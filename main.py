import os
import networkx as nx
import argparse

from data.create_embeddings import generate_embeddings, test_embeddings
from data.preprocess_data_main import preprocess_english_corpus
from data.summarize_tokenized_bart import summarize_corpus
from data.build_graph import build_knowledge_graph
from data.community_summarization import summarize_communities
from data.retrieve_and_generate import generate_output


# RAW_DATA_PATH = "../../data/raw/monolingual-n/raw_IITB.csv"
# PROCESSED_OUTPUT_PATH = "data/processed_corpus.csv"


# Get project root dynamically
PROJECT_ROOT = "/kaggle/working/"
RAW_DATA_PATH = "/kaggle/input/cnndailymail-dataset/cnn_dailymail.csv"
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/raw/cnn_dailymail.csv")
# Define absolute paths
TOKENIZED_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenized.csv")
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenised_summarized.csv")
# SUMMARY_CLEANED_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenised_summarized_cleaned.csv")
EMBEDDING_PATH = os.path.join(PROJECT_ROOT, "data/processed/summarized_embeddings.npy")
GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/knowledge_graph/summary_graph.graphml")
SUMMARY_GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/knowledge_graph/")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/output/answer.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_line_bound", type=int, default=5, help="Maximum number of lines per summary chunk")
    parser.add_argument("--chunk_size", type=int, default=1, help="No. of sentences to chunk for summary")
    parser.add_argument("--top_k_graph", type=int, default=5, help="Top K results for graph-based retrieval")
    parser.add_argument("--top_k_ret", type=int, default=5, help="Top K results to retrieve")
    parser.add_argument("--query", type=str, default="Tell me about Michael Jackson concert updates and ticket changes.", help="Query")
    args = parser.parse_args()

    max_line_bound = args.max_line_bound
    chunk_size = args.chunk_size
    top_k_graph = args.top_k_graph
    top_k_ret = args.top_k_ret
    query = args.query

    print("Running preprocessing")
    preprocess_english_corpus(RAW_DATA_PATH, max_lines=max_line_bound, project_root=PROJECT_ROOT)

    print("Running summarization")
    summarize_corpus(
        input_path=TOKENIZED_PATH,
        output_path=SUMMARY_PATH,
        chunk_size=chunk_size,
        max_lines=max_line_bound
    )
    print("Summarization Completed")


    generate_embeddings(SUMMARY_PATH, EMBEDDING_PATH, max_line_bound)
    # test_embeddings(EMBEDDING_PATH)

    print("Building Graph")
    build_knowledge_graph(
        summary_path=SUMMARY_PATH,
        embedding_path=EMBEDDING_PATH,
        graph_path=GRAPH_PATH,
        top_k=top_k_graph
    )

    print("Summarizing Communities")
    G = nx.read_graphml(GRAPH_PATH)
    summarize_communities(G, output_path_directory=SUMMARY_GRAPH_PATH)

    # query = input("Enter your query: ")
    print("\nusing BART: ")
    generate_output(top_k_ret, query, "BART", PROJECT_ROOT, SUMMARY_GRAPH_PATH, EMBEDDING_PATH, OUTPUT_PATH)
    print("\nusing mT5: ")
    generate_output(top_k_ret, query, "mT5", PROJECT_ROOT, SUMMARY_GRAPH_PATH, EMBEDDING_PATH, OUTPUT_PATH)
    # print("Not in dataset: ")
    # generate_output(top_k, query3, "mT5", PROJECT_ROOT, SUMMARY_GRAPH_PATH, EMBEDDING_PATH, OUTPUT_PATH)
