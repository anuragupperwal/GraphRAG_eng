import os
import networkx as nx

from data.create_embeddings import generate_embeddings, test_embeddings
from data.preprocess_data_main import preprocess_english_corpus
from data.summarize_tokenized_bart import summarize_corpus
from data.build_graph import build_knowledge_graph
from data.community_summarization import summarize_communities
from data.retrieve_and_generate import generate_output


# RAW_DATA_PATH = "../../data/raw/monolingual-n/raw_IITB.csv"
# PROCESSED_OUTPUT_PATH = "data/processed_corpus.csv"


# Get project root dynamically
# PROJECT_ROOT = "/kaggle/working/"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data/raw/cnn_dailymail.csv")
# Define absolute paths
# RAW_DATA_PATH = "/kaggle/input/hindi-corpus/bookcorpus_english_sample.csv"
TOKENIZED_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenized.csv")
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenised_summarized.csv")
SUMMARY_CLEANED_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenised_summarized_cleaned.csv")
EMBEDDING_PATH = os.path.join(PROJECT_ROOT, "data/processed/summarized_embeddings.npy")
GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/knowledge_graph/summary_graph.graphml")
SUMMARY_GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/knowledge_graph/")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/output/answer.txt")


max_line_bound = 5
print("Running preprocessing")
preprocess_english_corpus(RAW_DATA_PATH, max_lines=max_line_bound, project_root=PROJECT_ROOT)

print("Running summarization")
summarize_corpus(
    input_path=TOKENIZED_PATH,
    output_path=SUMMARY_PATH,
    cleaned_path=SUMMARY_CLEANED_PATH,
    chunk_size=1,
    max_lines=max_line_bound
)
print("Summarization Completed")


generate_embeddings(SUMMARY_CLEANED_PATH, EMBEDDING_PATH, max_line_bound)
# test_embeddings(EMBEDDING_PATH)

print("Building Graph")
build_knowledge_graph(
    summary_path=SUMMARY_CLEANED_PATH,
    embedding_path=EMBEDDING_PATH,
    graph_path=GRAPH_PATH,
    top_k=5
)

print("Summarizing Communities")
G = nx.read_graphml(GRAPH_PATH)
summarize_communities(G, output_path_directory=SUMMARY_GRAPH_PATH)

# retrieve based on query
query1 = "Tell me about Michael Jackson concert updates and ticket changes."  

# Generated:  

# query = input("Enter your query: ")
top_k = 5
print("in dataset: ")
generate_output(top_k, query1, "BART", PROJECT_ROOT, SUMMARY_GRAPH_PATH, EMBEDDING_PATH, OUTPUT_PATH)
print("in dataset 2 : ")
# generate_output(top_k, query2, "mT5", PROJECT_ROOT, SUMMARY_GRAPH_PATH, EMBEDDING_PATH, OUTPUT_PATH)
print("Not in dataset: ")
# generate_output(top_k, query3, "mT5", PROJECT_ROOT, SUMMARY_GRAPH_PATH, EMBEDDING_PATH, OUTPUT_PATH)
