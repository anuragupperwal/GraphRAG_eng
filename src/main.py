# src/main.py
import os
import argparse
import networkx as nx
from dotenv import load_dotenv

from src.common.paths import PROJECT_ROOT, RAW_DIR, PROC_DIR, KG_DIR, OUT_DIR
from src.data.preprocess_data_main import preprocess_english_corpus
from src.data.summarize_tokenized import summarize_corpus
from src.data.create_embeddings import generate_embeddings
from src.data.build_graph import build_knowledge_graph
from src.data.community_summarization import summarize_communities
from src.data.retrieve_and_generate import generate_output

# Load environment variables (for GOOGLE_API_KEY)
load_dotenv()


def ensure_dirs():
    """Make sure all data directories exist."""
    for d in [RAW_DIR, PROC_DIR, KG_DIR, OUT_DIR]:
        os.makedirs(d, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the simplified GraphRAG pipeline (Kaggle friendly)")
    parser.add_argument("--raw_csv", type=str, default=os.path.join(RAW_DIR, "cnn_dailymail.csv"),
                        help="Path to input CSV dataset")
    parser.add_argument("--text_col", type=str, default="article",
                        help="Column name containing the raw text")
    parser.add_argument("--nrows", type=int, default=3000,
                        help="Number of rows to process")
    parser.add_argument("--chunk_size", type=int, default=3,
                        help="Batch size for summarization (not used now, kept for compatibility)")
    parser.add_argument("--top_k_graph", type=int, default=7,
                        help="Number of nearest neighbors per node in the graph")
    parser.add_argument("--top_k_ret", type=int, default=5,
                        help="Number of retrieved community summaries for final QA")
    parser.add_argument("--query", type=str, default="Test query",
                        help="Question to ask the RAG system")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash",
                        help="Model used for final answer generation")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    # File paths
    TOKENIZED = os.path.join(PROC_DIR, "tokenized.csv")
    SUMMARIZED = os.path.join(PROC_DIR, "tokenized_summarized.csv")
    EMBEDS = os.path.join(PROC_DIR, "summarized_embeddings.npy")
    GRAPH = os.path.join(KG_DIR, "summary_graph.graphml")
    SUMMARY_GRAPH_DIR = KG_DIR

    # 1️-Preprocess raw corpus
    preprocess_english_corpus(
        raw_csv_path=args.raw_csv,
        nrows=args.nrows,
        text_col=args.text_col,
        final_output=TOKENIZED
    )

    # # 2️-Long-context summarization via Gemini
    # summarize_corpus(
    #     input_path=TOKENIZED,
    #     output_path=SUMMARIZED,
    #     batch_size=args.chunk_size
    # )

    print("Using cleaned documents directly — skipping per-document summaries.")
    source_csv = TOKENIZED
    content_col = "text"

    # 3️-Generate long-context embeddings
    generate_embeddings(
        INPUT_PATH=TOKENIZED,
        OUTPUT_PATH=EMBEDS,
        text_col=content_col
    )

    # 4️-Build graph (semantic k-NN + Louvain)
    build_knowledge_graph(
        summary_path=SUMMARIZED,
        embedding_path=EMBEDS,
        graph_path=GRAPH,
        max_rows=args.nrows,
        top_k=args.top_k_graph
    )

    # 5️-Community summarization (Gemini + Nomic)
    G = nx.read_graphml(GRAPH)
    summarize_communities(G, output_path_directory=SUMMARY_GRAPH_DIR)

    # 6-Retrieval + final answer generation
    generate_output(
        top_k=args.top_k_ret,
        query=args.query,
        model_name=args.model_name,
        project_root=PROJECT_ROOT,
        summary_graph_path=SUMMARY_GRAPH_DIR,
        embedding_path=EMBEDS,
        output_path=os.path.join(OUT_DIR, "answer.txt")
    )


if __name__ == "__main__":
    main()




# !python src/main.py \
#   --raw_csv data/raw/articles.csv \
#   --text_col article \
#   --nrows 1000 \
#   --chunk_size 3 \
#   --top_k_graph 7 \
#   --top_k_ret 5 \
#   --query "Explain the Mississippi Bridge Collapse 2007 and its impact" \
#   --model_name "gemini-2.5-flash"