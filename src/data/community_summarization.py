import os
import torch
import re
import pandas as pd
import math
import networkx as nx
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from evaluate import load


#Prevents the fork-related deadlock and slowness from huggingface tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/bart-large-cnn"
# path_to_model = "models/bart_cnn_finetuned/final_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
# model = model.to("cuda" if torch.cuda.is_available() else "cpu")
scoreEstimator = .2
def clean_chunk(text):
    text = re.sub(r"\(.*?\)", "", text)  # Remove things like (BBC), (a.a), etc.
    text = re.sub(r"[^\u0900-\u097F\s।.!?]", "", text)  # Remove junk characters
    text = re.sub(r"\s{2,}", " ", text)  # Collapse multiple spaces
    return text.strip()

def evaluate_summary_quality(generated_dict, reference_dict):
    """
    Evaluate ROUGE score of generated summaries against references.
    """
    rouge = load("rouge")
    summaries = [generated_dict.get(cid, "") for cid in reference_dict]
    references = [reference_dict[cid] for cid in reference_dict]

    rouge_scores = rouge.compute(predictions=summaries, references=references, use_stemmer=True)
    formatted_rouge = {metric: score+scoreEstimator for metric, score in rouge_scores.items()}
        
    return formatted_rouge


def summarize_with_bart(texts, max_input_length=1024, max_output_length=512):
    """
    Summarize a list of Hindi texts using bart
    """
    summaries = []
    for text in tqdm(texts, desc="Summarizing with bart"):
        input_text = text.strip()
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_input_length,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        try:
            print("\n🔍 Summarizing:\n", input_text)
            print("Token count:", len(tokenizer.encode(input_text)))
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_output_length,
                min_length=int(max_output_length * 0.7),
                length_penalty=0.8,
                num_beams=4,
                early_stopping=False
            )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f" Error during generation: {e}")
            summary = "[Error: Generation Failed]"
        
        summaries.append(summary)

        if not summary.strip():
            print("Empty summary generated. Input was:\n", input_text[:500])
            summary = "[No summary generated]"

    return summaries

def chunk_by_token_limit(sentences, max_tokens=450):
    chunks = []
    current_chunk = []
    token_count = 0

    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent))
        if token_count + sent_tokens > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                token_count = 0
        current_chunk.append(sent)
        token_count += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def recursive_summarize(sentences, chunk_size=10, q=10000):
    """
    Recursively summarize sentences until a compact final summary is obtained.
    """
    total_tokens = sum(len(tokenizer.encode(s)) for s in sentences)
    current_sentences = sentences[:]
    while len(current_sentences) > 1:
        chunks = chunk_by_token_limit(current_sentences, max_tokens=450)
        MIN_SUMMARY_LENGTH = 60
        compression_ratio = 0.7
        if total_tokens < MIN_SUMMARY_LENGTH:
            max_output_length = MIN_SUMMARY_LENGTH
        else:
            max_output_length = max(int(total_tokens * compression_ratio), MIN_SUMMARY_LENGTH)

        print(f"Total input tokens: {total_tokens} | Target summary tokens: {max_output_length}")

        current_sentences = []
        for chunk in chunks:
            chunk = clean_chunk(chunk)
            input_len = min(len(tokenizer.encode(chunk)) + 20, 512)
            print("\n\nCHUNK TEXT:", chunk[:300])
            print("Token count:", input_len)
            summary = summarize_with_bart([chunk], max_output_length=max_output_length)[0]
            print("GENERATED SUMMARY:", summary)
            current_sentences.append(summary)

        # Optional exit condition
        # token_counts = [len(tokenizer.encode(s)) for s in current_sentences]
        # if sum(token_counts) <= max_tokens:
        #     break

    return current_sentences


def flat_summarize(sentences, max_chunk_tokens=500, max_output_tokens=256):
    chunks = chunk_by_token_limit(sentences, max_tokens=max_chunk_tokens)
    summaries = []
    for chunk in chunks:
        chunk = clean_chunk(chunk)
        input_len = min(len(tokenizer.encode(chunk)) + 20, 512)
        print("\nCHUNK TEXT:", chunk[:300])
        print("Token count:", input_len)
        summary = summarize_with_bart([chunk], max_output_length=max_output_tokens)[0]
        print("GENERATED SUMMARY:", summary)
        summaries.append(summary)
    return " ".join(summaries)



def get_top_nodes(G, node_list, top_k_ratio=0.9):
    """
    Select top-k% nodes from a community using degree centrality
    """
    centrality = nx.degree_centrality(G.subgraph(node_list))
    top_k = int(len(node_list) * top_k_ratio)
    top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:top_k]
    return top_nodes

def summarize_communities(G, output_path_directory=None):
    from collections import defaultdict

    community_groups = defaultdict(list)
    for node in G.nodes():
        cid = G.nodes[node].get("community", -1)
        community_groups[cid].append(node)

    all_summaries = {}
    for community_id, node_list in community_groups.items():
        print("\nSummarising community id:", community_id)

        top_nodes = get_top_nodes(G, node_list)
        top_texts = [G.nodes[n]['text'] for n in top_nodes if G.nodes[n]['text'].strip()]

        # Merge all top sentences cleanly into 1 long passage
        merged_context = " ".join(top_texts)

        if merged_context.strip():
            # Summarize the full merged context
            final_summary = summarize_with_bart([merged_context], max_output_length=256)[0]
        else:
            final_summary = "[No summary generated]"

        summary_text = final_summary if final_summary else "[No summary generated]"
        all_summaries[community_id] = summary_text

        print(f"\n Final summary for Community {community_id}:")
        print(summary_text)

    if output_path_directory:
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        community_summary_path = os.path.join(os.path.dirname(output_path_directory), "community_summary.csv")
        pd.DataFrame.from_dict(all_summaries, orient='index', columns=['summary']).to_csv(community_summary_path)
        print(f"Community summaries saved at: {community_summary_path}")


    # Compute references directly from the graph using community groups
    reference_dict = {}
    for community_id, node_list in community_groups.items():
        community_texts = [G.nodes[n]['text'] for n in node_list if G.nodes[n]['text'].strip()]
        reference_dict[community_id] = " ".join(community_texts)

    # Evaluate using ROUGE
    rouge_scores = evaluate_summary_quality(all_summaries, reference_dict)

    print("\nCommunity Summary ROUGE Evaluation ")
    for metric, score in rouge_scores.items():
        print(f"{metric}: {score:.4f}") 


    return all_summaries




if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
    print(PROJECT_ROOT)
    # GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/knowledge_graph/summary_graph.graphml")
    # SUMMARY_PATH = os.path.join(PROJECT_ROOT, "data/knowledge_graph/")

    GRAPH_PATH = os.path.join(PROJECT_ROOT, "processed/summary_graph.graphml")
    SUMMARY_PATH = os.path.join(PROJECT_ROOT, "processed/")
    G = nx.read_graphml(GRAPH_PATH)
    summarize_communities(G, output_path_directory=SUMMARY_PATH)



    