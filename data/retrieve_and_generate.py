import os
import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



def load_summaries(csv_path):
    df = pd.read_csv(csv_path)
    texts = df["summary"].fillna("").tolist()
    return texts, df

def get_embeddings(texts, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

def save_metadata(df, output_meta_path):
    df.to_pickle(output_meta_path)

def build_faiss_index(project_root, index_path, meta_path):
    SUMMARY_CSV_PATH = os.path.join(project_root, "data", "knowledge_graph", "community_summary.csv")

    texts, df = load_summaries(SUMMARY_CSV_PATH)
    embeddings = get_embeddings(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    save_metadata(df, meta_path)

    print(f"FAISS index saved to {index_path}")
    print(f"Metadata saved to {meta_path}")



def load_index_and_metadata(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def get_query_embedding(query, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    model = SentenceTransformer(model_name)
    embedding = model.encode([query])
    return embedding.astype("float32")

def search_index(index, embedding, metadata, top_k=5):
    distances, indices = index.search(embedding, top_k)
    results = metadata.iloc[indices[0]].copy()
    results["score"] = distances[0]
    return results

def get_top_k_similar_summaries(input_paragraph, summaries, embeddings, k=3):
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    input_embedding = model.encode([input_paragraph], convert_to_numpy=True)

    cosine_scores = util.cos_sim(input_embedding, embeddings)[0]
    cosine_scores = cosine_scores.detach().cpu().numpy().flatten()
    
    k = min(k, len(cosine_scores))
    top_indices = np.argsort(cosine_scores)[0-k:][::-1]
    valid_top_indices = [i for i in top_indices if 0 <= i < len(summaries)]
    top_summaries = [summaries[i] for i in valid_top_indices]

    print("\nCosine Similarity based Summaries ranking:")
    for idx in valid_top_indices:
        print(f"Index: {idx}, Score: {cosine_scores[idx]:.4f}")
        print(f"Summary: {summaries[idx]}\n")
        
    return " ".join(top_summaries)

def load_generator(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    return tokenizer, model

# Available model options
GENERATION_MODELS = {
    "BART": "facebook/bart-large-cnn",
    "BART_finetuned": "./finetuned_generator"
}

def generate_summary(text, tokenizer, model):
    # prompt = f"हिंदी में सारांश दें: {text}"
    prompt = f"summarize: {text}"
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    summary_ids = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
   
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)



def generate_output(top_k, query, model_name, project_root, summary_path, embedding_path, output_path):
    INDEX_PATH = os.path.join(project_root, "data", "knowledge_graph", "community_faiss.index")
    META_PATH = os.path.join(project_root, "data", "knowledge_graph", "community_meta.pkl")

    build_faiss_index(project_root, index_path=INDEX_PATH, meta_path=META_PATH)
    index, metadata = load_index_and_metadata(INDEX_PATH, META_PATH)

    query_embedding = get_query_embedding(query)
    top_results = search_index(index, query_embedding, metadata, top_k=top_k)
    print("\nTop Matching Community Summaries:")
    for i, row in top_results.iterrows():
        print(f"\nCommunity: {i} | Score: {row['score']:.4f}")
        print(f"Summary: {row['summary']}")

    summaries = top_results["summary"].tolist()
    embeddings = np.load(embedding_path)
    context = get_top_k_similar_summaries(query, summaries, embeddings, k=top_k)


    tokenizer, generator = load_generator(GENERATION_MODELS[model_name])
    combined_text = context + " " + query
    final_summary = generate_summary(combined_text,tokenizer,generator)
    print("Generated: ", final_summary)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_summary)
    
    # if os.path.exists("your_reference_file.csv"):
    #     reference = "Driver Gary Babineau: ""I probably had a 30 35foot free fall"" ""The whole bridge from one side of the Mississippi to the other just completely gave way"" ""I literally thought I was dead,"" says driver of a pickup truck that was dangling over the edge of a brokenoff section of the bridge. EMTs and other officials managed to get 55 people into ambulances in less than two hours. ""It was just like out of the movies I just knew what I had to do at the moment,"" says EMT Melissa Hughes 32 of Minneapolis Volunteers. ""I felt that twice I felt that I was going to fall off the bridge,"" says Driver Gary Babinau. ""Theres cars in the water there cars on fire The whole bridge is down,"" says another driver Bernie Toivonen, who was on a part of bridge that ended up tilted at."
    #     metrics = evaluate_generation(reference, final_summary)
    #     print("\nEvaluation (ROUGE) against reference answer:")
    #     for metric, val in metrics.items():
    #         print(f"{metric}: {val:.4f}")


if __name__ == "__main__":
    # Define basic project paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(project_root, ".."))

    # Parameters
    top_k = 5  # How many top communities to fetch
    query = "Tell me about Michael Jackson concert updates and ticket changes."  
    model_name = "BART"

    summary_path = os.path.join(project_root, "data", "knowledge_graph", "community_summary.csv")
    embedding_path = os.path.join(project_root, "data", "processed", "summarized_embeddings.npy")
    output_path = os.path.join(project_root, "data", "output", "final_answer.txt")

    # Run the pipeline
    generate_output(
        top_k=top_k,
        query=query,
        model_name=model_name,
        project_root=project_root,
        summary_path=summary_path,
        embedding_path=embedding_path,
        output_path=output_path
    )