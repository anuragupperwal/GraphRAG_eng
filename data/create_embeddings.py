from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os


SUMMARY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

'''
    Converts each sentence into a dense vector of 384 dimensions
    row represents one sentence’s meaning captured in 384 numbers.
'''
def generate_embeddings(INPUT_PATH, OUTPUT_PATH, nrows=10000):
    df = pd.read_csv(INPUT_PATH, nrows=nrows)
    sentences = df["summary"].fillna("").astype(str).tolist()
    model = SentenceTransformer(SUMMARY_MODEL)
    embeddings = model.encode(sentences, convert_to_numpy=True)
    np.save(OUTPUT_PATH, embeddings)
    print(f"Embeddings saved at {OUTPUT_PATH} | Shape: {embeddings.shape}")

    
def test_embeddings(OUTPUT_PATH):
    # Load the .npy file
    embeddings = np.load(OUTPUT_PATH)

    # Show shape
    print("Shape of embeddings:", embeddings.shape)
    # See a sample vector
    # print("\nSample embedding vector (first one):\n", embeddings[0])

    # Check a few stats
    print("\nStats:")
    print("Min:", np.min(embeddings))
    print("Max:", np.max(embeddings))
    print("Mean:", np.mean(embeddings))


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenized_summarized.csv")
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/processed/summarized_embeddings.npy")

    generate_embeddings(INPUT_PATH=INPUT_PATH, OUTPUT_PATH=OUTPUT_PATH)
#     test_embeddings()
