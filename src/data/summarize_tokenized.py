import os
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
from src.common.paths import PROC_DIR

# load environment key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def _summarize_batch_gemini(batch_texts):
    model = genai.GenerativeModel("gemini-2.5-flash")
    summaries = []
    for text in tqdm(batch_texts, desc="Gemini summarization"):
        prompt = (
            "You are summarizing for a knowledge graph retrieval system.\n"
            "Create a long, information-rich rewrite of the following text, "
            "preserving *all* factual details and combining related sentences coherently.\n\n"
            f"Text:\n{text}\n\nSummary:"
        )
        try:
            response = model.generate_content(prompt)
            summaries.append(response.text.strip())
        except Exception as e:
            print(f"[Warning] Gemini API error: {e}")
            summaries.append(text[:4000])  # fallback: truncated raw text
    return summaries


def summarize_corpus(input_path, output_path=None, batch_size=5):
    df = pd.read_csv(input_path)
    texts = df["text"].fillna("").astype(str).tolist()

    # process in batches to avoid hitting rate limits
    summaries = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        summaries.extend(_summarize_batch_gemini(batch))

    if output_path is None:
        output_path = os.path.join(PROC_DIR, "tokenized_summarized.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame({"summary": summaries}).to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(summaries)} summaries to {output_path}")