import os
import pandas as pd
import re
from tqdm import tqdm
from fuzzywuzzy import fuzz

def remove_fuzzy_duplicate_sentences(text, similarity_threshold=90):
    """Remove fuzzy duplicate sentences (even if minor word changes)."""
    sentences = re.split(r'(?<=[.?!])\s+', text)
    filtered = []
    for sent in sentences:
        if sent.strip() == "":
            continue
        duplicate = False
        for existing in filtered:
            if fuzz.ratio(existing.lower(), sent.lower()) > similarity_threshold:
                duplicate = True
                break
        if not duplicate:
            filtered.append(sent)
    return " ".join(filtered)

def is_good_summary(text):
    """Check if a summary is meaningful and clean."""
    text = text.strip()
    if len(text.split()) < 8:  # Too few words
        return False
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars / max(1, len(text)) < 0.4:  # Too few letters
        return False
    return True

def basic_text_fix(text):
    """Light basic cleaning for spacing and punctuation."""
    text = text.replace(" .", ".")
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
    text = re.sub(r'(\.\.+)', '.', text)  # Fix multiple periods
    return text.strip()

def clean_summarized_csv(input_path, output_path):
    print(f"Loading summarized input from: {input_path}")
    df = pd.read_csv(input_path)

    original_count = len(df)
    summaries = df['summary'].fillna("").astype(str).tolist()

    cleaned_summaries = []
    for summary in tqdm(summaries, desc="Cleaning Summaries"):
        summary = basic_text_fix(summary)
        summary = remove_fuzzy_duplicate_sentences(summary)  # use fuzzy duplicate removal        
        if is_good_summary(summary):
            cleaned_summaries.append(summary)

    cleaned_df = pd.DataFrame({"summary": cleaned_summaries})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"✅ Cleaning complete. Original rows: {original_count}, After cleaning: {len(cleaned_summaries)}")
    print(f"Cleaned summaries saved to: {output_path}")

# === Entry point ===
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_SUMMARIZED_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenized_summarized.csv")
    CLEANED_SUMMARIZED_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenized_summarized.csv")

    clean_summarized_csv(
        input_path=RAW_SUMMARIZED_PATH,
        output_path=CLEANED_SUMMARIZED_PATH
    )