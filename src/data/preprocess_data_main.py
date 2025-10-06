import os
import re
import pandas as pd
from tqdm import tqdm
from src.common.paths import PROC_DIR

_SENT_SPLIT = re.compile(r'(?<=[.?!])\s+(?=[A-Z])')

def _simple_sentence_split(text: str):
    # very lightweight English-ish splitter
    parts = _SENT_SPLIT.split(text.strip())
    # keep only meaningful-ish sentences
    cleaned = []
    for s in parts:
        s = re.sub(r'\s+', ' ', s).strip()
        if len(s.split()) >= 5:
            cleaned.append(s)
    return cleaned

def preprocess_english_corpus(raw_csv_path, nrows, text_col="text", final_output=None):
    df = pd.read_csv(raw_csv_path, nrows=nrows)
    assert text_col in df.columns, f"Column '{text_col}' not found. Available: {list(df.columns)[:10]}"

    joined_sentences = []
    for txt in tqdm(df[text_col].fillna("").astype(str).tolist(), desc="Preprocessing"):
        sents = _simple_sentence_split(txt)
        joined_sentences.append(" ".join(sents))

    if final_output is None:
        final_output = os.path.join(PROC_DIR, "tokenized.csv")
    os.makedirs(os.path.dirname(final_output), exist_ok=True)
    pd.DataFrame({"text": joined_sentences}).to_csv(final_output, index=False, encoding="utf-8")