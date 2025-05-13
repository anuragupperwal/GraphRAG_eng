import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from .clean_summarised_csv import clean_summarized_csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "facebook/bart-large-cnn"
# path_to_model = "models/bart_cnn_finetuned/final_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)


import pandas as pd
from datasets import load_metric

def evaluate_rouge_for_summaries(reference_path, prediction_path, reference_col="text", prediction_col="summary", max_rows=None):
    """
    Computes ROUGE scores between reference texts and generated summaries.

    Parameters:
    - reference_path (str): Path to the original tokenized CSV.
    - prediction_path (str): Path to the generated summaries CSV.
    - reference_col (str): Column name in the reference file. Default is 'text'.
    - prediction_col (str): Column name in the summary file. Default is 'summary'.
    - max_rows (int): Optionally limit the number of rows for faster testing.

    Returns:
    - dict: ROUGE-1, ROUGE-2, ROUGE-L scores
    """
    rouge = load_metric("rouge")

    ref_df = pd.read_csv(reference_path, nrows=max_rows)
    pred_df = pd.read_csv(prediction_path, nrows=max_rows)

    references = ref_df[reference_col].fillna("").astype(str).tolist()
    predictions = pred_df[prediction_col].fillna("").astype(str).tolist()

    assert len(references) == len(predictions), "Mismatch in number of rows"

    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )

    # Format scores nicely
    formatted_results = {
        metric: score.mid.fmeasure for metric, score in results.items()
    }

    return formatted_results

def group_sentences(sentences, chunk_size=5):  
    """Group smaller chunks for summarization."""
    return [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]


def summarize_chunks(chunks):
    """Summarize each chunk using the loaded mT5 model."""
    summaries = []
    for chunk in tqdm(chunks, desc="Summarizing"):
        if not chunk.strip() or len(chunk.split()) < 10:  # Skip small meaningless chunks
            summaries.append("")
            continue

        input_text = chunk
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True).to(device)
        input_length = len(tokenizer.encode(input_text, truncation=True, max_length=1024))
        max_output_length = max(30, int(input_length * 0.8))
        summary_ids = model.generate(
            inputs,
            max_length=min(512, max_output_length),
            min_length=int(max_output_length * 0.6),
            length_penalty=1.1,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(f"Total Chunks: {len(chunks)}, Summarized Chunks: {len([s for s in summaries if s.strip()])}")
        summaries.append(summary)

    total_chunks = len(chunks)
    summarized_chunks = len([s for s in summaries if s.strip()])
    print(f"Total Chunks Given: {total_chunks}, Successful Summaries: {summarized_chunks}, Empty/Skipped: {total_chunks - summarized_chunks}")

    return summaries



def summarize_corpus(input_path, output_path, chunk_size=4, max_lines=10000):
    """Main callable function to run summarization pipeline."""
    
    print(f"Loading tokenized input from: {input_path}")
    df = pd.read_csv(input_path, nrows=max_lines)
    sentences = df['text'].fillna("").astype(str).tolist()

    print(f"Loaded {len(df)} rows for summarization.")
    
    print("Grouping sentences...")
    grouped_chunks = group_sentences(sentences, chunk_size=chunk_size)

    print("Running summarization...")
    final_summaries = summarize_chunks(grouped_chunks)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame({"summary": final_summaries}).to_csv(output_path, index=False, encoding='utf-8')

    print(f"Summarized data saved to: {output_path}")

    #Rouge score test for summarisation quality
    rouge_scores = evaluate_rouge_for_summaries(input_path, output_path)
    print("ROUGE scores for chunk-level summarization:", rouge_scores)

    # PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # clean_summarized_csv(output_path, cleaned_path)


# #Run directly for testing
if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    TOKENIZED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenized.csv")
    SUMMARY_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenized_summarized.csv")
    # SUMMARY_CLEANED_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenized_summarized_cleaned.csv")
    summarize_corpus(
        input_path=TOKENIZED_DATA_PATH,
        output_path=SUMMARY_OUTPUT_PATH,
        chunk_size=1,
        max_lines=10
    )