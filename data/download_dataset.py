import os
import argparse
from datasets import load_dataset
import pandas as pd

def download_cnn_dailymail_subset(output_path, split_percent="5%"):
    """Download CNN/DailyMail subset and save as CSV."""
    
    # Load subset of CNN/DailyMail
    dataset = load_dataset("cnn_dailymail", "3.0.0", split=f"train[:{split_percent}]")  # use version 3.0.0
    
    df = dataset.to_pandas()

    # Make sure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save 'article' and 'highlights' columns
    df[['article', 'highlights']].to_csv(output_path, index=False)
    
    print(f"CNN/DailyMail subset saved to: {output_path}")

# === Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_percent", type=str, default="2%", help="Percentage of dataset to load")
    args = parser.parse_args()
    split_percent = args.split_percent
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data/raw/cnn_dailymail.csv")
    download_cnn_dailymail_subset(OUTPUT_PATH, split_percent=split_percent)