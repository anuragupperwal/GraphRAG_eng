import os
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
# from .download_dataset import download_cnn_dailymail_subset

# === CONFIG ===
STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), "stopwords-en.txt")

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def load_english_stopwords(file_path):
    """Load stopwords from file or fallback to NLTK stopwords."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            stopwords_set = set(line.strip() for line in f.readlines())
    else:
        from nltk.corpus import stopwords
        stopwords_set = set(stopwords.words('english'))
    return stopwords_set

def remove_stopwords_english(text, stopwords_set):
    """Remove English stopwords."""
    words = text.split()
    return " ".join([word for word in words if word.lower() not in stopwords_set])

def basic_english_fix(text):
    """Fix spacing and punctuation before tokenization."""
    text = text.replace(" n't", "n't")
    text = text.replace(" '", "'")
    text = re.sub(r"\s([?.!,'])", r"\1", text)  # remove space before punctuation
    # text = re.sub(r"([a-z]) ([A-Z])", r"\1. \2", text)  # add period if lowercase followed by uppercase
    return text

def clean_text(text):
    """Basic cleaning: remove unwanted characters, extra spaces, etc."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove special characters except basic punctuation
    return text.strip()

def is_good_line(text):
    """Filter to keep only meaningful sentences."""
    text = text.strip()
    if len(text.split()) < 5:
        return False
    if sum(c.isalpha() for c in text) / max(1, len(text)) < 0.3:
        return False
    return True

def sentence_tokenize_text_grouped(sentences):
    """Sentence tokenize a list of cleaned texts using NLTK and filter out junk sentences."""
    tokenized = []
    total_sentences = 0
    kept_sentences = 0
    for sent in tqdm(sentences, desc="Tokenizing Sentences"):
        sents = sent_tokenize(sent)
        total_sentences += len(sents)
        sents = [s.strip() for s in sents if s.strip() and is_good_line(s)]
        kept_sentences += len(sents)
        tokenized.append(sents)
    # print(f"Total sentences: {total_sentences}, Kept sentences after cleaning: {kept_sentences}")
    return tokenized

def preprocess_english_corpus(RAW_DATA_PATH, max_lines=10000, stopwords_path=STOPWORDS_PATH, project_root=None):
    """Runs the complete preprocessing pipeline."""
    # Absolute paths
    input1 = RAW_DATA_PATH
    print("Project root detected as:", project_root)
    final_output = os.path.join(project_root, "data/processed/tokenized.csv")

    # Step 1: Load data
    raw_data = pd.read_csv(input1, nrows=max_lines)

    # Step 2: Basic Cleaning
    texts = raw_data['article'].fillna("").astype(str).tolist()
    texts = [basic_english_fix(text) for text in texts]
    cleaned_texts = [clean_text(text) for text in texts]
    cleaned_texts = [text for text in cleaned_texts if is_good_line(text)]

    # Step 3: Stopword removal (optional)
    # stopwords_set = load_english_stopwords(stopwords_path)
    # cleaned_texts = [remove_stopwords_english(text, stopwords_set) for text in cleaned_texts]

    # Step 4: Sentence tokenization
    tks = sentence_tokenize_text_grouped(cleaned_texts)
    joined_sentences = []
    for sentence in tks:
        tokenized_sentences = sentence_tokenize_text_grouped(sentence)
        for sent_list in tokenized_sentences:
            joined = " ".join(sent_list)  # join sentences in each document
            joined_sentences.append(joined)
            
    # Step 5: Save output
    os.makedirs(os.path.dirname(final_output), exist_ok=True)
    pd.DataFrame({"text": joined_sentences}).to_csv(final_output, index=False, encoding='utf-8')
    
    print(f"\n✅ Preprocessing completed! Final tokenized output saved at: {final_output}")

# === Entry Point ===
if __name__ == "__main__":
    RAW_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "/raw/cnn_dailymail_sample.csv"))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    preprocess_english_corpus(RAW_DATA_PATH, 1000, project_root=project_root)