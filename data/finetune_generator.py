from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd

# Load your dataset from CSV
df = pd.read_csv("fine_tune_generator_refined.csv")
dataset = Dataset.from_pandas(df)

# Load tokenizer and model
model_name = "facebook/bart-large-cnn"  # or "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Preprocessing
def preprocess(example):
    input_text = f"query: {example['query']} context: {example['context']}"
    model_inputs = tokenizer(input_text, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(example["answer"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize
tokenized = dataset.map(preprocess, batched=True)

# Training arguments
args = Seq2SeqTrainingArguments(
    output_dir="./finetuned_generator",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=2,
    save_steps=500,
    predict_with_generate=True,
    logging_steps=10,
    report_to="none"  # disables wandb, etc.
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

# Train!
trainer.train()