from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd


def fun(): 
        
    # Load your dataset from CSV
    df = pd.read_csv("fine_tune_generator_refined.csv")
    dataset = Dataset.from_pandas(df)

    # Load tokenizer and model
    model_name = "facebook/bart-large-cnn"  # or "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Preprocessing
    def preprocess(batch):
        input_texts = [f"query: {q} context: {c}" for q, c in zip(batch["query"], batch["context"])]
        target_texts = batch["answer"]

        model_inputs = tokenizer(input_texts, max_length=512, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(target_texts, max_length=128, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    # Tokenize
    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

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


fun()