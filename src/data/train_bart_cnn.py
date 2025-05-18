import os
from datasets import load_dataset, Dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

def fine_tune_bart_on_cnn(project_root, csv_path, output_dir):

    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Load your CSV as a Huggingface dataset
    dataset = Dataset.from_pandas(
        load_dataset('csv', data_files=csv_path, split='train')
    )

    # Rename columns
    dataset = dataset.rename_columns({
        "article": "input_text",
        "highlights": "target_text"
    })

    # Tokenization function
    def tokenize_function(batch):
        inputs = tokenizer(batch["input_text"], max_length=1024, truncation=True, padding="max_length")
        targets = tokenizer(batch["target_text"], max_length=128, truncation=True, padding="max_length")

        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=True, # if your GPU supports
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Fine-tune
    trainer.train()

    # Save model
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

    print(f"Fine-tuning completed. Model saved at {os.path.join(output_dir, 'final_model')}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH = os.path.join(PROJECT_ROOT, "data/raw/cnn_dailymail_sample.csv")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models/bart_cnn_finetuned")
    
    fine_tune_bart_on_cnn(PROJECT_ROOT, CSV_PATH, OUTPUT_DIR)