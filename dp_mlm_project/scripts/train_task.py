# scripts/train_task.py

import argparse
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Train a classifier on a privatized dataset.")
    parser.add_argument("--task", type=str, default="sst2", help="GLUE task name.")
    parser.add_argument("--eps", type=float, required=True, help="Epsilon value of the dataset to use.")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Classifier model.")
    args = parser.parse_args()

    data_path = f"data/privatized_{args.task}/eps_{args.eps}"
    dataset = load_dataset('csv', data_files={'train': f'{data_path}/train.csv', 'validation': f'{data_path}/validation.csv'})

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    metric = evaluate.load("glue", args.task)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=f"runs/training_{args.task}_eps_{args.eps}",
        evaluation_strategy="epoch",
        num_train_epochs=1, # Keep it to 1 for a quick test
        per_device_train_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"--- Utility Results ---")
    print(f"Epsilon: {args.eps}")
    print(f"Accuracy: {eval_results.get('eval_accuracy')}")

if __name__ == "__main__":
    main()