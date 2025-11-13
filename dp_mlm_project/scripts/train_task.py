# scripts/train_task.py

import argparse
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def make_training_args(output_dir, num_epochs=1, batch_size=16):
    """
    Build TrainingArguments that work on old/new Transformers without using
    evaluation_strategy (we'll call evaluate() after training).
    """
    try:
        # Modern-ish versions (per_device_*, report_to)
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_steps=50,
            report_to="none",
            seed=42,
        )
    except TypeError:
        # Very old versions (per_gpu_*, no report_to)
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_gpu_train_batch_size=batch_size,
            per_gpu_eval_batch_size=batch_size,
            logging_steps=50,
            seed=42,
        )


def main():
    parser = argparse.ArgumentParser(description="Train a classifier on a privatized dataset.")
    parser.add_argument("--task", type=str, default="sst2", help="GLUE task name (e.g., sst2).")
    parser.add_argument("--eps", type=float, required=True, help="Epsilon value of the dataset to use.")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", help="Classifier model.")
    args = parser.parse_args()

    # Quiet TF/XLA noise (optional)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    data_path = f"data/privatized_{args.task}/eps_{args.eps}"
    data_files = {"train": f"{data_path}/train.csv", "validation": f"{data_path}/validation.csv"}
    print(f"[Info] Loading CSVs from: {data_path}", flush=True)
    dataset = load_dataset("csv", data_files=data_files)

    print(f"[Info] Columns (train): {dataset['train'].column_names}", flush=True)
    print(f"[Info] Rows (train/val): {len(dataset['train'])} / {len(dataset['validation'])}", flush=True)

    # Column names from your privatizer
    text_col = "sentence"
    label_col = "label"

    # ---- Fix 1: filter out rows with missing/empty text ----
    def _keep_ok(example):
        t = example.get(text_col, None)
        return t is not None and str(t).strip() != ""

    dataset = dataset.filter(_keep_ok)
    print(
        f"[Info] After filtering empty '{text_col}': "
        f"{len(dataset['train'])} train / {len(dataset['validation'])} val",
        flush=True,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ---- Fix 2: force everything to string before tokenizing ----
    def tokenize_function(examples):
        texts = [
            x if isinstance(x, str) else ("" if x is None else str(x))
            for x in examples[text_col]
        ]
        return tokenizer(texts, padding="max_length", truncation=True)

    print("[Info] Tokenizing...", flush=True)
    tokenized = dataset.map(tokenize_function, batched=True)

    # Metric (GLUE task's default metric; SST-2 = accuracy)
    metric = evaluate.load("glue", args.task)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    print(f"[Info] Loading model: {args.model}", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    output_dir = f"runs/training_{args.task}_eps_{args.eps}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = make_training_args(output_dir, num_epochs=1, batch_size=16)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    print("[Info] Starting training...", flush=True)
    train_output = trainer.train()
    print(f"[Info] Training finished. Train metrics: {getattr(train_output, 'metrics', {})}", flush=True)

    print("[Info] Running evaluation...", flush=True)
    eval_results = trainer.evaluate()
    print("[Result] --- Utility Results ---", flush=True)
    print(f"[Result] Epsilon: {args.eps}", flush=True)
    print(f"[Result] Raw eval dict:\n{json.dumps(eval_results, indent=2)}", flush=True)
    acc = eval_results.get("eval_accuracy") or eval_results.get("accuracy")
    print(f"[Result] Accuracy: {acc}", flush=True)

    with open(Path(output_dir) / "final_eval.json", "w") as f:
        json.dump({"epsilon": args.eps, "eval_metrics": eval_results, "accuracy": acc}, f, indent=2)
    print(f"[Info] Wrote metrics to {output_dir}/final_eval.json", flush=True)


if __name__ == "__main__":
    main()
