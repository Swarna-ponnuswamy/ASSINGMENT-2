# scripts/privatize_glue.py

import argparse
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
from rewrite import rewrite_sentence 

def privatize_dataset(dataset, model, tokenizer, epsilon, clip_val, device):
    def privatize_example(example):
        example['sentence'] = rewrite_sentence(example['sentence'], model, tokenizer, epsilon, clip_val, device)
        return example

    return dataset.map(privatize_example, batched=False)

def main():
    parser = argparse.ArgumentParser(description="Privatize a GLUE dataset.")
    parser.add_argument("--task", type=str, default="sst2", help="GLUE task name.")
    parser.add_argument("--eps", type=float, required=True, help="Epsilon privacy budget.")
    parser.add_argument("--clip", type=float, default=10.0, help="Clipping value C.")
    parser.add_argument("--model", type=str, default="roberta-base", help="MLM model name.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = load_dataset("glue", args.task)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()

    for split in dataset.keys():
        if split == 'test': continue # Skip test set
        privatized_split = privatize_dataset(dataset[split], model, tokenizer, args.eps, args.clip, device)
        output_dir = f"data/privatized_{args.task}/eps_{args.eps}"
        os.makedirs(output_dir, exist_ok=True)
        privatized_split.to_csv(os.path.join(output_dir, f"{split}.csv"))
        print(f"Saved privatized {split} split to {output_dir}")

if __name__ == "__main__":
    main()