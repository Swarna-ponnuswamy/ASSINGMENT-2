# scripts/rewrite.py

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import os

def rewrite_sentence(sentence, model, tokenizer, epsilon, clip_val, device):
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    tokens = inputs['input_ids'][0]
    
    rewritten_tokens = tokens.clone()
    n = len(rewritten_tokens)
    delta_u = 2 * clip_val
    temperature = (2 * delta_u) / epsilon if epsilon > 0 else float('inf')

    for i in range(1, n - 1):
        masked_tokens = rewritten_tokens.clone()
        masked_tokens[i] = tokenizer.mask_token_id
        
        with torch.no_grad():
            outputs = model(masked_tokens.unsqueeze(0))
            logits = outputs.logits[0, i, :]

        clipped_logits = torch.clamp(logits, -clip_val, clip_val)
        
        if temperature == float('inf'):
            probs = torch.ones_like(clipped_logits)
        else:
            probs = torch.softmax(clipped_logits / temperature, dim=-1)
        
        sampled_token_id = torch.multinomial(probs, 1).item()
        rewritten_tokens[i] = sampled_token_id

    return tokenizer.decode(rewritten_tokens, skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Rewrite text using DP-MLM.")
    parser.add_argument("--model", type=str, default="roberta-base", help="MLM model name.")
    parser.add_argument("--eps", type=float, required=True, help="Epsilon privacy budget per token.")
    parser.add_argument("--clip", type=float, default=10.0, help="Clipping value C for logits [-C, C].")
    parser.add_argument("--input", type=str, required=True, help="Path to input text file.")
    parser.add_argument("--out", type=str, required=True, help="Path to output directory.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()
    
    os.makedirs(args.out, exist_ok=True)
    
    with open(args.input, 'r', encoding='utf-8') as infile, \
         open(os.path.join(args.out, f"rewritten_eps_{args.eps}.txt"), 'w', encoding='utf-8') as outfile:
        
        lines = infile.readlines()
        for line in tqdm(lines, desc=f"Rewriting with eps={args.eps}"):
            rewritten_line = rewrite_sentence(line.strip(), model, tokenizer, args.eps, args.clip, device)
            outfile.write(rewritten_line + '\n')

if __name__ == "__main__":
    main()