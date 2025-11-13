import os, sacrebleu, numpy as np, pandas as pd

base_dir = "runs/form_email"
eps_values = [10, 25, 50, 100, 250]

with open("data/form_email/original.txt") as f:
    orig_lines = [l.strip() for l in f if l.strip()]

rows = []
for eps in eps_values:
    path = f"{base_dir}/rewritten_eps_{eps}.0.txt"
    if not os.path.exists(path):
        print(f"[Skip] {path} not found")
        continue

    with open(path) as f:
        priv_lines = [l.strip() for l in f if l.strip()]

    n = min(len(orig_lines), len(priv_lines))
    orig = orig_lines[:n]
    priv = priv_lines[:n]

    bleu = sacrebleu.corpus_bleu(priv, [orig]).score
    diffs = [sum(a != b for a, b in zip(o.split(), p.split())) / max(1, len(o.split()))
             for o, p in zip(orig, priv)]
    change_rate = np.mean(diffs)
    avg_len = np.mean([len(p.split()) for p in priv])

    rows.append({"epsilon": eps, "BLEU": bleu, "change_rate": change_rate, "avg_len": avg_len})
    print(f"ε={eps:<4}  BLEU={bleu:6.2f}  change={change_rate:5.2f}  len={avg_len:5.2f}")

df = pd.DataFrame(rows)
os.makedirs("runs/form_email", exist_ok=True)
df.to_csv("runs/form_email/eval_results.csv", index=False)
print("\n✅ Saved runs/form_email/eval_results.csv")
