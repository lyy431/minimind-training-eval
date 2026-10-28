"""Language-modelling check: loss per answer token on held-out chat text (chosen answers of dpo_eval300.jsonl, never
trained on, inside the chat template), paired against the first model. Lower = predicts real answers better.
Reports the historical 300-row metric and a corrected answer-only metric that
excludes rows whose answers were entirely removed by truncation.
Usage: heldout_loss.py PROJECT WEIGHT[,WEIGHT...]   (names ending in _moe load the MoE weights)"""
import json
import math
import os
import statistics as st
import sys
import warnings

PROJECT = sys.argv[1]
MODELS = sys.argv[2].split(",")
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

tok = AutoTokenizer.from_pretrained("./model")


def load(name):
    moe = name.endswith("_moe")
    m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=moe))
    m.load_state_dict(torch.load(f"./out/{name[:-4] if moe else name}_768{'_moe' if moe else ''}.pth", map_location="cuda"), strict=True)
    return m.to("cuda").half().eval()


rows = [json.loads(l) for l in open("dataset/dpo_eval300.jsonl", encoding="utf-8")]
texts = []
valid_indices = []
for i, r in enumerate(rows):
    p = tok(tok.apply_chat_template(r["chosen"][:-1], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
    a = tok(r["chosen"][-1]["content"], add_special_tokens=False).input_ids
    ids = (p + a)[:768]
    texts.append((ids, min(len(p), len(ids) - 1)))
    if len(p) < len(ids):
        valid_indices.append(i)

nll = {}
for name in MODELS:
    m = load(name)
    losses = []
    with torch.no_grad():
        for ids, start in texts:
            x = torch.tensor([ids], device="cuda")
            lp = F.log_softmax(m(x).logits.float()[0, start - 1:-1], -1)
            losses.append(-lp.gather(1, x[0, start:, None]).mean().item())
    nll[name] = losses
    del m
    torch.cuda.empty_cache()

base = MODELS[0]
if len(valid_indices) < 2:
    raise ValueError("Fewer than two answers survive truncation; cannot compute paired standard errors")
excluded = [i for i in range(len(texts)) if i not in valid_indices]
print(f"dpo_eval300: {len(valid_indices)}/{len(texts)} answers survive 768-token truncation; excluded row indices (zero-based): {excluded}\n")
for label, indices in [("corrected answer-only", valid_indices), ("historical metric (includes one prompt token for fully truncated answers)", list(range(len(texts))))]:
    print(f"{label}: {len(indices)} rows; mean of per-row token losses, lower = better")
    for name in MODELS:
        values = [nll[name][i] for i in indices]
        line = f"{name:16} loss {st.mean(values):.4f}  perplexity {math.exp(st.mean(values)):6.2f}"
        if name != base:
            d = [nll[name][i] - nll[base][i] for i in indices]
            line += f"   vs {base}: {st.mean(d):+.4f} +/- {st.stdev(d) / len(d) ** 0.5:.4f} (1 SE), lower on {sum(v < 0 for v in d)}/{len(d)}"
        print(line)
    print()
