"""Held-out preference accuracy: does each model put more probability on the chosen answer than the rejected one?"""
import os
import random
import statistics as st
import sys
import warnings

PROJECT, EVAL_FILE = sys.argv[1:3]
WEIGHTS = sys.argv[3].split(",")
DEVICE = sys.argv[4] if len(sys.argv) > 4 else "cuda"
LIMIT = int(sys.argv[5]) if len(sys.argv) > 5 else None
BETA = 0.15  # train_dpo.py default
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import DPODataset

tokenizer = AutoTokenizer.from_pretrained("./model")
ds = DPODataset(EVAL_FILE, tokenizer, max_length=1024)
n = min(LIMIT or len(ds), len(ds))
random.seed(0)  # DPODataset randomly drops empty <think>; fix it so every model sees identical inputs
items = [ds[i] for i in range(n)]


@torch.no_grad()
def seq_logps(model, x, y, mask):
    logits = model(x.unsqueeze(0).to(DEVICE)).logits.float()
    lp = F.log_softmax(logits, dim=-1).gather(2, y.unsqueeze(0).unsqueeze(2).to(DEVICE)).squeeze(-1)[0]
    m = mask.to(DEVICE).float()
    return (lp * m).sum().item(), m.sum().item()


scores = {}
for w in WEIGHTS:
    moe = w.endswith("_moe")  # e.g. full_sft_moe -> out/full_sft_768_moe.pth
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=moe))
    model.load_state_dict(torch.load(f"./out/{w[:-4] if moe else w}_768{'_moe' if moe else ''}.pth", map_location=DEVICE), strict=True)
    model = model.to(DEVICE).eval()
    if DEVICE == "cuda":
        model = model.half()
    rows = []
    for it in items:
        c, nc = seq_logps(model, it["x_chosen"], it["y_chosen"], it["mask_chosen"])
        r, nr = seq_logps(model, it["x_rejected"], it["y_rejected"], it["mask_rejected"])
        rows.append((c, r, nc, nr))
    scores[w] = rows
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

ref = scores[WEIGHTS[0]]
print(f"{n} held-out preference pairs (reference = {WEIGHTS[0]})\n")
print(f"{'':42}" + "".join(f"{w:>14}" for w in WEIGHTS))
rows = [
    ("chosen more likely than rejected (sum) %", lambda S: 100 * st.mean(c > r for c, r, _, _ in S)),
    ("chosen more likely (per-token avg) %", lambda S: 100 * st.mean(c / nc > r / nr for c, r, nc, nr in S)),
    ("prefers chosen more than reference %", lambda S: 100 * st.mean((c - r) - (rc - rr) > 0 for (c, r, _, _), (rc, rr, _, _) in zip(S, ref))),
    ("implicit reward margin beta*(...)", lambda S: BETA * st.mean((c - r) - (rc - rr) for (c, r, _, _), (rc, rr, _, _) in zip(S, ref))),
    ("held-out DPO loss", lambda S: st.mean(-F.logsigmoid(torch.tensor(BETA * ((c - r) - (rc - rr)))).item() for (c, r, _, _), (rc, rr, _, _) in zip(S, ref))),
    ("avg log-prob of chosen (per token)", lambda S: st.mean(c / nc for c, _, nc, _ in S)),
]
for label, f in rows:
    print(f"{label:42}" + "".join(f"{f(scores[w]):>14.3f}" for w in WEIGHTS))
