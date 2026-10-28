"""MoE router analysis: per-layer expert load, and which expert each kind of token (Chinese/English/digit/punct) goes to."""
import collections
import json
import os
import re
import sys
import warnings

PROJECT, WEIGHT, TEXT_FILE = sys.argv[1:4]
N_TEXTS = int(sys.argv[4]) if len(sys.argv) > 4 else 300
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, MOEFeedForward

tok = AutoTokenizer.from_pretrained("./model")
model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=True))
model.load_state_dict(torch.load(f"./out/{WEIGHT}_768_moe.pth", map_location="cuda"), strict=True)
model = model.to("cuda").half().eval()
E = model.config.num_experts

picks = {}  # layer -> list of chosen expert per token (current text)
def hook(layer):
    def fn(module, inputs, output):
        x = inputs[0].reshape(-1, inputs[0].shape[-1])
        picks[layer] = torch.softmax(module.gate(x).float(), -1).argmax(-1).tolist()  # top-1, same as the router
    return fn
moe_layers = [(i, l.mlp) for i, l in enumerate(model.model.layers) if isinstance(l.mlp, MOEFeedForward)]
for i, m in moe_layers:
    m.register_forward_hook(hook(i))

def kind(piece):
    s = piece.strip()
    if not s: return "space"
    if re.search(r"[一-鿿]", s): return "chinese"
    if re.fullmatch(r"[0-9.,]+", s): return "digit"
    if re.search(r"[A-Za-z]", s): return "english"
    return "punct/other"

# texts: every conversation turn of a jsonl file (DPO / SFT / RLAIF formats)
texts = []
for line in open(TEXT_FILE, encoding="utf-8"):
    row = json.loads(line)
    msgs = row.get("chosen") or row.get("conversations") or []
    texts.append("\n".join(m["content"] for m in msgs if m.get("content")))
    if len(texts) >= N_TEXTS: break

load = {i: collections.Counter() for i, _ in moe_layers}
by_kind = {i: collections.defaultdict(collections.Counter) for i, _ in moe_layers}
with torch.no_grad():
    for t in texts:
        ids = tok(t, add_special_tokens=False, truncation=True, max_length=512).input_ids
        if len(ids) < 2: continue
        model(torch.tensor([ids], device="cuda"))
        kinds = [kind(tok.decode([x])) for x in ids]
        for i, _ in moe_layers:
            for k, e in zip(kinds, picks[i]):
                load[i][e] += 1
                by_kind[i][k][e] += 1

print(f"{WEIGHT}_moe: {len(texts)} texts from {os.path.basename(TEXT_FILE)}\n")
print("expert load per layer (% of tokens routed to experts 0..3; 25/25/25/25 = perfectly balanced)")
for i, _ in moe_layers:
    tot = sum(load[i].values())
    print(f"  layer {i}: " + " ".join(f"{100 * load[i][e] / tot:5.1f}" for e in range(E)))
print("\nwhich expert each token type prefers (% of that type's tokens, per layer)")
kinds_seen = sorted({k for i, _ in moe_layers for k in by_kind[i]}, key=lambda k: -sum(by_kind[0][k].values()))
for k in kinds_seen:
    row = []
    for i, _ in moe_layers:
        c = by_kind[i][k]; tot = sum(c.values()) or 1
        top = max(range(E), key=lambda e: c[e])
        row.append(f"L{i}:e{top}({100 * c[top] / tot:.0f}%)")
    print(f"  {k:12} n={sum(by_kind[0][k].values()):6d}  " + "  ".join(row))
