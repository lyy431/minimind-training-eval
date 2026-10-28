"""Dense vs MoE: held-out next-token loss (paired per text) and greedy generation speed."""
import json
import math
import os
import statistics as st
import sys
import time
import warnings

PROJECT = sys.argv[1]
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


# held-out text: the chosen answers of the 300 DPO pairs no model trained on
rows = [json.loads(l) for l in open("dataset/dpo_eval300.jsonl", encoding="utf-8")]
plain = [tok(r["chosen"][-1]["content"], add_special_tokens=False, truncation=True, max_length=512).input_ids for r in rows]
# same answers inside the chat template, scoring only the answer tokens (how an SFT model is used)
chat = []
for r in rows:
    prompt = tok.apply_chat_template(r["chosen"][:-1], tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    a_ids = tok(r["chosen"][-1]["content"], add_special_tokens=False).input_ids
    ids = (p_ids + a_ids)[:768]
    chat.append((ids, min(len(p_ids), len(ids) - 1)))


@torch.no_grad()
def nll(model, ids, start):
    x = torch.tensor([ids], device="cuda")
    logp = F.log_softmax(model(x).logits.float()[0, :-1], -1).gather(1, x[0, 1:, None])[:, 0]
    return -logp[max(start - 1, 0):].mean().item()


res = {}
for name in ["pretrain", "pretrain_moe", "full_sft", "full_sft_moe"]:
    m = load(name)
    res[name] = {"plain": [nll(m, ids, 1) for ids in plain if len(ids) > 2],
                 "chat": [nll(m, ids, s) for ids, s in chat]}
    if name.startswith("full_sft"):  # greedy speed, batch 1
        prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
                   for q in ["介绍一下长城", "什么是机器学习？", "Write a short story about a cat.", "如何学好数学？", "解释一下光合作用"]]
        n_tok, secs = 0, 0.0
        for i, p in enumerate(prompts + prompts[:1]):  # first one is warm-up
            ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
            torch.cuda.synchronize(); t0 = time.time()
            out = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=200, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            torch.cuda.synchronize()
            if i > 0:
                n_tok += out.shape[1] - ids.shape[1]; secs += time.time() - t0
        res[name]["speed"] = n_tok / secs
    del m
    torch.cuda.empty_cache()

print("held-out text = chosen answers of 300 DPO pairs (never trained on)\n")
print(f"{'':34}{'pretrain':>10}{'pretrain_moe':>14}{'full_sft':>10}{'full_sft_moe':>14}")
for key, label in [("plain", "plain text: loss / token"), ("chat", "in chat template: loss / token")]:
    print(f"{label:34}" + "".join(f"{st.mean(res[n][key]):>{w}.3f}" for n, w in
                                   [("pretrain", 10), ("pretrain_moe", 14), ("full_sft", 10), ("full_sft_moe", 14)]))
print()
for d, m in [("pretrain", "pretrain_moe"), ("full_sft", "full_sft_moe")]:
    for key in ("plain", "chat"):
        diff = [b - a for a, b in zip(res[d][key], res[m][key])]
        print(f"{m} - {d} [{key:5}]: loss {st.mean(diff):+.4f} +/- {st.stdev(diff) / len(diff) ** 0.5:.4f} (1 SE), "
              f"MoE lower on {sum(x < 0 for x in diff)}/{len(diff)} texts, perplexity x{math.exp(st.mean(diff)):.3f}")
print(f"\ngreedy generation speed (batch 1, fp16): full_sft {res['full_sft']['speed']:.0f} tok/s | full_sft_moe {res['full_sft_moe']['speed']:.0f} tok/s")
