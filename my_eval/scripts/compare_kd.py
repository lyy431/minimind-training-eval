"""Distillation check on held-out text: loss, closeness to the MoE teacher, and generation speed."""
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
MODELS = ["full_sft", "full_dist", "full_sft_moe"]  # control, student, teacher
TEACHER = "full_sft_moe"


def load(name):
    moe = name.endswith("_moe")
    m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=moe))
    m.load_state_dict(torch.load(f"./out/{name[:-4] if moe else name}_768{'_moe' if moe else ''}.pth", map_location="cuda"), strict=True)
    return m.to("cuda").half().eval()


# held-out text: chosen answers of the 300 DPO pairs (never trained on), scored inside the chat template
rows = [json.loads(l) for l in open("dataset/dpo_eval300.jsonl", encoding="utf-8")]
texts = []
for r in rows:
    p = tok(tok.apply_chat_template(r["chosen"][:-1], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
    a = tok(r["chosen"][-1]["content"], add_special_tokens=False).input_ids
    ids = (p + a)[:768]
    texts.append((ids, min(len(p), len(ids) - 1)))

logps = {}  # name -> list of [answer_len, vocab] log-prob tensors on CPU (fp16 to save memory)
nll = {}
for name in MODELS:
    m = load(name)
    out, losses = [], []
    with torch.no_grad():
        for ids, start in texts:
            x = torch.tensor([ids], device="cuda")
            lp = F.log_softmax(m(x).logits.float()[0, start - 1:-1], -1)  # predictions for answer tokens
            losses.append(-lp.gather(1, x[0, start:, None]).mean().item())
            out.append(lp.half().cpu())
    logps[name], nll[name] = out, losses
    del m
    torch.cuda.empty_cache()

print("held-out text = chosen answers of 300 DPO pairs, in the chat template (answer tokens only)\n")
print(f"{'':40}" + "".join(f"{n:>14}" for n in MODELS))
print(f"{'loss / token (lower = better)':40}" + "".join(f"{st.mean(nll[n]):>14.4f}" for n in MODELS))
kl, agree = {}, {}
for n in MODELS[:2]:
    kls, ags = [], []
    for t, s in zip(logps[TEACHER], logps[n]):
        t, s = t.float(), s.float()
        kls.append((t.exp() * (t - s)).sum(-1).mean().item())  # KL(teacher || model) per token
        ags.append((t.argmax(-1) == s.argmax(-1)).float().mean().item())
    kl[n], agree[n] = kls, ags
print(f"{'KL(teacher || model) per token':40}" + "".join(f"{st.mean(kl[n]):>14.4f}" for n in MODELS[:2]) + f"{'-':>14}")
print(f"{'top-1 prediction same as teacher %':40}" + "".join(f"{100 * st.mean(agree[n]):>14.1f}" for n in MODELS[:2]) + f"{'-':>14}")

print()
for a, b in [("full_dist", "full_sft"), ("full_sft_moe", "full_sft"), ("full_dist", "full_sft_moe")]:
    d = [y - x for x, y in zip(nll[b], nll[a])]
    print(f"loss {a} - {b}: {st.mean(d):+.4f} +/- {st.stdev(d) / len(d) ** 0.5:.4f} (1 SE), {a} lower on {sum(v < 0 for v in d)}/{len(d)}, perplexity x{math.exp(st.mean(d)):.3f}")
d = [y - x for x, y in zip(kl["full_sft"], kl["full_dist"])]
print(f"KL to teacher, student - control: {st.mean(d):+.4f} +/- {st.stdev(d) / len(d) ** 0.5:.4f}, student closer on {sum(v < 0 for v in d)}/{len(d)}")

# greedy generation speed, batch 1
prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
           for q in ["介绍一下长城", "什么是机器学习？", "Write a short story about a cat.", "如何学好数学？", "解释一下光合作用"]]
for name in ["full_sft", "full_dist"]:
    m = load(name)
    n_tok, secs = 0, 0.0
    for i, p in enumerate(prompts[:1] + prompts):  # first one is warm-up
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        torch.cuda.synchronize(); t0 = time.time()
        out = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=200, do_sample=False,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        if i:
            n_tok += out.shape[1] - ids.shape[1]; secs += time.time() - t0
    print(f"speed {name}: {n_tok / secs:.0f} tok/s")
    del m
    torch.cuda.empty_cache()
