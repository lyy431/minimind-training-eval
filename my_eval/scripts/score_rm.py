"""Score any set of weights on held-out rlaif prompts with the GRPO training reward (same prompts/seeds as score_grpo3.py)."""
import json
import os
import random
import re
import statistics as st
import sys
import warnings

PROJECT, RM_PATH, OUT_JSON = sys.argv[1:4]
MODELS = sys.argv[4].split(",")
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import LMForRewardModel

N_PROMPTS, N_SAMPLES, MAX_NEW, MAX_PROMPT = 30, 3, 512, 768

# Exclude the subsets the GRPO runs trained on (seed 42, 1000 / 5000 lines), then the same 30 prompts as before
lines = [l.rstrip("\n") + "\n" for l in open("dataset/rlaif.jsonl", encoding="utf-8") if l.strip()]
trained = set()
for k in (1000, 5000):
    random.seed(42)
    trained |= set(random.sample(lines, k))
held_out = [l for l in lines if l not in trained]
convs = [json.loads(l)["conversations"][:-1] for l in random.Random(7).sample(held_out, N_PROMPTS)]


def rep_penalty(text, n=3, cap=0.5):  # copied from trainer/train_grpo.py
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def score(response, messages, rm):  # same terms as calculate_rewards() in train_grpo.py
    r = {"len": 0.5 if 20 <= len(response.strip()) <= 800 else -0.5, "think": 0.0, "close": 0.0}
    answer, r["think_chars"] = response, None
    if "</think>" in response:
        think, answer = response.split("</think>", 1)
        r["think"] = 1.0 if 20 <= len(think.strip()) <= 300 else -0.5
        r["close"] = 0.25 if response.count("</think>") == 1 else -0.25
        r["think_chars"], answer = len(think.strip()), answer.strip()
    r["rep"] = rep_penalty(answer)
    r["rm"] = rm.get_score(messages, answer)
    r["answer_chars"] = len(answer)
    r["total"] = r["len"] + r["think"] + r["close"] - r["rep"] + r["rm"]
    return r


tokenizer = AutoTokenizer.from_pretrained("./model")
rm = LMForRewardModel(RM_PATH, device="cuda", dtype=torch.float16)
results = {}
for weight in MODELS:
    moe = weight.endswith("_moe")  # e.g. full_sft_moe -> out/full_sft_768_moe.pth
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=moe))
    model.load_state_dict(torch.load(f"./out/{weight[:-4] if moe else weight}_768{'_moe' if moe else ''}.pth", map_location="cuda"), strict=True)
    model = model.to("cuda").half().eval()
    rows = []
    for i, conv in enumerate(convs):
        text = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, open_thinking=True)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[:, -MAX_PROMPT:].to("cuda")
        torch.manual_seed(1000 + i)  # same seed per prompt for every model
        with torch.no_grad():
            out = model.generate(input_ids=ids.repeat(N_SAMPLES, 1), attention_mask=torch.ones_like(ids).repeat(N_SAMPLES, 1),
                                 max_new_tokens=MAX_NEW, do_sample=True, temperature=0.8,
                                 pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        messages = [{"role": m["role"], "content": m["content"]} for m in conv]
        for resp in tokenizer.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True):
            rows.append({"prompt": i, "response": resp, **score(resp, messages, rm)})
    results[weight] = rows
    del model
    torch.cuda.empty_cache()

json.dump({"prompts": convs, "results": results}, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print(f"{N_PROMPTS} held-out prompts x {N_SAMPLES} samples, thinking on, temperature 0.8, max {MAX_NEW} tokens\n")
print(f"{'':22}" + "".join(f"{w:>14}" for w in MODELS))
for key, label in [("total", "total reward"), ("rm", "  reward-model score"), ("len", "  length term"),
                   ("think", "  think-length term"), ("close", "  </think> term"), ("rep", "  repetition penalty")]:
    print(f"{label:22}" + "".join(f"{st.mean(r[key] for r in results[w]):>14.3f}" for w in MODELS))
print(f"{'answer chars (mean)':22}" + "".join(f"{st.mean(r['answer_chars'] for r in results[w]):>14.1f}" for w in MODELS))
for b, test in [("empty", lambda t: t in (None, 0)), ("20-300", lambda t: t and 20 <= t <= 300)]:
    print(f"{'think ' + b:22}" + "".join(f"{sum(bool(test(r['think_chars'])) for r in results[w]):>14d}" for w in MODELS))


def paired(x, y, key):
    d = [st.mean(r[key] for r in results[x] if r["prompt"] == i) - st.mean(r[key] for r in results[y] if r["prompt"] == i) for i in range(N_PROMPTS)]
    return st.mean(d), st.stdev(d) / len(d) ** 0.5, sum(v > 0 for v in d)


print()
for x in MODELS[1:]:
    for key in ("total", "rm"):
        m, se, w = paired(x, MODELS[0], key)
        print(f"{x} - {MODELS[0]} [{key:5}]: {m:+.3f} +/- {se:.3f} (1 SE), better on {w}/{N_PROMPTS} prompts")
