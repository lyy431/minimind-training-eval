"""Math tool-use accuracy: run train_agent.py's own multi-turn rollout (with real tool execution) and its GT check."""
import json
import os
import random
import statistics as st
import sys
import warnings

PROJECT, EVAL_FILE, OUT_JSON = sys.argv[1:4]
WEIGHTS = sys.argv[4].split(",")
N_PROBLEMS, N_SAMPLES, MAX_NEW = int(sys.argv[5]), int(sys.argv[6]), 384
sys.path.insert(0, PROJECT)
os.chdir(os.path.join(PROJECT, "trainer"))
warnings.filterwarnings("ignore")

import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer
from trainer.rollout_engine import create_rollout_engine
from trainer.train_agent import rollout_single, parse_tool_calls, validate_gt_in_text
from dataset.lm_dataset import AgentRLDataset

tokenizer = AutoTokenizer.from_pretrained("../model")
ds = AgentRLDataset(EVAL_FILE, tokenizer)
problems = [ds[i] for i in range(min(N_PROBLEMS, len(ds)))]
autocast = torch.cuda.amp.autocast(dtype=torch.bfloat16)

summary = {}
for weight in WEIGHTS:
    moe = weight.endswith("_moe")  # e.g. full_sft_moe -> out/full_sft_768_moe.pth
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=moe))
    model.load_state_dict(torch.load(f"../out/{weight[:-4] if moe else weight}_768{'_moe' if moe else ''}.pth", map_location="cuda"), strict=True)
    model = model.to("cuda").eval()
    engine = create_rollout_engine(engine_type="torch", policy_model=model, tokenizer=tokenizer, device="cuda", autocast_ctx=autocast)
    rows = []
    for pi, p in enumerate(problems):
        for s in range(N_SAMPLES):
            random.seed(10_000 * pi + s); torch.manual_seed(10_000 * pi + s)  # same seeds for every weight
            msgs = [dict(m) for m in p["messages"]]
            final, _, _, _, _, _, turns, unfinished = rollout_single(engine, tokenizer, msgs, p["tools"], max_turns=3,
                                                                    max_new_tokens=MAX_NEW, thinking_ratio=0.0, device="cuda")
            answers = [t.split("</think>", 1)[-1].strip() for t in turns]
            calls = [c for a in answers for c in parse_tool_calls(a)]
            last = answers[-1] if answers else ""
            final_text = "" if unfinished else (last.split("</tool_call>")[-1] if "</tool_call>" in last else last)
            hit = validate_gt_in_text(final_text, p["gt"])
            rows.append({"problem": pi, "gt": p["gt"], "n_calls": len(calls), "unfinished": unfinished,
                         "frac": len(hit) / len(p["gt"]), "all": len(hit) == len(p["gt"]),
                         "calc_calls": sum(c.get("name") == "calculate_math" for c in calls), "turns": turns})
    summary[weight] = rows
    del model, engine
    torch.cuda.empty_cache()

json.dump({w: [{k: v for k, v in r.items()} for r in rows] for w, rows in summary.items()},
          open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print(f"{len(problems)} held-out problems x {N_SAMPLES} samples (thinking off, max {MAX_NEW} tokens/turn, up to 3 turns)\n")
print(f"{'':30}" + "".join(f"{w:>14}" for w in WEIGHTS))
metrics = [
    ("all answers correct %", lambda R: 100 * st.mean(r["all"] for r in R)),
    ("answers correct (fraction) %", lambda R: 100 * st.mean(r["frac"] for r in R)),
    ("used a tool %", lambda R: 100 * st.mean(r["n_calls"] > 0 for r in R)),
    ("tool calls = #gt %", lambda R: 100 * st.mean(r["calc_calls"] == len(r["gt"]) for r in R)),
    ("ran out of turns %", lambda R: 100 * st.mean(r["unfinished"] for r in R)),
    ("correct | used tool %", lambda R: 100 * st.mean([r["all"] for r in R if r["n_calls"]] or [0])),
    ("correct | no tool %", lambda R: 100 * st.mean([r["all"] for r in R if not r["n_calls"]] or [0])),
]
for label, f in metrics:
    print(f"{label:30}" + "".join(f"{f(summary[w]):>14.1f}" for w in WEIGHTS))
for k in (1, 2, 3):
    print(f"{'all correct, ' + str(k) + ' expr %':30}" + "".join(
        f"{100 * st.mean([r['all'] for r in summary[w] if len(r['gt']) == k] or [0]):>14.1f}" for w in WEIGHTS))
