"""Black-box distillation data: a strong teacher (Qwen) solves math problems with the same mock tools,
answers are checked with train_agent.py's GT check, and correct trajectories are written in MiniMind SFT format."""
import json
import os
import sys
import time
import warnings

PROJECT, TEACHER, IN_FILE, OUT_FILE = sys.argv[1:5]
LIMIT = int(sys.argv[5]) if len(sys.argv) > 5 else None
BATCH, MAX_NEW, MAX_TURNS = 32, 256, 3
sys.path.insert(0, PROJECT)
os.chdir(os.path.join(PROJECT, "trainer"))
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trainer.train_agent import parse_tool_calls, execute_tool, validate_gt_in_text

tok = AutoTokenizer.from_pretrained(TEACHER)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=torch.bfloat16).to("cuda").eval()

rows = [json.loads(l) for l in open(IN_FILE, encoding="utf-8") if l.strip()][:LIMIT]


def tools_of(row):
    return json.loads(row["conversations"][0]["tools"])


@torch.no_grad()
def generate(chats, tools_list):
    texts = [tok.apply_chat_template(c, tools=[{"type": "function", **t} for t in tl], tokenize=False, add_generation_prompt=True)
             for c, tl in zip(chats, tools_list)]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)


stats = {"n": 0, "correct": 0, "used_tool": 0}
t0 = time.time()
with open(OUT_FILE, "w", encoding="utf-8", newline="\n") as fout:
    for b in range(0, len(rows), BATCH):
        batch = rows[b:b + BATCH]
        chats = [[{"role": "user", "content": r["conversations"][1]["content"]}] for r in batch]   # teacher's view (Qwen format)
        mini = [[{"role": "system", "content": "", "tools": r["conversations"][0]["tools"]},     # student's view (MiniMind format)
                 {"role": "user", "content": r["conversations"][1]["content"]}] for r in batch]
        tools_list = [tools_of(r) for r in batch]
        active, final, used = list(range(len(batch))), [""] * len(batch), [False] * len(batch)
        for turn in range(MAX_TURNS):
            if not active:
                break
            outs = generate([chats[i] for i in active], [tools_list[i] for i in active])
            still = []
            for i, text in zip(active, outs):
                calls = parse_tool_calls(text)
                if not calls or turn == MAX_TURNS - 1:
                    final[i] = text.strip()
                    mini[i].append({"role": "assistant", "content": final[i]})
                    continue
                used[i] = True
                prefix = text.split("<tool_call>")[0].strip()
                q_calls, m_calls = [], []
                for c in calls:
                    args = c.get("arguments", {})
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except Exception: args = {}
                    q_calls.append({"type": "function", "function": {"name": c.get("name", ""), "arguments": args}})
                    m_calls.append({"function": {"name": c.get("name", ""), "arguments": json.dumps(args, ensure_ascii=False)}})
                chats[i].append({"role": "assistant", "content": prefix, "tool_calls": q_calls})
                mini[i].append({"role": "assistant", "content": prefix, "tool_calls": json.dumps(m_calls, ensure_ascii=False)})
                for c in q_calls:  # same tool execution and result string as train_agent.rollout_single
                    result = execute_tool(c["function"]["name"], c["function"]["arguments"])
                    result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]
                    chats[i].append({"role": "tool", "content": result_str})
                    mini[i].append({"role": "tool", "content": result_str})
                still.append(i)
            active = still
        for i, r in enumerate(batch):
            ok = bool(final[i]) and len(validate_gt_in_text(final[i], r["gt"])) == len(r["gt"])
            stats["n"] += 1; stats["correct"] += ok; stats["used_tool"] += used[i]
            if ok:
                fout.write(json.dumps({"conversations": mini[i]}, ensure_ascii=False) + "\n")
        done = b + len(batch)
        print(f"{done}/{len(rows)} | correct {100 * stats['correct'] / stats['n']:.1f}% | used tool {100 * stats['used_tool'] / stats['n']:.1f}% "
              f"| {time.time() - t0:.0f}s", flush=True)

print(f"\nteacher: {stats['correct']}/{stats['n']} fully correct ({100 * stats['correct'] / stats['n']:.1f}%), "
      f"used a tool {100 * stats['used_tool'] / stats['n']:.1f}% -> {stats['correct']} SFT samples in {OUT_FILE}")
