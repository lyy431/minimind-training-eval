"""Build lora_identity_yuyang_mix.jsonl: identity samples x2 + general SFT samples without identity content."""
import json
import random
import sys

sft_path, identity_path, out_path = sys.argv[1:4]
N_GENERAL = 200
SKIP_WORDS = ["minimind", "jingyao", "你是谁", "你叫什么", "名字", "开发", "创造", "openai", "chatgpt",
              "gpt", "模型", "基于", "人工智能助手", "ai助手", "ai 助手", "assistant", "<tool", "<think"]

random.seed(42)
general = []
with open(sft_path, encoding="utf-8") as f:
    for line in f:
        if random.random() > 0.01:
            continue
        row = json.loads(line)
        conv = row.get("conversations", [])[:2]  # first user/assistant exchange of each dialogue
        if len(conv) != 2 or [m.get("role") for m in conv] != ["user", "assistant"]:
            continue
        if any(set(m) - {"role", "content"} for m in conv):  # skip tool / reasoning samples
            continue
        text = (conv[0]["content"] + conv[1]["content"]).lower()
        if len(text) > 450 or any(w in text for w in SKIP_WORDS):
            continue
        general.append({"conversations": [{"role": m["role"], "content": m["content"]} for m in conv]})
        if len(general) >= N_GENERAL:
            break

identity = [json.loads(l) for l in open(identity_path, encoding="utf-8") if l.strip()]
rows = identity * 2 + general
random.shuffle(rows)
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"identity {len(identity)} x2 + general {len(general)} = {len(rows)} samples -> {out_path}")
for r in general[:3]:
    print("general example:", json.dumps(r, ensure_ascii=False)[:160])
