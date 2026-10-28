"""Rebuild every dataset file these experiments derived from the official data, and check it matches dataset/.

Each block is the code that originally built the file. Outputs go to OUT_DIR (never into dataset/), then each
file's SHA256 is compared with the copy in dataset/. Usage: python make_datasets.py PROJECT OUT_DIR
Needs in dataset/: agent_rl_math.jsonl, dpo.jsonl, lora_identity.jsonl, sft_t2t_mini.jsonl, teacher_math_sft.jsonl
(teacher_math_sft.jsonl is the Qwen output from teacher_gen.py, so bb_mix.jsonl depends on it)."""
import hashlib
import json
import os
import random
import runpy
import sys

PROJECT, OUT = sys.argv[1:3]
D = os.path.join(PROJECT, "dataset")
os.makedirs(OUT, exist_ok=True)


def src(name):
    return os.path.join(D, name)


def out(name):
    return os.path.join(OUT, name)


def write(name, lines):
    open(out(name), "w", encoding="utf-8", newline="\n").writelines(lines)


def digest(xs):
    return hashlib.sha256("".join(xs).encode()).hexdigest()[:16]


# LoRA identity: replace the author's name
rows = [json.loads(l) for l in open(src("lora_identity.jsonl"), encoding="utf-8") if l.strip()]
for r in rows:
    for m in r["conversations"]:
        m["content"] = m["content"].replace("Jingyao Gong", "Yuyang LU")
write("lora_identity_yuyang.jsonl", [json.dumps(r, ensure_ascii=False) + "\n" for r in rows])

# LoRA identity + general SFT samples (stops identity leaking into other answers)
argv = sys.argv
sys.argv = ["make_mix.py", src("sft_t2t_mini.jsonl"), out("lora_identity_yuyang.jsonl"), out("lora_identity_yuyang_mix.jsonl")]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_mix.py"), run_name="__main__")
sys.argv = argv

# DPO: 300 held-out pairs; exact copies of eval pairs are dropped from train too
lines = [l.rstrip("\n") + "\n" for l in open(src("dpo.jsonl"), encoding="utf-8") if l.strip()]
held = set(random.Random(123).sample(range(len(lines)), 300))
ev = [l for i, l in enumerate(lines) if i in held]
evs = set(ev)
write("dpo_eval300.jsonl", ev)
write("dpo_train.jsonl", [l for i, l in enumerate(lines) if i not in held and l not in evs])

# Agent math: fixed 200-problem eval split (same as cloud_train.sh)
lines = [l.rstrip("\n") + "\n" for l in open(src("agent_rl_math.jsonl"), encoding="utf-8") if l.strip()]
held = set(random.Random(123).sample(range(len(lines)), 200))
eval_list = [l for i, l in enumerate(lines) if i in held]
evals = set(eval_list)
train = [l for i, l in enumerate(lines) if i not in held]
write("agent_math_eval200.jsonl", eval_list)
write("agent_math_train.jsonl", train)

# The 3000 problems of the agent RL run, which the Qwen teacher also solved (eval copies removed)
random.seed(42)
used = random.sample(train, 3000)
t3000 = [l for l in used if l not in evals]
write("agent_math_train_3000.jsonl", t3000)

# 3000 fresh problems for RL after distillation (none from the set above, none from eval)
rest = [l for l in train if l not in set(used)]
random.seed(42)
new = [l for l in random.sample(rest, 3000) if l not in evals]
write("agent_math_train_3000_excl3000.jsonl", new)
print("digests: eval200", digest(eval_list), "| train_3000", digest(t3000), "| excl3000", digest(new))

# Black-box distillation mix: teacher trajectories + 3000 general SFT dialogues (reservoir sample)
teacher = open(src("teacher_math_sft.jsonl"), encoding="utf-8").readlines()
rng = random.Random(0)
general = []
for i, line in enumerate(open(src("sft_t2t_mini.jsonl"), encoding="utf-8")):
    if len(general) < 3000:
        general.append(line)
    else:
        j = rng.randint(0, i)
        if j < 3000:
            general[j] = line
mix = teacher + general
random.Random(1).shuffle(mix)
write("bb_mix.jsonl", mix)

# Identity LoRA trained on top of bb_rl: identity mix + 400 teacher math trajectories, so tool use is not forgotten
# (the full_sft identity LoRA merged into bb_rl dropped math from 81.8% to 8.8%)
id_math = open(out("lora_identity_yuyang_mix.jsonl"), encoding="utf-8").readlines() + random.Random(7).sample(teacher, 400)
random.Random(8).shuffle(id_math)
write("lora_identity_yuyang_math.jsonl", id_math)

print(f"\n{'file':40}{'lines':>8}  same as dataset/?")
ok = True
for name in sorted(os.listdir(OUT)):
    a = open(out(name), "rb").read()
    b = open(src(name), "rb").read() if os.path.exists(src(name)) else None
    same = b is not None and hashlib.sha256(a).digest() == hashlib.sha256(b).digest()
    ok &= same
    n_lines = a.count(b"\n")
    print(f"{name:40}{n_lines:>8}  {'yes' if same else ('MISSING in dataset/' if b is None else 'NO')}")
print("\nall files reproduced exactly" if ok else "\nsome files differ")
