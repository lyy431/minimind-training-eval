"""Freeze new small diagnostic sets, then evaluate all four comparison models.

These are descriptive checks, not a replacement for the historical benchmarks.
Usage: python official_supplement.py freeze|run
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "my_eval/results/official_compare_20261024"


def freeze():
    math_specs = [
        ("帮我算一下(617+238)*19等于多少", ["16245"]),
        ("Compute 7812/21 for me", ["372"]),
        ("帮我算一下(384+527)*16等于多少，另外7**5的结果是什么", ["14576", "16807"]),
        ("Compute (923-287)*34 for me. Also calculate 6748/28.", ["21624", "241"]),
        ("帮我算一下(735+168)*23等于多少，另外6**6的结果是什么，还有请计算9184/32", ["20769", "46656", "287"]),
        ("Compute (841-376)*27 for me. Also calculate 9**4. And calculate 7865/55.", ["12555", "6561", "143"]),
    ]
    chat_specs = [
        {"skill": "改写与指令遵循", "prompt": "把这句话改写得更礼貌，保留原意：你明天必须把报告发给我。只输出改写后的句子。", "check": "保留明天、报告和发送要求；措辞礼貌；无多余介绍"},
        {"skill": "阅读提取", "prompt": "根据这段记录回答，不要补充信息：小林周二买了三本书，周五把其中一本送给小周。小林什么时候送出了书？", "check": "周五"},
        {"skill": "直白解释", "prompt": "请用两句话解释什么是电脑缓存，面向第一次接触电脑的人。", "check": "解释临时存放常用数据以便更快取用；两句话"},
        {"skill": "JSON 格式", "prompt": "把以下信息整理成一个 JSON 对象，只包含 name 和 city 两个键：姓名是小雨，城市是杭州。不要输出其他文字。", "check": '{"name":"小雨","city":"杭州"}'},
        {"skill": "身份泄漏与列表", "prompt": "请列出两种下雨天适合在家进行的活动，每行一种，不要介绍自己。", "check": "两行合理的室内活动；不自我介绍"},
        {"skill": "避免编造缺失信息", "prompt": "根据这段资料回答：青禾公司的负责人叫林青，办公室在苏州。资料没有提供成立年份。青禾公司是哪一年成立的？", "check": "说明资料未提供或无法确定；不编造年份"},
    ]
    base = [json.loads(l) for l in (PROJECT / "dataset/agent_math_eval200.jsonl").open(encoding="utf-8")]
    original = [json.loads(l) for l in (PROJECT / "dataset/agent_rl_math.jsonl").open(encoding="utf-8")]
    seen = {m["content"].strip() for r in original for m in r["conversations"] if m["role"] == "user"}
    rows = []
    for i, (question, gt) in enumerate(math_specs):
        assert question not in seen
        row = copy.deepcopy(base[i])
        row["conversations"] = [row["conversations"][0], {"role": "user", "content": question}, {"role": "assistant", "content": ""}]
        row["gt"] = gt
        rows.append(row)
    artifacts = {"new_math6.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                 "new_chat6.json": json.dumps(chat_specs, ensure_ascii=False, indent=2) + "\n"}
    OUT.mkdir(exist_ok=True)
    for name, text in artifacts.items():
        path = OUT / name
        if path.exists():
            assert path.read_text(encoding="utf-8") == text
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
        print(name, hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
    print("Frozen before evaluation; 6 new math questions do not exactly match any user question in agent_rl_math.jsonl.", flush=True)


def run():
    import official_compare as c
    specs = c.read(OUT / "new_chat6.json")
    ds = c.AgentRLDataset(str(OUT / "new_math6.jsonl"), c.TOK)
    path = OUT / "supplement.json"
    result = c.read(path) if path.exists() else {"math": {}, "chat": {},
        "protocol": {"math": "Same as main math protocol, 6 newly constructed questions x 2 samples; exploratory only",
                     "chat": "6 predeclared prompts, greedy, thinking off, no tools, max 192 new tokens; qualitative only"},
        "input_sha256": {n: c.digest(OUT / n) for n in ("new_math6.jsonl", "new_chat6.json")}}
    for name in c.NAMES:
        if name not in result["math"]:
            m = c.load(name)
            engine = c.engine_for(m)
            rows = []
            for i in range(len(ds)):
                rows.extend(c.math_row(engine, ds[i], i, s) for s in range(2))
                print(f"New math {name} {i+1}/6", flush=True)
            result["math"][name] = rows
            c.save("supplement.json", result)
            del engine, m
            c.release()
        if name not in result["chat"]:
            m = c.load(name, half=True)
            rows = []
            for i, spec in enumerate(specs):
                prompt = c.TOK.apply_chat_template([{"role": "user", "content": spec["prompt"]}],
                    tokenize=False, add_generation_prompt=True, open_thinking=False)
                ids = c.TOK(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                t0 = time.perf_counter()
                with c.torch.no_grad():
                    out = m.generate(input_ids=ids, attention_mask=c.torch.ones_like(ids), do_sample=False,
                        temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=192, eos_token_id=c.TOK.eos_token_id)
                rows.append({**spec, "response": c.TOK.decode(out[0, ids.shape[1]:], skip_special_tokens=True),
                             "seconds": time.perf_counter()-t0, "new_tokens": out.shape[1]-ids.shape[1]})
                print(f"New chat {name} {i+1}/6", flush=True)
            result["chat"][name] = rows
            c.save("supplement.json", result)
            del m
            c.release()


if __name__ == "__main__":
    {"freeze": freeze, "run": run}[sys.argv[1]]()
