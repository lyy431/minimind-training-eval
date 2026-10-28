"""Identity repair: frozen development sets, prompting, gentle LoRA, final evaluation.

All files and caches stay on D:. Final benchmarks are used only after selection.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import re
import statistics as st
import sys
import time

import official_compare as c

P = c.PROJECT
OUT = P / "my_eval/results/identity_repair_20261024"
OUT.mkdir(exist_ok=True)
torch, TOK = c.torch, c.TOK
SYS = {
    "prompt_short": "你是由 Yuyang LU 开发的 MiniMind。",
    "prompt_conditional": "你是 MiniMind，由 Yuyang LU 开发。只有用户询问你的名字、身份或开发者时才介绍自己；其他问题直接回答，不提及身份。",
}
DEV_ID = ["说说你叫什么，以及是谁开发了你。", "请告诉我构建这个助手的人的姓名。", "如果我要介绍你，应该写哪位开发者？",
    "你这个人工智能的开发者全名是什么？", "请用一句话交代你的名称和开发者。", "是谁训练并开发了你这个助手？",
    "State your assistant name and the full name of your developer.", "Who is responsible for developing this assistant?",
    "Name the person who built this AI assistant.", "Give your developer's full name in one sentence."]
DEV_GEN = ["请简短说明什么是文件夹。", "用一句话解释为什么植物需要阳光。", "给第一次做饭的人两条建议。", "写一句祝朋友考试顺利的话。",
    "把“我昨天去公园散步”翻译成英文。", "根据记录回答：阿宁周一寄出包裹，周三送达。包裹哪天送达？", "用两句话介绍中国的春天。",
    "Explain what a computer keyboard is in one sentence.", "Give two ideas for a quiet indoor hobby.", "请写一条提醒大家带雨伞的通知。"]
FINAL_ID = ["你是谁？", "Who created you?", "你的作者叫什么名字？", "是哪个人把你做出来的？", "介绍一下你的开发者", "Who is your developer?", "你和 Jingyao Gong 是什么关系？"]
FINAL_GEN = ["解释什么是机器学习", "推荐一些中国的美食", "如何开始学习Python？", "写一首关于秋天的短诗", "为什么要多喝水？",
    "给我讲一个笑话", "怎样才能提高睡眠质量？", "太阳系有几颗行星？", "帮我写一封请假条", "推荐几本值得读的书"]


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_rows(name, rows):
    path = OUT / name
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if path.exists():
        assert path.read_text(encoding="utf-8") == text, f"Refusing to alter frozen dataset {name}"
    else:
        path.write_text(text, encoding="utf-8", newline="\n")


def save(name, obj):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    tmp.replace(path)


def norm(text):
    return re.sub(r"\W+", "", text.casefold())


def question(row):
    conv = row.get("conversations", row.get("chosen", []))
    return "\n".join(m["content"] for m in conv if m["role"] == "user")


def add_system(messages, system):
    msgs = copy.deepcopy(messages)
    if system:
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] = system + ("\n" + msgs[0]["content"] if msgs[0].get("content") else "")
        else:
            msgs.insert(0, {"role": "system", "content": system})
    return msgs


def load(spec, half=True):
    return c.load("bb_rl" if spec in SYS else spec, half=half)


def prepare():
    tests = read_rows(P / "dataset/agent_math_eval200.jsonl")
    used = read_rows(P / "dataset/agent_math_train_3000.jsonl") + read_rows(P / "dataset/agent_math_train_3000_excl3000.jsonl")
    excluded = {norm(question(r)) for r in tests + used}
    excluded |= {norm(question(r)) for r in read_rows(P / "my_eval/results/official_compare_20261024/new_math6.jsonl")}
    pool = {}
    for row in read_rows(P / "dataset/agent_rl_math.jsonl"):
        key = norm(question(row))
        if key not in excluded:
            pool.setdefault(key, row)
    rng = random.Random(202610242)
    dev_math = []
    for k in (1, 2, 3):
        dev_math += rng.sample([r for r in pool.values() if len(r["gt"]) == k], 8)
    rng.shuffle(dev_math)
    write_rows("dev_math24.jsonl", dev_math)
    dev_keys = {norm(question(r)) for r in dev_math}
    final_loss = read_rows(P / "dataset/dpo_eval300.jsonl")
    final_keys = {norm(question(r)) for r in final_loss}
    language = []
    for row in read_rows(P / "dataset/dpo_train.jsonl"):
        if norm(question(row)) in final_keys:
            continue
        text = TOK.apply_chat_template(row["chosen"][:-1], tokenize=False, add_generation_prompt=True)
        plen = len(TOK(text, add_special_tokens=False).input_ids)
        if 8 < plen < 550 and row["chosen"][-1]["content"].strip():
            language.append(row)
    dev_language = rng.sample(language, 100)
    write_rows("dev_language100.jsonl", dev_language)
    reserved_questions = {norm(q) for q in DEV_ID + DEV_GEN + FINAL_ID + FINAL_GEN}
    reserved_questions |= {norm(question(r)) for r in dev_language + final_loss}
    reserved_questions |= {norm(m["content"]) for conv in c.prompts() for m in conv if m["role"] == "user"}
    identity = []
    for row in read_rows(P / "dataset/lora_identity_yuyang.jsonl"):
        q = question(row)
        if norm(q) in reserved_questions:
            continue
        # Identity-focused examples only; remove greetings and generic capability claims.
        if "yuyang" not in json.dumps(row).casefold():
            continue
        if not re.search(r"谁|名字|姓名|叫|身份|开发|作者|介绍|你是|描述自己|who|name|develop|creat|built|trained|based|yourself|are you", q, re.I):
            continue
        identity.append(row)
    assert len(identity) >= 30
    write_rows("identity_train.jsonl", identity)
    general = []
    seen = set()
    for row in read_rows(P / "dataset/bb_mix.jsonl"):
        text = json.dumps(row, ensure_ascii=False).casefold()
        key = norm(question(row))
        if key in reserved_questions or key in seen or any(w in text for w in ("yuyang", "minimind", "jingyao", "<tool_call>")):
            continue
        if any(m.get("tools") or m.get("tool_calls") or m["role"] == "tool" for m in row["conversations"]):
            continue
        seen.add(key)
        general.append(row)
    general = rng.sample(general, 1000)
    write_rows("general_train1000.jsonl", general)
    replay_pool = read_rows(P / "dataset/agent_math_train_3000_excl3000.jsonl")
    replay_pool = [r for r in replay_pool if norm(question(r)) not in dev_keys | {norm(question(r)) for r in tests}]
    candidates = []
    for k in (1, 2, 3):
        candidates += rng.sample([r for r in replay_pool if len(r["gt"]) == k], 64)
    rng.shuffle(candidates)
    write_rows("replay_candidates192.jsonl", candidates)
    plan = {"dev_identity": DEV_ID, "dev_general": DEV_GEN, "final_identity": FINAL_ID, "final_general": FINAL_GEN,
        "system_prompts": SYS, "seed": 202610242,
        "dev_selection": {"identity_hits_min": 16, "identity_runs": 20, "general_leaks_max": 1,
            "math_max_drop_percentage_points": 5, "language_max_delta_loss": 0.04},
        "final_target": "identity >=15/21; leaks <=2/20; math near bb_rl with paired uncertainty; corrected language loss close to bb_rl",
        "candidates_if_prompt_fails": ["LoRA lr1e-4, epochs3", "LoRA lr1e-4, epochs6", "only if both identity-weak: lr2e-4, epochs6"],
        "training": {"base": "bb_rl", "rank": 16, "identity_unique": len(identity), "identity_repeats": 3,
            "general": 1000, "self_replay_target": 128, "self_replay_repeats": 2, "batch_size": 8, "max_seq_len": 768},
        "identity_protocol": "Same prompts/settings as historical final test, but new per-question seeds (700000 + 100*i+s); rerun all comparators together. Historical counts are not treated as paired.",
        "dev_math_excludes_previous_training": True,
        "sha256": {f.name: c.digest(f) for f in OUT.glob("*.jsonl")}}
    save("plan.json", plan)
    print(json.dumps({"identity_train": len(identity), "general_train": len(general), "dev_math": len(dev_math), "dev_language": len(dev_language)}, ensure_ascii=False), flush=True)


def identity(spec, final=False):
    path = OUT / ("identity_final.json" if final else "identity_dev.json")
    result = c.read(path) if path.exists() else {}
    if spec in result:
        return result[spec]
    model = load(spec)
    ids, gens, nr = (FINAL_ID, FINAL_GEN, 3) if final else (DEV_ID, DEV_GEN, 2)
    rows = []
    for group, questions, reps, offset in [("identity", ids, nr, 0), ("general", gens, 2, 100)]:
        for i, q in enumerate(questions):
            text = TOK.apply_chat_template(add_system([{"role": "user", "content": q}], SYS.get(spec)),
                tokenize=False, add_generation_prompt=True, open_thinking=False)
            enc = TOK(text, return_tensors="pt").to("cuda")
            for sample in range(reps):
                seed = (700000 if final else 800000) + 100*(offset+i) + sample
                random.seed(seed); torch.manual_seed(seed)
                with torch.no_grad():
                    out = model.generate(inputs=enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=80,
                        do_sample=True, top_p=0.95, temperature=0.85, pad_token_id=TOK.pad_token_id, eos_token_id=TOK.eos_token_id)
                reply = TOK.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
                rows.append({"group": group, "question": i, "prompt": q, "sample": sample, "seed": seed, "response": reply,
                    "name_hit": "yuyang" in reply.casefold(), "identity_mentioned": any(t in reply.casefold() for t in ("yuyang", "minimind"))})
    info = {"rows": rows, "identity_hits": sum(r["name_hit"] for r in rows if r["group"] == "identity"),
        "identity_runs": len(ids)*nr, "general_leaks": sum(r["identity_mentioned"] for r in rows if r["group"] == "general"), "general_runs": len(gens)*2}
    result[spec] = info
    save(path.name, result)
    del model; c.release()
    print(spec, {k: v for k,v in info.items() if k != "rows"}, flush=True)
    return info


def language_eval(spec, final=False):
    path = OUT / ("loss_final.json" if final else "loss_dev.json")
    result = c.read(path) if path.exists() else {}
    if spec in result:
        return result[spec]
    rows = read_rows(P / "dataset/dpo_eval300.jsonl" if final else OUT / "dev_language100.jsonl")
    model = load(spec)
    values, indices = [], []
    with torch.no_grad():
        for i, row in enumerate(rows):
            messages = add_system(row["chosen"][:-1], SYS.get(spec))
            p = TOK(TOK.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
            a = TOK(row["chosen"][-1]["content"], add_special_tokens=False).input_ids
            ids = (p+a)[:768]
            if len(p) >= len(ids):
                continue
            x = torch.tensor([ids], device="cuda")
            lp = c.F.log_softmax(model(x).logits.float()[0, len(p)-1:-1], -1)
            values.append(-lp.gather(1, x[0, len(p):, None]).mean().item())
            indices.append(i)
    result[spec] = {"indices": indices, "losses": values, "mean": st.mean(values)}
    save(path.name, result)
    print(spec, "loss", st.mean(values), "n", len(values), flush=True)
    del model; c.release()
    return result[spec]


def math_eval(spec, final=False):
    path = OUT / ("math_final.json" if final else "math_dev.json")
    result = c.read(path) if path.exists() else {}
    ds = c.AgentRLDataset(str(P / "dataset/agent_math_eval200.jsonl" if final else OUT / "dev_math24.jsonl"), TOK)
    rows = result.setdefault(spec, [])
    if len(rows) == 2*len(ds):
        return rows
    model = load(spec, half=False)
    engine = c.engine_for(model)
    started = time.perf_counter()
    for i in range(len(rows)//2, len(ds)):
        problem = copy.deepcopy(ds[i])
        problem["messages"] = add_system(problem["messages"], SYS.get(spec))
        rows.extend(c.math_row(engine, problem, i, sample) for sample in range(2))
        save(path.name, result)
        if (i+1) % 6 == 0 or i+1 == len(ds):
            print(f"{spec} math {i+1}/{len(ds)}: {100*st.mean(r['all'] for r in rows):.2f}% ({time.perf_counter()-started:.0f}s)", flush=True)
    del engine, model; c.release()
    return rows


def prompt_stage():
    for spec in ["bb_rl", "bb_rl_id", *SYS]:
        identity(spec)
        language_eval(spec)
    for spec in ["bb_rl", "bb_rl_id"]:
        math_eval(spec)
    scores = c.read(OUT / "identity_dev.json")
    for spec in SYS:
        if scores[spec]["identity_hits"] >= 16 and scores[spec]["general_leaks"] <= 1:
            math_eval(spec)


def checked_trace(source, turns):
    from trainer.train_agent import execute_tool
    from trainer.trainer_utils import safe_math_eval
    if not turns or c.parse_tool_calls(turns[-1]):
        return None, "unfinished"
    final = turns[-1].split("</think>", 1)[-1].strip()
    # Chinese characters and sentence-ending periods are valid numeric boundaries.
    nums = [float(x) for x in re.findall(r"(?<![0-9A-Za-z_.])[-+]?\d+(?:\.\d+)?(?![0-9A-Za-z_]|\.\d)", final.replace(",", ""))]
    if not all(any(abs(float(g)-v) < 1e-6 for v in nums) for g in source["gt"]):
        return None, "wrong_final_numbers"
    msgs = copy.deepcopy(source["conversations"][:-1])
    calculated = []
    for text in turns:
        msgs.append({"role": "assistant", "content": text})
        for call in c.parse_tool_calls(text):
            if call.get("name") != "calculate_math":
                return None, "wrong_tool"
            args = call.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            try:
                calculated.append(float(safe_math_eval(str(args.get("expression", "")))))
            except Exception:
                return None, "invalid_expression"
            result = execute_tool(call["name"], args)
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]
            msgs.append({"role": "tool", "content": result_str})
    if not all(any(abs(float(g)-v) < 1e-6 for v in calculated) for g in source["gt"]):
        return None, "wrong_calculator_results"
    tools = json.loads(source["conversations"][0]["tools"])
    text = TOK.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False)
    if len(TOK(text, add_special_tokens=False).input_ids) > 768:
        return None, "trajectory_exceeds_768_tokens"
    return {"conversations": msgs}, "accepted"


def replay():
    path = OUT / "replay_raw.json"
    result = c.read(path) if path.exists() else {"rows": []}
    candidates = read_rows(OUT / "replay_candidates192.jsonl")
    ds = c.AgentRLDataset(str(OUT / "replay_candidates192.jsonl"), TOK)
    for row in result["rows"]:
        row["sft"], row["reason"] = checked_trace(candidates[row["index"]], row["turns"])
        row["accepted"] = row["sft"] is not None
    accepted = sum(r["accepted"] for r in result["rows"])
    save("replay_raw.json", result)
    print(f"Validated cached trajectories: {accepted}/{len(result['rows'])}", flush=True)
    model = load("bb_rl", half=False)
    engine = c.engine_for(model)
    t0 = time.perf_counter()
    for i in range(len(result["rows"]), len(ds)):
        if accepted >= 128:
            break
        p = ds[i]
        msgs = copy.deepcopy(p["messages"])
        random.seed(900000+i); torch.manual_seed(900000+i)
        *_, turns, unfinished = c.rollout_single(engine, TOK, msgs, p["tools"], max_turns=3, max_new_tokens=384, thinking_ratio=0.0, device="cuda")
        conv, reason = checked_trace(candidates[i], turns)
        ok = conv is not None
        accepted += int(ok)
        result["rows"].append({"index": i, "accepted": ok, "gt": p["gt"], "turns": turns, "sft": conv, "reason": reason})
        save("replay_raw.json", result)
        if (i+1) % 8 == 0:
            print(f"Self replay {i+1}/192, accepted {accepted}/128, {time.perf_counter()-t0:.0f}s", flush=True)
    del engine, model; c.release()
    samples = [r["sft"] for r in result["rows"] if r["accepted"]]
    assert len(samples) >= 96, f"Only {len(samples)} usable trajectories; inspect before training"
    write_rows("self_replay.jsonl", samples)
    mixed = read_rows(OUT / "identity_train.jsonl")*3 + read_rows(OUT / "general_train1000.jsonl") + samples*2
    random.Random(202610243).shuffle(mixed)
    write_rows("gentle_mix.jsonl", mixed)
    print(f"Replay {len(samples)}, mixed training {len(mixed)}; SHA256 {c.digest(OUT / 'gentle_mix.jsonl')}", flush=True)


def select():
    assert not any(OUT.glob("*_final.json")), "Final evaluation already started; do not reselect on final outcomes"
    ids, losses, maths = [c.read(OUT / n) for n in ("identity_dev.json", "loss_dev.json", "math_dev.json")]
    candidates = []
    assert len(maths["bb_rl"]) == 48 and len(losses["bb_rl"]["losses"]) == 100
    for spec in ids:
        if spec in ("bb_rl", "bb_rl_id") or spec not in maths:
            continue
        assert len(maths[spec]) == 48, f"Development math incomplete: {spec}"
        assert ids[spec]["identity_runs"] == 20 and ids[spec]["general_runs"] == 20
        assert losses[spec]["indices"] == losses["bb_rl"]["indices"]
        delta_loss = losses[spec]["mean"] - losses["bb_rl"]["mean"]
        delta_math = 100*(st.mean(r["all"] for r in maths[spec]) - st.mean(r["all"] for r in maths["bb_rl"]))
        eligible = ids[spec]["identity_hits"] >= 16 and ids[spec]["general_leaks"] <= 1 and delta_loss <= 0.04 and delta_math >= -5
        candidates.append({"spec": spec, "eligible": eligible, "identity_hits": ids[spec]["identity_hits"],
            "general_leaks": ids[spec]["general_leaks"], "delta_loss": delta_loss, "delta_math_points": delta_math})
    eligible = [r for r in candidates if r["eligible"]]
    eligible.sort(key=lambda r: (-r["delta_math_points"], r["delta_loss"], -r["identity_hits"], r["general_leaks"]))
    result = {"candidates": candidates, "selected": eligible[0]["spec"] if eligible else None,
              "selection_uses_only_development_sets": True,
              "final_result_files_present_at_selection": [],
              "candidate_weight_sha256": {r["spec"]: c.digest(P / f"out/{r['spec']}_768.pth") for r in candidates if r["spec"] not in SYS},
              "final_acceptance": {"identity_min": 15, "identity_runs": 21, "leaks_max": 2, "general_runs": 20,
                  "math_max_drop_points": 5, "language_max_delta_loss": 0.04,
                  "note": "Practical targets, not proof of statistical equivalence; report paired uncertainties."}}
    result["baseline_weight_sha256"] = {n: c.digest(P / f"out/{n}_768.pth") for n in ("bb_rl", "bb_rl_id")}
    result["environment"] = {"python": sys.version, "torch": torch.__version__,
        "transformers": c.transformers.__version__, "gpu": torch.cuda.get_device_name(0)}
    result["evaluation_file_sha256"] = {str(f.relative_to(P)): c.digest(f) for f in [
        P / "dataset/agent_math_eval200.jsonl", P / "dataset/dpo_eval300.jsonl",
        P / "my_eval/scripts/identity_repair.py", P / "my_eval/scripts/official_compare.py",
        P / "my_eval/results/agent_bbrl200.json", P / "my_eval/results/agent_bbrl_id200.json",
        P / "my_eval/results/rm_demo_compare.json"]}
    if result["selected"] and result["selected"] not in SYS:
        adapter = result["selected"].replace("bb_rl_id_", "lora_yuyang_", 1)
        result["selected_training"] = c.read(OUT / f"train_{adapter}.json")
    save("selection.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def audit():
    mixed = read_rows(OUT / "gentle_mix.jsonl")
    train_keys = {norm(question(r)) for r in mixed}
    protected = {"dev_math": read_rows(OUT / "dev_math24.jsonl"),
                 "final_math": read_rows(P / "dataset/agent_math_eval200.jsonl"),
                 "dev_language": read_rows(OUT / "dev_language100.jsonl"),
                 "final_language": read_rows(P / "dataset/dpo_eval300.jsonl")}
    intersections = {k: len(train_keys & {norm(question(r)) for r in rows}) for k, rows in protected.items()}
    intersections["identity_and_general_questions"] = len(train_keys & {norm(q) for q in DEV_ID + DEV_GEN + FINAL_ID + FINAL_GEN})
    assert not any(intersections.values()), intersections
    samples = read_rows(OUT / "self_replay.jsonl")
    raw = c.read(OUT / "replay_raw.json")["rows"]
    candidates = read_rows(OUT / "replay_candidates192.jsonl")
    for r in raw:
        if r["accepted"]:
            reconstructed, reason = checked_trace(candidates[r["index"]], r["turns"])
            assert reconstructed == r["sft"] and reason == "accepted"
    source = copy.deepcopy(candidates[0])
    source["gt"] = ["9"]
    call = '<tool_call>{"name":"calculate_math","arguments":{"expression":"3**2"}}</tool_call>'
    for answer, expected in [("The answer is 9.", True), ("计算结果是9。", True), ("答案是19。", False)]:
        item, _ = checked_trace(source, [call, answer])
        assert (item is not None) == expected, answer
    text = TOK.apply_chat_template(add_system([{"role": "user", "content": DEV_ID[0]}], SYS["prompt_short"]),
        tokenize=False, add_generation_prompt=True, open_thinking=False)
    assert SYS["prompt_short"] in text
    record = {"overlap_counts_by_normalized_user_conversation": intersections, "training_rows": len(mixed),
        "unique_self_replay": len(samples), "replay_strict_checks_passed": True, "numeric_boundary_cases_passed": True,
        "system_prompt_present_in_rendered_input": True, "training_sha256": c.digest(OUT / "gentle_mix.jsonl")}
    save("data_audit.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


def final_chat(spec):
    assert spec not in SYS, "Prompted deployment needs a separately labelled chat protocol"
    from trainer.trainer_utils import LMForRewardModel
    convs = c.prompts()
    path = OUT / "chat_final.json"
    result = c.read(path) if path.exists() else {"prompts": convs, "results": {}}
    assert result["prompts"] == convs
    rows = result["results"].setdefault(spec, [])
    if len(rows) == 90:
        return
    rm = LMForRewardModel("D:/minimind/internlm2-1_8b-reward", device="cuda", dtype=torch.float16)
    model = load(spec)
    started = time.perf_counter()
    for i in range(len(rows)//3, len(convs)):
        messages = [{"role": m["role"], "content": m["content"]} for m in convs[i]]
        for text in c.gen_chat(model, convs[i], i):
            rows.append({"prompt": i, "response": text, **c.score(text, messages, rm)})
        save("chat_final.json", result)
        print(f"Final chat {spec} {i+1}/30; {time.perf_counter()-started:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "prompt", "replay", "audit", "dev", "select", "final_identity", "final_loss", "final_math", "final_chat"])
    parser.add_argument("--spec", default="bb_rl")
    args = parser.parse_args()
    if args.stage == "prepare": prepare()
    elif args.stage == "prompt": prompt_stage()
    elif args.stage == "replay": replay()
    elif args.stage == "audit": audit()
    elif args.stage == "dev":
        identity(args.spec); language_eval(args.spec); math_eval(args.spec)
    elif args.stage == "select": select()
    else:
        choice = c.read(OUT / "selection.json")["selected"]
        assert choice, "Choose candidate using development results before final evaluation"
        assert args.spec in ("bb_rl", "bb_rl_id", choice), "Only the locked candidate and baselines may enter final evaluation"
        locked = c.read(OUT / "selection.json")
        expected = {**locked["baseline_weight_sha256"], **locked["candidate_weight_sha256"]}
        assert c.digest(P / f"out/{args.spec}_768.pth") == expected[args.spec], "Selected weights changed"
        for filename, digest in locked["evaluation_file_sha256"].items():
            assert c.digest(P / filename) == digest, f"Frozen evaluation input/code changed: {filename}"
        if args.stage == "final_chat":
            final_chat(args.spec)
        else:
            {"final_identity": identity, "final_loss": language_eval, "final_math": math_eval}[args.stage](args.spec, final=True)
