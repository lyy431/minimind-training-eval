"""Compare the local official MiniMind-3 with the existing experiment models.

Uses the historical evaluation settings, validates conversion and baseline replay,
and saves per-question results incrementally. No training or network is needed.
Usage: python official_compare.py prepare|smoke|loss|chat|math|report
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics as st
import sys
import time
import warnings

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "my_eval/results/official_compare_20261024"
OUT.mkdir(parents=True, exist_ok=True)
RUNTIME = PROJECT / "my_eval/runtime"
RUNTIME.mkdir(exist_ok=True)
os.environ.update(HF_HOME="D:/conda/hf-cache", PYTHONIOENCODING="utf-8",
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false", TEMP=str(RUNTIME), TMP=str(RUNTIME))
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT / "trainer")
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.rollout_engine import create_rollout_engine
from trainer.train_agent import rollout_single, parse_tool_calls, validate_gt_in_text
from dataset.lm_dataset import AgentRLDataset

NAMES = ["full_sft", "official", "bb_rl", "bb_rl_id"]
TOK = AutoTokenizer.from_pretrained(PROJECT / "model", local_files_only=True)
OLD_MATH = {"full_sft": "agent_base200.json", "bb_rl": "agent_bbrl200.json",
            "bb_rl_id": "agent_bbrl_id200.json"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(name, half=False):
    m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    m.load_state_dict(torch.load(PROJECT / f"out/{name}_768.pth", map_location="cpu", weights_only=True), strict=True)
    m = m.cuda().eval()
    return m.half() if half else m


def release():
    gc.collect()
    torch.cuda.empty_cache()


def prompts():
    # Exactly score_rm.py's content exclusion, seed, and order.
    lines = [line.rstrip("\n") + "\n" for line in
             (PROJECT / "dataset/rlaif.jsonl").open(encoding="utf-8") if line.strip()]
    trained = set()
    for k in (1000, 5000):
        random.seed(42)
        trained |= set(random.sample(lines, k))
    held = [line for line in lines if line not in trained]
    return [json.loads(line)["conversations"][:-1] for line in random.Random(7).sample(held, 30)]


def gen_chat(model, conv, index):
    text = TOK.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, open_thinking=True)
    ids = TOK(text, return_tensors="pt", add_special_tokens=False).input_ids[:, -768:].cuda()
    torch.manual_seed(1000 + index)
    with torch.no_grad():
        generated = model.generate(input_ids=ids.repeat(3, 1), attention_mask=torch.ones_like(ids).repeat(3, 1),
            max_new_tokens=512, do_sample=True, temperature=0.8,
            pad_token_id=TOK.pad_token_id, eos_token_id=TOK.eos_token_id)
    return TOK.batch_decode(generated[:, ids.shape[1]:], skip_special_tokens=True)


def engine_for(model):
    return create_rollout_engine(engine_type="torch", policy_model=model, tokenizer=TOK,
        device="cuda", autocast_ctx=torch.cuda.amp.autocast(dtype=torch.bfloat16))


def math_row(engine, problem, pi, sample):
    # Same rollout, extraction, grading, precision, and RNG as eval_agent_math.py.
    random.seed(10000 * pi + sample)
    torch.manual_seed(10000 * pi + sample)
    msgs = [dict(m) for m in problem["messages"]]
    *_, turns, unfinished = rollout_single(engine, TOK, msgs, problem["tools"],
        max_turns=3, max_new_tokens=384, thinking_ratio=0.0, device="cuda")
    answers = [t.split("</think>", 1)[-1].strip() for t in turns]
    calls = [c for a in answers for c in parse_tool_calls(a)]
    last = answers[-1] if answers else ""
    final = "" if unfinished else (last.split("</tool_call>")[-1] if "</tool_call>" in last else last)
    hit = validate_gt_in_text(final, problem["gt"])
    return {"problem": pi, "gt": problem["gt"], "n_calls": len(calls), "unfinished": unfinished,
        "frac": len(hit) / len(problem["gt"]), "all": len(hit) == len(problem["gt"]),
        "calc_calls": sum(c.get("name") == "calculate_math" for c in calls), "turns": turns}


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return min(cap, (len(grams)-len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def score(response, messages, rm):
    # Exactly score_rm.py, including scoring only the answer with the reward model.
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


def prepare():
    src = PROJECT / "minimind-3"
    cfg = read(src / "config.json")
    ours = MiniMindConfig().to_dict()
    fields = ["hidden_size", "num_hidden_layers", "vocab_size", "intermediate_size", "num_attention_heads",
              "num_key_value_heads", "head_dim", "rms_norm_eps", "rope_theta", "tie_word_embeddings"]
    assert all(cfg[k] == ours[k] for k in fields), "Architecture mismatch"
    ref_tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
    assert TOK.get_vocab() == ref_tok.get_vocab()
    assert (TOK.bos_token_id, TOK.eos_token_id, TOK.pad_token_id) == (ref_tok.bos_token_id, ref_tok.eos_token_id, ref_tok.pad_token_id)
    ds = AgentRLDataset(str(PROJECT / "dataset/agent_math_eval200.jsonl"), TOK)
    examples = [([{"role": "user", "content": "你是谁？请解释为什么天空看起来是蓝色。"}], None),
                ([{"role": "user", "content": "Please summarize: cats sleep. Dogs run."}], None),
                (ds[0]["messages"], ds[0]["tools"])]
    rendered = []
    for conv, toolset in examples:
        for think in (False, True):
            kw = dict(tokenize=False, add_generation_prompt=True, open_thinking=think, tools=toolset)
            a, b = TOK.apply_chat_template(conv, **kw), ref_tok.apply_chat_template(conv, **kw)
            assert a == b, "Chat template mismatch"
            assert TOK(a).input_ids == ref_tok(b).input_ids
            rendered.append(a)
    print("Architecture, vocabulary, special tokens, and six rendered templates match.", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32,
        local_files_only=True, trust_remote_code=False).cuda().eval()
    state = {k: v.detach().cpu().half() for k, v in hf.state_dict().items()}
    native = MiniMindForCausalLM(MiniMindConfig())
    native.load_state_dict(state, strict=True)
    native = native.cuda().eval()
    checks = []
    for text in rendered:
        ids = TOK(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        with torch.no_grad():
            a, b = hf(ids).logits.float(), native(ids).logits.float()
        max_diff = (a-b).abs().max().item()
        checks.append({"tokens": ids.shape[1], "max_logit_diff": max_diff,
                       "top1_fraction": (a.argmax(-1) == b.argmax(-1)).float().mean().item()})
        assert max_diff < 0.02, checks[-1]
    ids = TOK(rendered[0], return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        a = hf.generate(input_ids=ids, max_new_tokens=24, do_sample=False, eos_token_id=TOK.eos_token_id,
                        pad_token_id=TOK.pad_token_id)
        b = native.generate(input_ids=ids, max_new_tokens=24, do_sample=False, temperature=1.0, top_p=1.0, top_k=0)
    assert torch.equal(a, b), "Cached greedy generation differs"
    target = PROJECT / "out/official_768.pth"
    if target.exists():
        old = torch.load(target, map_location="cpu", weights_only=True)
        assert old.keys() == state.keys() and all(torch.equal(old[k], state[k]) for k in state), "Existing official weight differs"
    else:
        torch.save(state, target)
    dest = PROJECT / "scripts/MiniMind-Official"
    dest.mkdir(exist_ok=True)
    copied = []
    for name in ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"]:
        a, b = src / name, dest / name
        if not a.exists():
            continue
        if b.exists():
            assert digest(a) == digest(b), f"Existing demo file differs: {b}"
        else:
            shutil.copy2(a, b)
        assert digest(a) == digest(b)
        copied.append(name)
    files = [src / "model.safetensors", PROJECT / "model/tokenizer.json",
        PROJECT / "dataset/agent_math_eval200.jsonl", PROJECT / "dataset/dpo_eval300.jsonl",
        *[PROJECT / f"out/{n}_768.pth" for n in NAMES],
        *[PROJECT / f"my_eval/scripts/{s}.py" for s in ("eval_agent_math", "score_rm", "heldout_loss", "official_compare")]]
    meta = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(), "architecture": {k: cfg[k] for k in fields},
        "conversion_checks": checks, "cached_greedy_24_tokens_identical": True,
        "sha256": {str(p.relative_to(PROJECT)): digest(p) for p in files}, "demo_files": copied,
        "scope": "Comparison of the local official artifact; official training-data overlap is unknown.",
        "protocol": {"math": "200 x 2, seed=10000*i+s, T=.8, top_p=.85, top_k=50, fp32+bf16 autocast, 384 tokens, 3 turns, thinking off",
            "chat": "30 x 3 in one batch, seed=1000+i, T=.8, top_p=.85, top_k=50, fp16, 512 tokens, thinking on",
            "loss": "300 chosen answers; per-answer mean token NLL then mean across answers; truncate full input to 768; fp16"}}
    save("manifest.json", meta)
    print(json.dumps({"conversion_checks": checks, "demo": str(dest)}, ensure_ascii=False), flush=True)


def smoke():
    convs = prompts()
    old_chat = read(PROJECT / "my_eval/results/rm_demo_compare.json")
    assert convs == old_chat["prompts"], "Historical chat prompt order differs"
    model = load("full_sft", half=True)
    started = time.perf_counter()
    chat_checks = []
    for i in (0, 1):
        responses = gen_chat(model, convs[i], i)
        old = [r["response"] for r in old_chat["results"]["full_sft"] if r["prompt"] == i]
        chat_checks.append({"prompt": i, "identical": responses == old})
    chat_secs = time.perf_counter() - started
    del model
    release()
    ds = AgentRLDataset(str(PROJECT / "dataset/agent_math_eval200.jsonl"), TOK)
    old_math = read(PROJECT / "my_eval/results/agent_base200.json")["full_sft"]
    model = load("full_sft")
    engine = engine_for(model)
    started = time.perf_counter()
    math_checks = []
    for i in (0, 50, 150):
        new = [math_row(engine, ds[i], i, s) for s in range(2)]
        old = [r for r in old_math if r["problem"] == i]
        math_checks.append({"problem": i, "identical": new == old})
        print(f"Baseline math replay {i}: {new == old}", flush=True)
    result = {"chat_checks": chat_checks, "math_checks": math_checks,
        "chat_seconds_for_2_prompts": chat_secs, "math_seconds_for_3_problems": time.perf_counter()-started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "reuse_historical": all(x["identical"] for x in chat_checks + math_checks)}
    save("smoke.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    assert result["reuse_historical"], "Replay mismatch: do not reuse history without investigating"


def loss_eval():
    rows = [json.loads(l) for l in (PROJECT / "dataset/dpo_eval300.jsonl").open(encoding="utf-8")]
    texts = []
    truncated = []
    for i, r in enumerate(rows):
        p = TOK(TOK.apply_chat_template(r["chosen"][:-1], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
        a = TOK(r["chosen"][-1]["content"], add_special_tokens=False).input_ids
        ids = (p+a)[:768]
        if len(p) >= len(ids):
            truncated.append({"row": i, "prompt_tokens": len(p), "answer_tokens_before_truncation": len(a)})
        texts.append((ids, min(len(p), len(ids)-1)))
    excluded = {r["row"] for r in truncated}
    valid = [i for i in range(len(texts)) if i not in excluded]
    result = {"protocol": "Legacy: identical to heldout_loss.py. Corrected: exclude rows with no answer token surviving the same 768-token truncation. Both average per-answer means.",
              "n_answers": len(texts), "valid_answer_indices": valid, "fully_truncated_answers": truncated,
              "answer_tokens": [len(ids)-start if i not in excluded else 0 for i, (ids, start) in enumerate(texts)],
              "results": {}, "answer_only_results": {}}
    print(f"Loss audit: {len(valid)} valid answers; {len(truncated)} fully truncated answers, indices {sorted(excluded)}", flush=True)
    for name in NAMES:
        m = load(name, half=True)
        losses = []
        with torch.no_grad():
            for ids, start in texts:
                x = torch.tensor([ids], device="cuda")
                lp = F.log_softmax(m(x).logits.float()[0, start-1:-1], -1)
                losses.append(-lp.gather(1, x[0, start:, None]).mean().item())
        result["results"][name] = losses
        result["answer_only_results"][name] = [losses[i] for i in valid]
        save("loss.json", result)
        print(f"{name}: legacy loss={st.mean(losses):.6f}, corrected answer-only loss={st.mean(result['answer_only_results'][name]):.6f}", flush=True)
        del m
        release()


def chat_eval():
    assert read(OUT / "smoke.json")["reuse_historical"]
    from trainer.trainer_utils import LMForRewardModel
    convs = prompts()
    path = OUT / "chat.json"
    result = read(path) if path.exists() else {"prompts": convs, "results": {"official": []}}
    assert result["prompts"] == convs
    rm = LMForRewardModel("D:/minimind/internlm2-1_8b-reward", device="cuda", dtype=torch.float16)
    m = load("official", half=True)
    rows = result["results"]["official"]
    assert len(rows) % 3 == 0
    t0 = time.perf_counter()
    start_i = len(rows) // 3
    for i in range(start_i, len(convs)):
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in convs[i]]
        for resp in gen_chat(m, convs[i], i):
            rows.append({"prompt": i, "response": resp, **score(resp, messages, rm)})
        result["complete"] = i == len(convs)-1
        result["seconds_this_run"] = time.perf_counter()-t0
        result["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        save("chat.json", result)
        print(f"Chat {i+1}/30; elapsed {time.perf_counter()-t0:.0f}s; mean RM {st.mean(r['rm'] for r in rows):.3f}; GPU peak {result['peak_gpu_allocated_gib']:.2f} GiB", flush=True)


def math_eval():
    assert read(OUT / "smoke.json")["reuse_historical"]
    ds = AgentRLDataset(str(PROJECT / "dataset/agent_math_eval200.jsonl"), TOK)
    path = OUT / "math.json"
    result = read(path) if path.exists() else {"official": []}
    rows = result["official"]
    assert len(rows) % 2 == 0
    m = load("official")
    engine = engine_for(m)
    start_i = len(rows) // 2
    t0 = time.perf_counter()
    for i in range(start_i, len(ds)):
        rows.extend(math_row(engine, ds[i], i, s) for s in range(2))
        save("math.json", result)
        if (i+1) % 5 == 0 or i == start_i:
            print(f"Math {i+1}/{len(ds)}; elapsed {time.perf_counter()-t0:.0f}s; accuracy {100*st.mean(r['all'] for r in rows):.2f}%", flush=True)


def paired(a, b, higher=True):
    assert len(a) == len(b)
    d = [x-y for x, y in zip(a, b)]
    sign = 1 if higher else -1
    return {"mean_difference": st.mean(d), "se": st.stdev(d)/len(d)**0.5,
            "better": sum(sign*x > 0 for x in d), "worse": sum(sign*x < 0 for x in d),
            "same": sum(x == 0 for x in d), "n": len(d)}


def report():
    loss_data = read(OUT / "loss.json")
    loss = loss_data["results"]
    corrected = loss_data["answer_only_results"]
    old = read(PROJECT / "my_eval/results/rm_demo_compare.json")
    new = read(OUT / "chat.json")
    assert new["complete"] and old["prompts"] == new["prompts"]
    chats = {n: (new if n == "official" else old)["results"][n] for n in NAMES}
    maths = {n: (read(OUT / "math.json") if n == "official" else
                 read(PROJECT / "my_eval/results" / OLD_MATH[n]))[n] for n in NAMES}
    result = {"models": {}, "official_minus": {}, "historical_results_reused": [n for n in NAMES if n != "official"],
              "loss_audit": {"valid_answer_count": len(loss_data["valid_answer_indices"]), "fully_truncated_answers": loss_data["fully_truncated_answers"]}}
    for name in NAMES:
        assert len(chats[name]) == 90 and len(maths[name]) == 400 and len(loss[name]) == 300
        assert [r["prompt"] for r in chats[name]] == [i for i in range(30) for _ in range(3)]
        assert [r["problem"] for r in maths[name]] == [i for i in range(200) for _ in range(2)]
        assert [r["gt"] for r in maths[name]] == [r["gt"] for r in maths["full_sft"]]
        result["models"][name] = {"loss": st.mean(loss[name]), "perplexity": math.exp(st.mean(loss[name])),
            "answer_only_loss": st.mean(corrected[name]), "answer_only_perplexity": math.exp(st.mean(corrected[name])),
            "math_percent": 100*st.mean(r["all"] for r in maths[name]),
            "math_by_expressions": {str(k): 100*st.mean(r["all"] for r in maths[name] if len(r["gt"]) == k) for k in (1, 2, 3)},
            **{k: st.mean(r[k] for r in chats[name]) for k in ("rm", "total", "answer_chars", "rep", "len", "think", "close")}}
        if name != "official":
            get_chat = lambda n, key: [st.mean(r[key] for r in chats[n] if r["prompt"] == i) for i in range(30)]
            get_math = lambda n: [100*st.mean(r["all"] for r in maths[n] if r["problem"] == i) for i in range(200)]
            result["official_minus"][name] = {"loss": paired(loss["official"], loss[name], higher=False),
                "answer_only_loss": paired(corrected["official"], corrected[name], higher=False),
                "math_percentage_points": paired(get_math("official"), get_math(name)),
                **{key: paired(get_chat("official", key), get_chat(name, key)) for key in ("rm", "total")}}
    save("summary.json", result)
    manifest = read(OUT / "manifest.json")
    manifest["final_runner_sha256"] = digest(Path(__file__))
    manifest["corrected_heldout_loss_script_sha256"] = digest(PROJECT / "my_eval/scripts/heldout_loss.py")
    manifest["historical_result_sha256"] = {name: digest(PROJECT / "my_eval/results" / name)
        for name in ["rm_demo_compare.json", *OLD_MATH.values()]}
    manifest["loss_audit"] = result["loss_audit"]
    manifest["official_local_metadata"] = (PROJECT / "minimind-3/.mv").read_text(encoding="utf-8").strip()
    save("manifest.json", manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "smoke", "loss", "chat", "math", "report"])
    args = parser.parse_args()
    {"prepare": prepare, "smoke": smoke, "loss": loss_eval, "chat": chat_eval,
     "math": math_eval, "report": report}[args.stage]()
