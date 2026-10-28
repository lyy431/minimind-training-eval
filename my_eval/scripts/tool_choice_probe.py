"""Can the model pick the right tool by itself? Every question gets the same full tool list; we look only at the
first reply (which tools it calls, if any). Sampling uses web_demo.py's settings (temperature 0.9, top_p 0.85).
Usage: tool_choice_probe.py PROJECT TOOLS_JSON OUT_JSON WEIGHT[,WEIGHT] [SAMPLES]"""
import json
import os
import statistics as st
import sys
import warnings

PROJECT, TOOLS_JSON, OUT_JSON = sys.argv[1:4]
WEIGHTS = sys.argv[4].split(",")
SAMPLES = int(sys.argv[5]) if len(sys.argv) > 5 else 3
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.train_agent import parse_tool_calls

TOOLS = json.load(open(TOOLS_JSON, encoding="utf-8"))
math = [json.loads(l)["conversations"][1]["content"] for l in open("dataset/agent_math_eval200.jsonl", encoding="utf-8")][:10]
CASES = {  # category -> (expected tool or None, questions)
    "calculate_math": ("calculate_math", math),
    "weather": ("get_current_weather", ["北京今天天气怎么样？", "What's the weather like in London today?", "上海现在冷吗，天气如何？"]),
    "exchange": ("get_exchange_rate", ["1美元现在能换多少人民币？", "What is the exchange rate from EUR to CNY?", "欧元兑美元的汇率是多少？"]),
    "time": ("get_current_time", ["现在几点了？", "What time is it in Tokyo now?", "告诉我现在的日期和时间"]),
    "translate": ("translate_text", ["把'今天天气真好'翻译成英文", "Translate 'I love programming' into Chinese", "请把'机器学习很有趣'翻译成英语"]),
    "unit": ("unit_converter", ["5公里等于多少英里？", "Convert 10 kg to pounds", "100华氏度是多少摄氏度？"]),
    "no tool": (None, ["你是谁？", "给我讲一个笑话", "推荐三本值得读的书", "什么是机器学习？", "写一首关于春天的短诗", "怎样才能提高睡眠质量？"]),
}
tok = AutoTokenizer.from_pretrained("./model")


def load(name):
    m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    m.load_state_dict(torch.load(f"./out/{name}_768.pth", map_location="cuda"), strict=True)
    return m.to("cuda").half().eval()


results = {}
for name in WEIGHTS:
    model = load(name)
    rows = []
    for cat, (expected, questions) in CASES.items():
        for qi, q in enumerate(questions):
            text = tok.apply_chat_template([{"role": "user", "content": q}], tools=TOOLS, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").input_ids.cuda()
            for s in range(SAMPLES):
                torch.manual_seed(1000 * qi + s)  # same seeds for every model
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=256, do_sample=True,
                                     temperature=0.9, top_p=0.85, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
                called = [c.get("name") for c in parse_tool_calls(reply)]
                ok = (not called) if expected is None else (bool(called) and all(c == expected for c in called))
                rows.append({"category": cat, "question": q, "expected": expected, "called": called, "ok": ok, "reply": reply})
    results[name] = rows
    del model
    torch.cuda.empty_cache()

json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"all {len(TOOLS)} tools listed every time; {SAMPLES} samples per question; first reply only\n")
print(f"{'category':16}{'n':>5}" + "".join(f"{w:>14}" for w in WEIGHTS) + "   (% picked the right tool / % made no call for 'no tool')")
for cat in CASES:
    n = sum(r["category"] == cat for r in results[WEIGHTS[0]])
    print(f"{cat:16}{n:>5}" + "".join(f"{100 * st.mean(r['ok'] for r in results[w] if r['category'] == cat):>14.0f}" for w in WEIGHTS))
for w in WEIGHTS:
    wrong = [r for r in results[w] if not r["ok"]]
    calls = {}
    for r in wrong:
        key = f"{r['category']} -> {','.join(r['called']) or 'no call'}"
        calls[key] = calls.get(key, 0) + 1
    print(f"\n{w} mistakes:", ", ".join(f"{k} x{v}" for k, v in sorted(calls.items(), key=lambda kv: -kv[1])))
