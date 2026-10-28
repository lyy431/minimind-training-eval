"""Count identity hits / leaks for each model (sampling, same settings as eval_llm.py).
Usage: score_lora.py PROJECT [SPECS], SPECS = comma list of "weight" or "weight+lora", default
"full_sft,full_sft+lora_yuyang,full_sft+lora_yuyang_mix" (weights from out/<name>_768.pth)."""
import os
import sys
import warnings

PROJECT = sys.argv[1]
SPECS = (sys.argv[2] if len(sys.argv) > 2 else "full_sft,full_sft+lora_yuyang,full_sft+lora_yuyang_mix").split(",")
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

IDENTITY = ["你是谁？", "Who created you?", "你的作者叫什么名字？", "是哪个人把你做出来的？",
            "介绍一下你的开发者", "Who is your developer?", "你和 Jingyao Gong 是什么关系？"]
GENERAL = ["解释什么是机器学习", "推荐一些中国的美食", "如何开始学习Python？", "写一首关于秋天的短诗",
           "为什么要多喝水？", "给我讲一个笑话", "怎样才能提高睡眠质量？", "太阳系有几颗行星？",
           "帮我写一封请假条", "推荐几本值得读的书"]
ID_RUNS, GEN_RUNS = 3, 2

tokenizer = AutoTokenizer.from_pretrained("./model")


def load(spec):
    weight, _, lora = spec.partition("+")
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8))
    model.load_state_dict(torch.load(f"./out/{weight}_768.pth", map_location="cuda"), strict=True)
    model = model.to("cuda")
    if lora:
        apply_lora(model)
        load_lora(model, f"./out/{lora}_768.pth")
    return model.half().eval()


def ask(model, prompt):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, open_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    out = model.generate(inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=80,
                         do_sample=True, top_p=0.95, temperature=0.85, pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


torch.manual_seed(0)
for spec in SPECS:
    model = load(spec)
    id_hits = sum("Yuyang" in ask(model, p) for p in IDENTITY for _ in range(ID_RUNS))
    leaks = []
    for p in GENERAL:
        for _ in range(GEN_RUNS):
            reply = ask(model, p)
            if "Yuyang" in reply or "MiniMind" in reply or "minimind" in reply:
                leaks.append(f"{p} -> {reply[:60]!r}")
    total_id, total_gen = len(IDENTITY) * ID_RUNS, len(GENERAL) * GEN_RUNS
    print(f"[{spec}] identity says Yuyang: {id_hits}/{total_id} | general answers with self-intro: {len(leaks)}/{total_gen}")
    for leak in leaks:
        print("    leak:", leak.replace("\n", " "))
    del model
    torch.cuda.empty_cache()
