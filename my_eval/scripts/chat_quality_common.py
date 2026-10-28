"""Local, frozen-model chat-quality experiment; all storage stays on D:."""
import tool_choice_round2 as old
import copy
import json
import time
from pathlib import Path

P=old.P
OUT=P/'my_eval/results/chat_quality_20261027'
OUT.mkdir(parents=True,exist_ok=True)
MODEL='MiniMind-YuyangLU-v3'
STANDARD='你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。'
CONCISE='你是MiniMind，由Yuyang LU基于jingyaogong/minimind定制的AI助手。请用用户使用的语言，直接、简短、清楚地回答。按用户的要求作答，不复述问题，不添加无关内容；不确定时如实说明。'
SYSTEMS={'standard':STANDARD,'concise':CONCISE}

def read(path):return old.read(path)
def sha(path):return old.sha(path)
def save(name,data):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');t.replace(p)

def with_system(messages,system):
    return [dict(role='system',content=system)]+copy.deepcopy([m for m in messages if m['role']!='system'])

def load(name=MODEL):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(P/'scripts'/name,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(P/'scripts'/name,local_files_only=True,torch_dtype=torch.float16).cuda().eval()
    return model,tok

def metadata():
    return {str(p.relative_to(P)):sha(p) for p in [P/'scripts'/MODEL/'model.safetensors',P/'scripts'/MODEL/'tokenizer.json',P/'scripts'/MODEL/'chat_template.jinja',P/'scripts/best_of_n.py',P/'scripts/demo_tools.py',P/'my_eval/scripts/strict_math_rescore.py']}

