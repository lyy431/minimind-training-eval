"""Supervised chat repair of MiniMind v3 itself, without a runtime teacher backend."""
import chat_quality_common as diagnostic
import copy
import json
from pathlib import Path
old=diagnostic.old
P=old.P
OUT=P/'my_eval/results/chat_repair_20261027'
OUT.mkdir(parents=True,exist_ok=True)
BASE='tool_choice2b_e3'
SYSTEM=diagnostic.STANDARD
sha=old.sha
read=old.read
rows=old.rows

def save(name,data):
    p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');t.replace(p)

def load(name=BASE,fp32=False):return old.load_model(name,fp32=fp32)
def tok():return old.load_tok()

