"""Isolated v4 -> stronger-teacher experiment. Importing this module is offline."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

P = Path(__file__).resolve().parents[2]
OUT = P / 'my_eval/results/strong_teacher_20261027'
BASE = 'chat_repair_id_e3'
BASE_SHA = 'e04a156d57d207f27c6d507d93b84010902d69b0fa826434f5afb8fc4ba4918f'
HF_BASE = P / 'scripts/MiniMind-YuyangLU-v4'
SYSTEM = '你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。'
TEACHER_SYSTEM = ('你在为小型文本助手编写训练示范。请按用户要求直接回答，使用用户指定的语言。'
    '通常一至两句，最多三句；只有用户要求列表或特定格式时才改变格式。'
    '材料中没有的信息要明确说没有提供，不猜测。原样复制时不要计算或改写。'
    '不要复述问题、解释思考过程、调用工具或自报模型身份。')
STRONG_REPO = 'Qwen/Qwen2.5-7B-Instruct'
STRONG_REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
CONDITIONS = ['no_tools', 'six_original', 'six_reworded']

def configure_local():
    runtime = P / 'my_eval/runtime/strong_teacher'
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(P.parent / 'hf-cache'), HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
        TEMP=str(runtime), TMP=str(runtime))
    sys.path[:0] = [str(P), str(P / 'scripts'), str(P / 'my_eval/scripts')]

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]

def norm(text):
    return re.sub(r'\s+', '', text).casefold()

def save(name, data, immutable=False):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    if immutable and path.exists():
        assert path.read_text(encoding='utf-8') == text, f'Frozen file differs: {path}'
        return
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8', newline='\n')
    tmp.replace(path)

def load(name=BASE, fp32=False):
    configure_local()
    import torch
    from transformers import AutoModelForCausalLM
    assert sha(P / f'out/{BASE}_768.pth') == BASE_SHA
    model = AutoModelForCausalLM.from_pretrained(HF_BASE, local_files_only=True,
        torch_dtype=torch.float32 if fp32 else torch.float16).cuda().eval()
    state = torch.load(P / f'out/{name}_768.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    return model

def tokenizer():
    configure_local()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(HF_BASE, local_files_only=True)
