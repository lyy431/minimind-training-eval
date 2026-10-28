"""Locked, resumable evaluation of the actual demo candidate/selector functions."""
import os
from pathlib import Path
P = Path(__file__).resolve().parents[2]
OUT = P/'my_eval/results/best_of_n_20261026'
os.environ.update(HF_HOME='D:/conda/hf-cache', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
    PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', TEMP=str(P/'my_eval/runtime'), TMP=str(P/'my_eval/runtime'))
import argparse
import hashlib
import json
import sys
import time
import warnings
warnings.filterwarnings('ignore')
sys.path[:0] = [str(P), str(P/'scripts'), str(P/'my_eval/scripts')]
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from best_of_n import RewardSelector, generate_candidate, choose
from demo_tools import execute_tool, TOOLS


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(name, obj):
    OUT.mkdir(exist_ok=True, parents=True)
    path=OUT/name
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def prepare():
    if (OUT/'protocol.json').exists():
        verify();return
    source=P/'dataset/agent_math_eval200.jsonl'
    math_rows=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines() if x.strip()]
    tasks=[]
    for i,row in enumerate(math_rows):
        if i in (15,125,181):continue
        conv=row['conversations']
        tasks.append(dict(id=f'math_{i}',kind='math',index=i,question=conv[1]['content'],
            messages=[dict(role='user',content=conv[1]['content'])],tools=json.loads(conv[0]['tools']),gt=row['gt']))
    chat_source=P/'my_eval/results/identity_repair_20261024/chat_final.json'
    for i,conv in enumerate(read(chat_source)['prompts']):
        if i==27:continue
        tasks.append(dict(id=f'chat_{i}',kind='chat',index=i,question=conv[-1]['content'],tools=None,
            messages=[dict(role='system',content='你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。')]+conv))
    save('tasks.json',tasks)
    files=[source,chat_source,P/'scripts/best_of_n.py',P/'scripts/demo_tools.py',
           Path(__file__),P/'my_eval/scripts/strict_math_rescore.py',OUT/'tasks.json']
    files+=list((P/'scripts/MiniMind-YuyangLU-v2').glob('*'))
    files+=list((P.parent/'internlm2-1_8b-reward').glob('*'))
    save('protocol.json',dict(date='2026-10-26',model='MiniMind-YuyangLU-v2',
        n_candidates=4, comparisons=[1,2,4],baseline='first candidate, even if ineligible',
        random_reference='mean correctness over all 4 candidates (not an executable chooser)',
        selector='raw local InternLM2 reward, final answer only, no reference or tool results; earliest wins ties',
        eligibility='complete nonempty final answer; no tool/token/context limit; RM context <=2047 tokens',
        no_eligible='return first attempt and flag fallback',
        seed_formula='2026102600 + task position * 10 + candidate index',
        temperature=.9,top_p=.85,top_k=50,max_new_tokens=512,max_turns=3,thinking=False,
        math_questions=197,chat_questions=29,samples_per_question=4,
        math_metric='frozen answer_consistency_v1; unknown cases reviewed, never silently failed',
        independent_chat_metric='assistant content review with hidden candidate selection and scores; not independent human blind judging',
        inference='one fixed training checkpoint, one pool per question; paired question bootstrap10000 seed20261026; no training-seed uncertainty',
        limitations='historical heldout set reused; same RM selection gain is circular and not quality evidence; no claims about other models',
        hashes={os.path.relpath(f,P):sha(f) for f in files if f.is_file()}))
    print('Locked: 197 math + 29 chat questions; 4 candidates each.',flush=True)


def verify():
    for file,value in read(OUT/'protocol.json')['hashes'].items():
        assert sha(P/file)==value,f'Frozen input changed: {file}'


def load():
    path=P/'scripts/MiniMind-YuyangLU-v2'
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16,local_files_only=True).cuda().eval()
    return model,tok


def generate(smoke=False):
    if not smoke:verify()
    tasks=read(OUT/'tasks.json') if not smoke else [
        dict(id='smoke_math',messages=[dict(role='user',content='请计算 13*17，另外 28+46 等于多少？')],tools=[TOOLS[0],TOOLS[5]]),
        dict(id='smoke_chat',messages=[dict(role='user',content='请用一句话解释什么是蒸馏模型。')],tools=None)]
    path=OUT/('smoke_generated.json' if smoke else 'generated.json')
    rows=read(path) if path.exists() else {}
    model,tok=load()
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for pos,task in enumerate(tasks):
        pool=rows.setdefault(task['id'],[])
        for k in range(len(pool),4):
            pool.append(generate_candidate(model,tok,task['messages'],tools=task['tools'],execute_tool=execute_tool,
                seed=(123456 if smoke else 2026102600)+pos*10+k,max_new_tokens=512))
            save(path.name,rows)
        print(f"{pos+1}/{len(tasks)} {task['id']} {sum(c['generated_tokens'] for c in pool)} tokens {time.perf_counter()-started:.1f}s",flush=True)
    save('smoke_generation_runtime.json' if smoke else 'generation_runtime.json',
        dict(seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))


def score(smoke=False):
    if not smoke:verify()
    rows=read(OUT/('smoke_generated.json' if smoke else 'generated.json'))
    tasks=read(OUT/'tasks.json') if not smoke else [
        dict(id='smoke_math',messages=[dict(role='user',content='请计算 13*17，另外 28+46 等于多少？')]),
        dict(id='smoke_chat',messages=[dict(role='user',content='请用一句话解释什么是蒸馏模型。')])]
    path=OUT/('smoke_scored.json' if smoke else 'scored.json')
    scored=read(path) if path.exists() else {}
    rm=RewardSelector()
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for pos,task in enumerate(tasks):
        if task['id'] in scored:continue
        pool=rows[task['id']]
        assert len(pool)==4
        seconds=rm.score(task['messages'],pool)
        scored[task['id']]=dict(candidates=pool,selected={str(n):choose(pool,n) for n in [1,2,4]},score_seconds=seconds)
        save(path.name,scored)
        print(f"{pos+1}/{len(tasks)} {task['id']} selected={scored[task['id']]['selected']} {time.perf_counter()-started:.1f}s",flush=True)
    save('smoke_score_runtime.json' if smoke else 'score_runtime.json',
        dict(seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','generate','score','smoke_generate','smoke_score'])
    stage=parser.parse_args().stage
    if stage=='prepare':prepare()
    elif stage.endswith('generate'):generate(stage.startswith('smoke'))
    else:score(stage.startswith('smoke'))
