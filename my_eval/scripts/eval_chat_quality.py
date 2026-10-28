"""Resumable development/final comparisons with the same frozen local model."""
import argparse
import copy
import gc
import time
import chat_quality_common as e
import torch
from best_of_n import generate_candidate
from chat_quality_support import ordinary_chat
from demo_tools import execute_tool

CONDITIONS=['no_tools','six_original','six_reworded']

def tools_for(task,condition):
    return None if condition=='no_tools' else e.old.menu(6,0 if condition=='six_original' else 6,task['seed']+900)

def generate(model,tok,task,condition,policy,chat_model=None,chat_tok=None):
    tools=tools_for(task,condition);messages=copy.deepcopy(task['messages'])
    if policy=='baseline':
        if not tools:messages=e.with_system(messages,e.STANDARD)
        return generate_candidate(model,tok,messages,tools=tools,execute_tool=execute_tool,
            seed=task['seed'],temperature=.9,max_new_tokens=384,thinking=False,max_turns=3)
    return ordinary_chat(model,tok,messages,tools=tools,execute_tool=execute_tool,seed=task['seed'],
        temperature=.9,max_new_tokens=384,system=e.SYSTEMS[policy],chat_model=chat_model,chat_tokenizer=chat_tok)

def verify():
    lock=e.read(e.OUT/'final_lock.json')
    for name,h in lock['files_sha256'].items():assert e.sha(e.P/name)==h,name

def run(split,model_name,policy):
    if split=='final':verify()
    tasks=e.read(e.OUT/('dev_tasks.json' if split=='dev' else 'final_tasks.json'))
    path=e.OUT/f'{split}_{model_name}_{policy}.json';done=e.read(path) if path.exists() else {}
    model,tok=e.load(model_name);chat_model,chat_tok=model,tok
    if split=='final' and policy!='baseline':
        backend=e.read(e.OUT/'selection.json')['chat_backend']
        if backend!=model_name:chat_model,chat_tok=e.load(backend)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    conditions=CONDITIONS if split=='final' or model_name==e.MODEL else CONDITIONS[:2]
    for i,t in enumerate(tasks):
        for condition in conditions:
            # Dev prompt alternatives need only no-schema outputs; two-pass behavior
            # can be formed from these and the already generated baseline routes.
            if split=='dev' and policy!='baseline' and condition!='no_tools':continue
            key=t['id']+'_'+condition
            if key in done:continue
            c=generate(model,tok,t,condition,policy,chat_model,chat_tok)
            done[key]=dict(task_id=t['id'],condition=condition,**c);e.save(path.name,done)
        if (i+1)%4==0 or i==len(tasks)-1:print(split,model_name,policy,i+1,'/',len(tasks),f'{time.perf_counter()-started:.1f}s',flush=True)
    e.save(path.stem+'_runtime.json',dict(seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        generated_tokens=sum(x['generated_tokens'] for x in done.values()),rows=len(done)))

def main():
    p=argparse.ArgumentParser();p.add_argument('split',choices=['dev','final']);p.add_argument('--model',default=e.MODEL);p.add_argument('--policy',default='baseline',choices=['baseline','standard','concise']);a=p.parse_args()
    run(a.split,a.model,a.policy)

if __name__=='__main__':main()

