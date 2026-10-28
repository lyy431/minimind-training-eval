"""Same single-answer engine for old v3 and repaired weights; no runtime teacher."""
import argparse
import copy
import time
import statistics as st
import chat_repair_common as e
import torch
import torch.nn.functional as F
from best_of_n import generate_candidate
from demo_tools import execute_tool
from strict_math_rescore import grade

CONDITIONS=['no_tools','six_original','six_reworded']

def verify_final(name):
    lock=e.read(e.OUT/'final_lock.json');assert name in lock['weights']
    for n,h in lock['weights'].items():assert e.sha(e.P/f'out/{n}_768.pth')==h,n
    for n,h in lock['files_sha256'].items():assert e.sha(e.P/n)==h,n

def chat_tasks(split):
    expanded=[]
    for t in e.read(e.OUT/('dev_tasks.json' if split=='dev' else 'final_tasks.json')):
        for condition in CONDITIONS:
            tools=None if condition=='no_tools' else e.old.menu(6,0 if condition=='six_original' else 6,t['seed']+900)
            messages=copy.deepcopy(t['messages'])
            if tools is None:messages=[dict(role='system',content=e.SYSTEM)]+messages
            expanded.append(dict(**t,task_id=t['id']+'_'+condition,condition=condition,tools=tools,input_messages=messages,kind='chat'))
    return expanded

def math_tasks(domain):
    if domain=='devmath':source=e.read(e.old.OUT/'dev_tasks.json');return [t for t in source if t['kind']=='math']
    if domain=='historical':return e.read(e.old.OUT/'guardrail_tasks.json')
    source=e.read(e.old.OUT/'test_tasks.json');return [t for t in source if t['kind']=='math' and t['tool_count']==6 and t['variant']==6]

def identity_tasks():
    qs=['你是谁？','Who created you?','你的作者叫什么名字？','是哪个人把你做出来的？','介绍一下你的开发者','Who is your developer?','你和 Jingyao Gong 是什么关系？']
    return [dict(id=f'identity_{i}_{j}',task_id=f'identity_{i}_{j}',question=q,kind='chat',category='identity',
        reference_for_review_only=('MiniMind原始项目由jingyaogong/Jingyao Gong开发，本地定制由Yuyang LU负责' if i==6 else '按问题准确给出MiniMind或本地定制者Yuyang LU，不张冠李戴'),
        messages=[dict(role='user',content=q)],input_messages=[dict(role='system',content=e.SYSTEM),dict(role='user',content=q)],
        tools=None,seed=550000+i*10+j) for i,q in enumerate(qs) for j in [0,1]]

def generate_rows(domain,name,model,tok,tasks):
    path=e.OUT/f'{domain}_{name}.json';done=e.read(path) if path.exists() else {};started=time.perf_counter()
    for i,t in enumerate(tasks):
        k=t.get('task_id',t.get('id'))
        if k in done:continue
        messages=t.get('input_messages',[dict(role='user',content=t['question'])])
        c=generate_candidate(model,tok,messages,tools=t['tools'],execute_tool=execute_tool,seed=t['seed'],
            temperature=.9,max_new_tokens=384,thinking=False,max_turns=3)
        g=grade(c['final'],t['question'],t['gt'],unfinished=not c['eligible']) if t['kind']=='math' else None
        done[k]=dict(task_id=k,id=t.get('id'),question=t['question'],condition=t.get('condition'),kind=t['kind'],**c,grade=g)
        e.save(path.name,done)
        if (i+1)%16==0 or i==len(tasks)-1:print(domain,name,i+1,'/',len(tasks),f'{time.perf_counter()-started:.1f}s',flush=True)
    e.save(path.stem+'_runtime.json',dict(seconds=time.perf_counter()-started,rows=len(done),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))

def nll(domain,name,model,tok):
    source=e.P/'my_eval/results/identity_repair_20261024/dev_language100.jsonl' if domain=='loss_dev' else e.P/'dataset/dpo_eval300.jsonl'
    result=[]
    with torch.inference_mode():
        for i,row in enumerate(e.rows(source)):
            conv=row['chosen'];p=tok(tok.apply_chat_template(conv[:-1],tokenize=False,add_generation_prompt=True),add_special_tokens=False).input_ids
            a=tok(conv[-1]['content'],add_special_tokens=False).input_ids;ids=(p+a)[:768]
            if len(p)>=len(ids):continue
            x=torch.tensor([ids],device='cuda');logits=model(x).logits[0,len(p)-1:-1].float()
            result.append(dict(index=i,loss=F.cross_entropy(logits,x[0,len(p):]).item(),answer_tokens=len(ids)-len(p)))
    e.save(f'{domain}_{name}.json',dict(rows=result,mean=st.mean(r['loss'] for r in result),source_sha256=e.sha(source)))
    print(domain,name,len(result),st.mean(r['loss'] for r in result),flush=True)

def prepare_baselines():
    assert not (e.OUT/'final_lock.json').exists()
    mapping={'devchat':e.diagnostic.OUT/f'dev_MiniMind-YuyangLU-v3_baseline.json',
        'loss_dev':e.old.OUT/f'loss_dev_{e.BASE}.json',
        'historical':e.old.OUT/f'guard_{e.BASE}.json','freshmath':e.old.OUT/f'fullmath_{e.BASE}.json',
        'loss_final':e.old.OUT/f'loss_final_{e.BASE}.json'}
    records={}
    for domain,source in mapping.items():
        data=e.read(source)
        if domain=='devmath':data={k:r for k,r in data.items() if r['kind']=='math'}
        e.save(f'{domain}_{e.BASE}.json',data);records[domain]=dict(source=str(source.relative_to(e.P)),sha256=e.sha(source))
    export=e.read(e.old.OUT/'demo_export.json');assert export['all_parameter_tensors_exact'] and export['tokenizer_and_template_equal']
    records['equivalence']=dict(pth_base_sha=e.sha(e.P/f'out/{e.BASE}_768.pth'),hf_sha=e.sha(e.P/'scripts/MiniMind-YuyangLU-v3/model.safetensors'),exact_parameters_and_template_verified=True,
        shared_engine_sha=e.sha(e.P/'scripts/best_of_n.py'),sampler='fp16;temperature.9;top_p.85;top_k50;max384;thinkingoff;3turns;exact task seeds/prompts/menus')
    e.save('baseline_reuse.json',records);e.save('identity_tasks.json',identity_tasks());print('Frozen-v3 baselines recorded.',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['baselines','dev','final']);p.add_argument('--model',default=e.BASE);a=p.parse_args()
    if a.stage=='baselines':prepare_baselines();return
    if a.stage=='final':verify_final(a.model)
    model=e.load(a.model);tok=e.tok();torch.cuda.reset_peak_memory_stats()
    if a.stage=='dev':
        generate_rows('devchat',a.model,model,tok,chat_tasks('dev'));generate_rows('devmath',a.model,model,tok,math_tasks('devmath'));nll('loss_dev',a.model,model,tok)
    else:
        generate_rows('finalchat',a.model,model,tok,chat_tasks('final'))
        if a.model!=e.BASE:
            generate_rows('historical',a.model,model,tok,math_tasks('historical'));generate_rows('freshmath',a.model,model,tok,math_tasks('freshmath'));nll('loss_final',a.model,model,tok)
        generate_rows('identity',a.model,model,tok,identity_tasks())

if __name__=='__main__':main()

