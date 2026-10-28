"""Resumable task-5 development/final/retention evaluations; same engine for both arms."""
import argparse
import gc
import math
import statistics as st
import time
import tool_choice_round2 as e
import torch
import torch.nn.functional as F
from best_of_n import generate_candidate
from demo_tools import execute_tool
from strict_math_rescore import grade


def evaluate(stage,name):
    if stage in ['final','guard','identity','fullmath']:verify_final(name)
    if stage=='dev':tasks=e.read(e.OUT/'dev_tasks.json')
    elif stage=='final':tasks=e.read(e.OUT/'test_tasks.json')
    elif stage=='guard':tasks=e.read(e.OUT/'guardrail_tasks.json')
    elif stage=='fullmath':tasks=[t for t in e.read(e.OUT/'test_tasks.json') if t['kind']=='math' and t['tool_count']==6 and t['variant']==6]
    else:
        qs=['你是谁？','Who created you?','你的作者叫什么名字？','是哪个人把你做出来的？','介绍一下你的开发者','Who is your developer?','你和 Jingyao Gong 是什么关系？']
        tasks=[dict(task_id=f'identity_{i}_{j}',kind='identity',question=q,tools=None,seed=550000+i*10+j) for i,q in enumerate(qs) for j in [0,1]]
    path=e.OUT/f'{stage}_{name}.json';done=e.read(path) if path.exists() else {}
    tok=e.load_tok();model=e.load_model(name);started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for i,task in enumerate(tasks):
        key=task.get('task_id',task.get('id'))
        if key in done:continue
        c=generate_candidate(model,tok,[dict(role='user',content=task['question'])],tools=task['tools'],execute_tool=execute_tool,
            seed=task['seed'],temperature=.9,max_new_tokens=384,thinking=False,max_turns=3)
        route=e.first_call_metrics(c,task) if stage!='identity' else {}
        g=grade(c['final'],task['question'],task['gt'],unfinished=not c['eligible']) if task['kind']=='math' else None
        done[key]=dict(task_id=key,question=task['question'],kind=task['kind'],**c,**route,grade=g,
            name_hit='minimind' in c['final'].casefold() and 'yuyang' in c['final'].casefold())
        e.save(path.name,done)
        if (i+1)%12==0 or i==len(tasks)-1:print(stage,name,i+1,'/',len(tasks),f'{time.perf_counter()-started:.1f}s',flush=True)
    e.save(f'{stage}_{name}_runtime.json',dict(seconds=time.perf_counter()-started,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,questions=len(tasks),
        core_source_sha256=e.sha(e.P/'scripts/best_of_n.py'),eval_source_sha256=e.sha(__file__)))
    if stage!='identity':
        mathrows=[x for x in done.values() if x['kind']=='math'];other=[x for x in done.values() if x['kind']=='no_tool']
        summary=dict(math_name=100*st.mean(x['route_ok'] for x in mathrows) if mathrows else None,
            math_arguments=100*st.mean(x['arguments_ok'] for x in mathrows) if mathrows else None,
            no_tool=100*st.mean(x['route_ok'] for x in other) if other else None)
        e.save(f'{stage}_{name}_routes.json',summary);print(summary,flush=True)


def nll(stage,name):
    if stage=='loss_final':verify_final(name)
    source=e.P/'my_eval/results/identity_repair_20261024/dev_language100.jsonl' if stage=='loss_dev' else e.P/'dataset/dpo_eval300.jsonl'
    tok=e.load_tok();model=e.load_model(name);result=[]
    with torch.inference_mode():
        for i,row in enumerate(e.rows(source)):
            conv=row['chosen']
            prompt=tok.apply_chat_template(conv[:-1],tokenize=False,add_generation_prompt=True)
            p=tok(prompt,add_special_tokens=False).input_ids
            a=tok(conv[-1]['content'],add_special_tokens=False).input_ids
            ids=(p+a)[:768]
            if len(p)>=len(ids):continue
            x=torch.tensor([ids],device='cuda');logits=model(x).logits[0,len(p)-1:-1].float()
            loss=F.cross_entropy(logits,x[0,len(p):]).item()
            result.append(dict(index=i,loss=loss,answer_tokens=len(ids)-len(p)))
    e.save(f'{stage}_{name}.json',dict(rows=result,mean=st.mean(x['loss'] for x in result),source_sha256=e.sha(source)))
    print(stage,name,len(result),st.mean(x['loss'] for x in result),flush=True)


def select():
    assert not (e.OUT/'final_lock.json').exists()
    base=e.read(e.OUT/f'dev_{e.BASE}_routes.json');base_loss=e.read(e.OUT/f'loss_dev_{e.BASE}.json')['mean']
    candidates=[]
    for epoch in [1,2,3]:
        name=f'tool_choice2b_e{epoch}';r=e.read(e.OUT/f'dev_{name}_routes.json');loss=e.read(e.OUT/f'loss_dev_{name}.json')['mean']
        gates=r['no_tool']>=max(85.,base['no_tool']-5) and loss<=base_loss+.05
        candidates.append(dict(name=name,epoch=epoch,**r,loss=loss,loss_change=loss-base_loss,passes_dev_gates=gates,
            balanced=(r['math_arguments']+r['no_tool'])/2))
    allowed=[c for c in candidates if c['passes_dev_gates']]
    selected=max(allowed or candidates,key=lambda c:(min(c['math_arguments'],c['no_tool']),c['balanced'],-c['loss'],-c['epoch']))
    e.save('selection.json',dict(base=base,base_loss=base_loss,candidates=candidates,selected=selected['name'],dev_gates_passed=bool(allowed)))
    files=['plan.json','train.jsonl','dev_tasks.json','test_tasks.json','guardrail_tasks.json','token_audit.json','selection.json']
    sources=[e.P/'my_eval/scripts'/x for x in ['tool_choice_round2.py','train_tool_choice_round2b.py','eval_tool_choice_round2.py','strict_math_rescore.py']]
    sources+=[e.P/'scripts/best_of_n.py',e.P/'scripts/demo_tools.py',e.P/'my_eval/scripts/eval_routing_round2.py',e.P/'my_eval/scripts/prepare_tool_choice_round2.py',e.P/'my_eval/scripts/balance_tool_choice_round2.py']
    e.save('final_lock.json',dict(selected=selected['name'],weights={n:e.sha(e.P/f'out/{n}_768.pth') for n in [e.BASE,selected['name']]},
        data_sha256={n:e.sha(e.OUT/n) for n in files},source_sha256={str(p.relative_to(e.P)):e.sha(p) for p in sources}))
    print('Selected before final:',selected,flush=True)


def refresh_routes(name):
    tasks=e.read(e.OUT/'dev_tasks.json');done=e.read(e.OUT/f'dev_{name}.json')
    assert len(done)==len(tasks)
    for task in tasks:done[task['task_id']].update(e.first_call_metrics(done[task['task_id']],task))
    e.save(f'dev_{name}.json',done)
    ms=[r for r in done.values() if r['kind']=='math'];ns=[r for r in done.values() if r['kind']=='no_tool']
    summary=dict(math_name=100*st.mean(r['route_ok'] for r in ms),math_arguments=100*st.mean(r['arguments_ok'] for r in ms),
        no_tool=100*st.mean(r['route_ok'] for r in ns))
    e.save(f'dev_{name}_routes.json',summary);print(summary,flush=True)


def verify_final(name):
    lock=e.read(e.OUT/'final_lock.json');assert name in lock['weights']
    assert e.sha(e.P/f'out/{name}_768.pth')==lock['weights'][name]
    for n,h in lock['data_sha256'].items():assert e.sha(e.OUT/n)==h,n
    for n,h in lock['source_sha256'].items():assert e.sha(e.P/n)==h,n


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['dev','final','guard','identity','fullmath','loss_dev','loss_final','select','refresh']);p.add_argument('--model',default=e.BASE);a=p.parse_args()
    if a.stage=='select':select()
    elif a.stage=='refresh':refresh_routes(a.model)
    elif a.stage.startswith('loss'):nll(a.stage,a.model)
    else:evaluate(a.stage,a.model)
