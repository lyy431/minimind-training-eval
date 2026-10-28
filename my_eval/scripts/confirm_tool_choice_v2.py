"""One fixed independent confirmation after the initial24-question primary CI included0.

Frozen weights; 96 new questions; no optional stopping; more conservative97.5% CI.
Only first replies are needed for the routing endpoint; batched left-padding is validated.
"""
import argparse
import copy
import gc
import random
import statistics as st
import time
import numpy as np
import torch
import tool_choice_v2 as e

def prepare():
    assert not (e.OUT/'confirmation_plan.json').exists()
    initial=e.read(e.OUT/'summary.json');assert not initial['accepted']
    assert all(v for k,v in initial['gates'].items() if k!='exact_math_arguments_gain_ci_above0')
    lock=e.read(e.OUT/'final_lock.json');qs=set();es=set()
    for r in e.rows(e.OUT/'train.jsonl'):
        for m in r['messages']:
            if m['role']=='user':
                qs.add(e.norm(m['content']))
                for expr,_,_ in e.expressions(m['content']):es.add(e.signature(expr))
    for file in ['dev_tasks.json','test_tasks.json','guardrail_tasks.json']:
        for r in e.read(e.OUT/file):
            qs.add(e.norm(r['question']))
            for expr,_,_ in e.expressions(r['question']):es.add(e.signature(expr))
    for name in e.read(e.OUT/'plan.json')['source_sha256']:
        for r in e.rows(e.P/name):
            for expr,_,_ in e.expressions(e.question(r)):es.add(e.signature(expr))
    cases=e.make_math('test',96,random.Random(202610266),qs,es)
    tasks=e.expand(cases,'test')
    for i,t in enumerate(tasks):
        t['id']='confirm_'+t['id'];t['task_id']='confirm_'+t['task_id'];t['batch_seed']=820000+i//8
    e.save('confirmation_tasks.json',tasks)
    e.save('confirmation_plan.json',dict(reason='initial primary+9.90pp,95%CI[-4.69,+25.00] did not pass; other retention gates passed',
        n_questions=96,conditions_per_question=8,total_conditions=768,weights=lock['weights'],selected=initial['candidate'],
        no_new_training=True,no_model_selection=True,no_optional_stopping=True,
        primary='first reply calculator name and all exact expression arguments',
        pass_rule='independent paired question-cluster bootstrap97.5% percentile interval lower>0; all initial other gates must remain passed',
        ci_reason='more conservative interval for an additional confirmation opportunity; both initial failure and confirmation reported',
        inference='same HF fp16,temp0.9,top_p0.85,top_k50,384tokens,thinking off; batch8 conditions per question, left padding; same seed per batch for both models',
        seed=202610266,bootstrap_seed=20261026,bootstrap_draws=20000,
        tasks_sha256=e.sha(e.OUT/'confirmation_tasks.json'),initial_summary_sha256=e.sha(e.OUT/'summary.json'),
        source_sha256=e.sha(__file__),scope='replication of original2 held-out wording families with fresh operands and menus; not arbitrary new semantics'))
    print('Locked96 new questions/768 conditions; unchanged models;97.5% interval.',flush=True)

@torch.inference_mode()
def generate(name):
    plan=e.read(e.OUT/'confirmation_plan.json');assert e.sha(__file__)==plan['source_sha256']
    assert e.sha(e.OUT/'confirmation_tasks.json')==plan['tasks_sha256']
    assert e.sha(e.P/f'out/{name}_768.pth')==plan['weights'][name]
    tok=e.load_tok();tok.padding_side='left';model=e.load_model(name)
    def texts(tasks):return [tok.apply_chat_template([dict(role='user',content=t['question'])],tools=t['tools'],tokenize=False,add_generation_prompt=True) for t in tasks]
    # Check masking/positions on development inputs, not confirmation outputs.
    dev=[e.read(e.OUT/'dev_tasks.json')[i] for i in [0,3]];rendered=texts(dev)
    batch=tok(rendered,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
    bg=model.generate(input_ids=batch.input_ids,attention_mask=batch.attention_mask,max_new_tokens=24,do_sample=False)
    def until_eos(tokens):
        values=tokens.tolist()
        return values[:values.index(tok.eos_token_id)+1] if tok.eos_token_id in values else values
    for i,text in enumerate(rendered):
        single=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda')
        sg=model.generate(input_ids=single.input_ids,attention_mask=single.attention_mask,max_new_tokens=24,do_sample=False)
        assert until_eos(bg[i,batch.input_ids.shape[1]:])==until_eos(sg[0,single.input_ids.shape[1]:]),'Left-padding greedy parity failed'
    e.save(f'confirmation_padding_{name}.json',dict(passed=True,dev_greedy24_equal=True))
    tasks=e.read(e.OUT/'confirmation_tasks.json');path=e.OUT/f'confirmation_{name}.json';done=e.read(path) if path.exists() else {}
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for start in range(0,len(tasks),8):
        group=tasks[start:start+8]
        if all(t['task_id'] in done for t in group):continue
        assert len({t['id'] for t in group})==1
        inputs=tok(texts(group),return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
        torch.manual_seed(group[0]['batch_seed'])
        out=model.generate(input_ids=inputs.input_ids,attention_mask=inputs.attention_mask,
            max_new_tokens=384,do_sample=True,temperature=.9,top_p=.85,top_k=50,repetition_penalty=1.,
            pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        for t,tokens in zip(group,out[:,inputs.input_ids.shape[1]:]):
            reply=tok.decode(tokens,skip_special_tokens=True)
            done[t['task_id']]=dict(question=t['question'],reply=reply,**e.first_call_metrics(dict(turns=[reply]),t))
        e.save(path.name,done)
        if (start//8+1)%12==0:print(name,start//8+1,'/96',f'{time.perf_counter()-started:.1f}s',flush=True)
    e.save(f'confirmation_runtime_{name}.json',dict(seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))

def report():
    plan=e.read(e.OUT/'confirmation_plan.json');tasks=e.read(e.OUT/'confirmation_tasks.json');vectors={};means={}
    for name in plan['weights']:
        raw=e.read(e.OUT/f'confirmation_{name}.json');assert len(raw)==768
        ids=sorted({t['id'] for t in tasks});vals=[]
        for i in ids:vals.append(100*st.mean(raw[t['task_id']]['arguments_ok'] for t in tasks if t['id']==i))
        vectors[name]=np.array(vals);means[name]=float(np.mean(vals))
    a=plan['selected'];b=e.BASE;d=vectors[a]-vectors[b]
    boot=d[np.random.default_rng(20261026).integers(0,len(d),size=(20000,len(d)))].mean(1)
    ci=np.quantile(boot,[.0125,.9875]).tolist();initial=e.read(e.OUT/'summary.json')
    result=dict(means=means,difference_pp=float(d.mean()),se=float(d.std(ddof=1)/len(d)**.5),
        bootstrap975=ci,better=int(sum(d>0)),worse=int(sum(d<0)),same=int(sum(d==0)),questions=96,
        passed=bool(ci[0]>0 and all(v for k,v in initial['gates'].items() if k!='exact_math_arguments_gain_ci_above0')),
        original_primary_passed=False,weights=plan['weights'],no_model_change=True)
    e.save('confirmation_summary.json',result);print(e.json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','base','candidate','report']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='report':report()
    else:generate(e.BASE if a.stage=='base' else e.read(e.OUT/'confirmation_plan.json')['selected'])
