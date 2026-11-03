"""Development-only selection, followed by a frozen three-arm paired comparison."""
import argparse
import statistics as st
import numpy as np
import strong_teacher_common as e
from review_strong_teacher import content_key,math_key

def paired(a,b,lower_is_better=False):
    d=np.asarray(a,float)-np.asarray(b,float);n=len(d);assert n>1
    boot=d[np.random.default_rng(20261029).integers(n,size=(10000,n))].mean(1)
    positive=int(sum(d>1e-10));negative=int(sum(d< -1e-10))
    return dict(difference=float(d.mean()),se=float(d.std(ddof=1)/n**.5),ci95=np.quantile(boot,[.025,.975]).tolist(),n=n,
        better=negative if lower_is_better else positive,worse=positive if lower_is_better else negative,same=int(sum(abs(d)<=1e-10)),lower_is_better=lower_is_better)

def score(task,row,reviews):
    s=reviews[content_key(task,row)]['score'];assert s in [0,1,2]
    if not row['eligible']:assert s<2
    if 'exact_answer' in task and row['final'].strip().replace('\r\n','\n')!=task['exact_answer']:assert s<2,'Exact-output tasks cannot receive2 for containing the answer somewhere'
    return s

def math_correct(row,reviews):
    s=row['grade']['status']
    if s=='review':s=reviews[math_key(row['question'],row['final'])]['decision']
    assert s in ['pass','fail'];return s=='pass'

def chat(split,name,reviews):
    tasks=e.read(e.OUT/f'{split}_tasks.json');domain='devchat' if split=='dev' else 'finalchat';raw=e.read(e.OUT/f'{domain}_{name}.json')
    assert len(raw)==3*len(tasks);vec={};scores=[];nc=[];conditions={};categories={};languages={}
    for task in tasks:
        rr=[raw[task['id']+'_'+c] for c in e.CONDITIONS];ss=[score(task,r,reviews) for r in rr]
        vec[task['id']]=dict(score=st.mean(ss),usable=100*st.mean(s==2 for s in ss));scores+=ss
        nc+=[bool(r['eligible'] and not any('<tool_call' in turn for turn in r['turns'])) for r in rr]
    for field,values,destination in [('condition',e.CONDITIONS,conditions),('category',sorted({t['category'] for t in tasks}),categories),('language',['zh','en'],languages)]:
        for value in values:
            ss=[score(t,raw[t['id']+'_'+c],reviews) for t in tasks for c in e.CONDITIONS if (c==value if field=='condition' else t[field]==value)]
            destination[value]=dict(mean_score=st.mean(ss),usable_percent=100*st.mean(s==2 for s in ss),responses=len(ss))
    return dict(mean_score=st.mean(scores),usable_percent=100*st.mean(s==2 for s in scores),nonempty_no_call_percent=100*st.mean(nc),conditions=conditions,categories=categories,languages=languages),vec

def identity(domain,name,reviews):
    tasks=e.read(e.OUT/('identity_dev_tasks.json' if domain=='devidentity' else 'identity_tasks.json'));raw=e.read(e.OUT/f'{domain}_{name}.json')
    assert len(raw)==len(tasks)
    return sum(score(t,raw[t['id']],reviews)==2 for t in tasks)

def select():
    assert not (e.OUT/'final_lock.json').exists()
    r=e.read(e.OUT/'content_reviews.json');mr=e.read(e.OUT/'math_reviews.json')
    baseline,_=chat('dev',e.BASE,r);bl=e.read(e.OUT/f'loss_dev_{e.BASE}.json')['mean']
    bm=100*st.mean(math_correct(x,mr) for x in e.read(e.OUT/f'devmath_{e.BASE}.json').values());bi=identity('devidentity',e.BASE,r)
    candidates=[];selected={}
    for arm in ['weak','strong']:
        arm_candidates=[]
        for epoch in [1,2]:
            name=f'strong_teacher_{arm}_e{epoch}';summary,_=chat('dev',name,r);loss=e.read(e.OUT/f'loss_dev_{name}.json')['mean']
            math=100*st.mean(math_correct(x,mr) for x in e.read(e.OUT/f'devmath_{name}.json').values());ids=identity('devidentity',name,r)
            gates=dict(no_call_ge90=summary['nonempty_no_call_percent']>=90,nll_increase_le005=loss<=bl+.05,math_drop_le10pp=math>=bm-10,identity_not_below_v4=ids>=bi)
            c=dict(name=name,arm=arm,**summary,loss=loss,math=math,identity_usable=ids,gates=gates,eligible=all(gates.values()))
            candidates.append(c);arm_candidates.append(c)
        allowed=[c for c in arm_candidates if c['eligible']]
        if allowed:selected[arm]=max(allowed,key=lambda c:(c['mean_score'],c['usable_percent'],-c['loss']))['name']
    result=dict(baseline=baseline,baseline_loss=bl,baseline_math=bm,baseline_identity=bi,candidates=candidates,selected=selected,formal_started=False)
    e.save('selection.json',result,immutable=True)
    if set(selected)!={'weak','strong'}:raise RuntimeError('At least one arm failed dev retention; preserve result and do not start formal comparison')
    assert e.read(e.OUT/'preflight_tests.json')['passed']
    files=[e.OUT/n for n in ['plan.json','training_lock.json','train_weak.jsonl','train_strong.jsonl','dev_tasks.json','final_tasks.json','selection.json','matched_data_audit.json','preflight_tests.json']]
    files+=[e.P/'my_eval/scripts'/n for n in ['strong_teacher_common.py','eval_strong_teacher.py','review_strong_teacher.py','analyse_strong_teacher.py','train_strong_teacher.py','eval_chat_repair.py','tool_choice_round2.py','strict_math_rescore.py']]
    files+=[e.P/'scripts/best_of_n.py',e.P/'scripts/demo_tools.py']
    e.save('final_lock.json',dict(weights={n:e.sha(e.P/f'out/{n}_768.pth') for n in [e.BASE,selected['weak'],selected['strong']]},files_sha256={str(p.relative_to(e.P)):e.sha(p) for p in files}),immutable=True)
    print(result)

def final():
    from eval_strong_teacher import verify_final
    selected=e.read(e.OUT/'selection.json')['selected'];verify_final(selected['strong'])
    names=[e.BASE,selected['weak'],selected['strong']];reviews=e.read(e.OUT/'content_reviews.json');mr=e.read(e.OUT/'math_reviews.json')
    tasks=e.read(e.OUT/'final_tasks.json');models={};vectors={};comparisons={};retention={}
    for name in names:models[name],vectors[name]=chat('final',name,reviews)
    for label,b in [('strong_minus_v4',e.BASE),('strong_minus_weak',selected['weak']),('weak_minus_v4',e.BASE)]:
        a=selected['weak'] if label=='weak_minus_v4' else selected['strong']
        comparisons[label]={k:paired([vectors[a][t['id']][k] for t in tasks],[vectors[b][t['id']][k] for t in tasks]) for k in ['score','usable']}
    for domain,n in [('historical',197),('freshmath',48)]:
        raw={name:e.read(e.OUT/f'{domain}_{name}.json') for name in names};keys=sorted(raw[e.BASE]);assert len(keys)==n and all(sorted(r)==keys for r in raw.values())
        vals={name:[100*math_correct(raw[name][k],mr) for k in keys] for name in names}
        retention[domain]=dict(means={name:st.mean(v) for name,v in vals.items()},strong_minus_v4=paired(vals[selected['strong']],vals[e.BASE]))
    losses={name:e.read(e.OUT/f'loss_final_{name}.json')['rows'] for name in names};indices=[r['index'] for r in losses[e.BASE]]
    assert len(indices)==296 and all([r['index'] for r in rr]==indices for rr in losses.values())
    vals={name:[r['loss'] for r in rr] for name,rr in losses.items()}
    retention['nll296']=dict(means={name:st.mean(v) for name,v in vals.items()},strong_minus_v4=paired(vals[selected['strong']],vals[e.BASE],True))
    ids={name:identity('identity',name,reviews) for name in names};s=models[selected['strong']];b=models[e.BASE];gain=comparisons['strong_minus_v4']
    gates=dict(mean_gain_ge015=gain['score']['difference']>=.15,mean_gain_ci_lower_gt0=gain['score']['ci95'][0]>0,usable_gain_ge8pp=gain['usable']['difference']>=8,
        no_tools_drop_le5pp=s['conditions']['no_tools']['usable_percent']>=b['conditions']['no_tools']['usable_percent']-5,
        historical_math_drop_le5pp=retention['historical']['strong_minus_v4']['difference']>=-5,
        prior_math48_drop_le5pp=retention['freshmath']['strong_minus_v4']['difference']>=-5,
        nll_increase_le005=retention['nll296']['strong_minus_v4']['difference']<=.05,
        identity_drop_le1=ids[selected['strong']]>=ids[e.BASE]-1,complete_no_call_ge90=s['nonempty_no_call_percent']>=90)
    summary=dict(models=models,comparisons=comparisons,retention=retention,identity14=ids,gates=gates,accepted=all(gates.values()),
        evidence_larger_teacher_beats_matched_smaller=comparisons['strong_minus_weak']['score']['ci95'][0]>0,
        interpretation='Accepted improvement over v4 and evidence for larger-teacher superiority are separate claims. Report both, including failures.')
    e.save('summary.json',summary);print(e.json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','final']);a=p.parse_args();select() if a.stage=='select' else final()
