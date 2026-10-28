"""Content-first selection and preregistered paired final acceptance."""
import argparse
import statistics as st
import numpy as np
import chat_repair_common as e
from review_chat_repair import key
from review_tool_choice_round2 import key as mathkey
from eval_chat_repair import CONDITIONS,verify_final

def paired(a,b):
    d=np.array(a,float)-np.array(b,float);n=len(d)
    boot=d[np.random.default_rng(20261028).integers(n,size=(10000,n))].mean(1)
    return dict(difference=float(d.mean()),se=float(d.std(ddof=1)/n**.5),ci95=np.quantile(boot,[.025,.975]).tolist(),n=n,
        better=int(sum(d>1e-10)),worse=int(sum(d< -1e-10)),same=int(sum(abs(d)<=1e-10)))

def content(task,row,reviews):
    r=reviews[key(task,row)];assert r['score'] in [0,1,2]
    if not row['eligible']:assert r['score']!=2
    return r['score']

def correct(row,reviews):
    status=row['grade']['status']
    if status=='review':status=reviews[mathkey(row['question'],row['final'])]['decision']
    assert status in ['pass','fail'];return status=='pass'

def chat_summary(domain,name,tasks,reviews):
    raw=e.read(e.OUT/f'{domain}_{name}.json');assert len(raw)==len(tasks)*3
    vec={};all_scores=[];times=[];tokens=[];nc=[];conditions={};categories={}
    for t in tasks:
        rows=[raw[t['id']+'_'+c] for c in CONDITIONS];ss=[content(t,r,reviews) for r in rows]
        vec[t['id']]=dict(score=st.mean(ss),usable=100*st.mean(s==2 for s in ss))
        all_scores.extend(ss);times.extend(r['generation_seconds'] for r in rows);tokens.extend(r['generated_tokens'] for r in rows)
        nc.extend(bool(r['eligible'] and not any('<tool_call' in v for v in r['turns'])) for r in rows)
    for c in CONDITIONS:
        vals=[content(t,raw[t['id']+'_'+c],reviews) for t in tasks]
        conditions[c]=dict(mean_score=st.mean(vals),usable_percent=100*st.mean(s==2 for s in vals))
    for cat in sorted({t['category'] for t in tasks}):
        vals=[content(t,raw[t['id']+'_'+c],reviews) for t in tasks if t['category']==cat for c in CONDITIONS]
        categories[cat]=dict(mean_score=st.mean(vals),usable_percent=100*st.mean(s==2 for s in vals))
    return dict(mean_score=st.mean(all_scores),usable_percent=100*st.mean(s==2 for s in all_scores),nonempty_no_call_percent=100*st.mean(nc),
        mean_seconds=st.mean(times),mean_generated_tokens=st.mean(tokens),conditions=conditions,categories=categories,responses=len(raw)),vec

def select():
    assert not (e.OUT/'final_lock.json').exists()
    tasks=e.read(e.OUT/'dev_tasks.json');reviews=e.read(e.OUT/'content_reviews.json');mathreviews=e.read(e.OUT/'math_reviews.json')
    base,_=chat_summary('devchat',e.BASE,tasks,reviews)
    base_loss=e.read(e.OUT/f'loss_dev_{e.BASE}.json')['mean']
    base_math=100*st.mean(correct(r,mathreviews) for r in e.read(e.OUT/f'devmath_{e.BASE}.json').values())
    candidates=[]
    for i in [1,2,3]:
        name=f'chat_repair_e{i}';s,_=chat_summary('devchat',name,tasks,reviews);loss=e.read(e.OUT/f'loss_dev_{name}.json')['mean']
        math=100*st.mean(correct(r,mathreviews) for r in e.read(e.OUT/f'devmath_{name}.json').values())
        gates=dict(no_call_ge90=s['nonempty_no_call_percent']>=90,nll_increase_le005=loss<=base_loss+.05,math_drop_le10pp=math>=base_math-10)
        candidates.append(dict(name=name,**s,loss=loss,math=math,dev_gates=gates,eligible=all(gates.values())))
    allowed=[c for c in candidates if c['eligible']]
    if not allowed:
        e.save('development_selection_failed.json',dict(base=e.BASE,baseline_chat=base,baseline_loss=base_loss,baseline_math=base_math,candidates=candidates,formal_started=False))
        raise RuntimeError('No qualified development candidate; retain results and do not start formal evaluation')
    selected=max(allowed,key=lambda c:(c['mean_score'],c['usable_percent'],-c['loss']))
    selection=dict(base=e.BASE,baseline_chat=base,baseline_loss=base_loss,baseline_math=base_math,candidates=candidates,
        selected=selected['name'],dev_gates_passed=bool(allowed),no_formal_results_seen=True)
    e.save('selection.json',selection)
    assert e.read(e.OUT/'tests.json')['passed']
    files=[e.OUT/n for n in ['plan.json','dev_tasks.json','final_tasks.json','rubric.json','train.jsonl','data_audit.json','token_audit.json','selection.json','tests.json','teacher_answers.json','teacher_curation.json','menu_balance_audit.json']]
    files+=[e.P/'my_eval/scripts'/n for n in ['chat_repair_common.py','train_chat_repair.py','eval_chat_repair.py','review_chat_repair.py','analyse_chat_repair.py']]
    files+=[e.P/'scripts/best_of_n.py',e.P/'scripts/demo_tools.py',e.P/'my_eval/scripts/strict_math_rescore.py']
    e.save('final_lock.json',dict(selected=selected['name'],weights={n:e.sha(e.P/f'out/{n}_768.pth') for n in [e.BASE,selected['name']]},files_sha256={str(p.relative_to(e.P)):e.sha(p) for p in files}))
    print(e.json.dumps(selection,ensure_ascii=False,indent=2),flush=True)

def final():
    selected=e.read(e.OUT/'selection.json')['selected'];verify_final(selected)
    tasks=e.read(e.OUT/'final_tasks.json');reviews=e.read(e.OUT/'content_reviews.json');mathreviews=e.read(e.OUT/'math_reviews.json')
    names=[e.BASE,selected];vectors={};summary=dict(base=e.BASE,candidate=selected,models={},pairs={},retention={},gates={})
    for name in names:summary['models'][name],vectors[name]=chat_summary('finalchat',name,tasks,reviews)
    for metric in ['score','usable']:
        summary['pairs'][metric]=paired([vectors[selected][t['id']][metric] for t in tasks],[vectors[e.BASE][t['id']][metric] for t in tasks])
    ts=[t for t in tasks if t['category']!='identity']
    summary['pairs']['nonidentity_score']=paired([vectors[selected][t['id']]['score'] for t in ts],[vectors[e.BASE][t['id']]['score'] for t in ts])
    for domain,n in [('historical',197),('freshmath',48)]:
        raw=[e.read(e.OUT/f'{domain}_{name}.json') for name in names];assert len(raw[0])==len(raw[1])==n and raw[0].keys()==raw[1].keys()
        vals=[[100*correct(r[k],mathreviews) for k in sorted(r)] for r in raw]
        summary['retention'][domain]=dict(baseline=st.mean(vals[0]),candidate=st.mean(vals[1]),paired=paired(vals[1],vals[0]))
    losses=[e.read(e.OUT/f'loss_final_{name}.json')['rows'] for name in names]
    assert [r['index'] for r in losses[0]]==[r['index'] for r in losses[1]] and len(losses[0])==296
    summary['retention']['nll296']=dict(baseline=st.mean(r['loss'] for r in losses[0]),candidate=st.mean(r['loss'] for r in losses[1]),
        paired=paired([r['loss'] for r in losses[1]],[r['loss'] for r in losses[0]]))
    ids=e.read(e.OUT/'identity_tasks.json');identity={}
    for name in names:
        raw=e.read(e.OUT/f'identity_{name}.json');assert len(raw)==14
        identity[name]=dict(usable=sum(content(t,raw[t['id']],reviews)==2 for t in ids),mean_score=st.mean(content(t,raw[t['id']],reviews) for t in ids),
            name_hits=sum('minimind' in r['final'].casefold() and 'yuyang' in r['final'].casefold() for r in raw.values()))
    summary['identity14']=identity;b,a=names;bb,aa=summary['models'][b],summary['models'][a]
    summary['gates']=dict(dev_gates=e.read(e.OUT/'selection.json')['dev_gates_passed'],content_gain_ge020=aa['mean_score']>=bb['mean_score']+.2,
        content_gain_ci_lower_gt0=summary['pairs']['score']['ci95'][0]>0,
        nonidentity_gain_ge020_ci_lower_gt0=summary['pairs']['nonidentity_score']['difference']>=.2 and summary['pairs']['nonidentity_score']['ci95'][0]>0,
        usable_gain_ge10pp=aa['usable_percent']>=bb['usable_percent']+10,
        no_tools_usable_drop_le5pp=aa['conditions']['no_tools']['usable_percent']>=bb['conditions']['no_tools']['usable_percent']-5,
        historical_math_drop_le5pp=summary['retention']['historical']['candidate']>=summary['retention']['historical']['baseline']-5,
        prior_round_math48_drop_le5pp=summary['retention']['freshmath']['candidate']>=summary['retention']['freshmath']['baseline']-5,
        nll_increase_le005=summary['retention']['nll296']['candidate']<=summary['retention']['nll296']['baseline']+.05,
        identity_full_content_drop_le1=identity[a]['usable']>=identity[b]['usable']-1,
        ordinary_nonempty_no_call_ge90=aa['nonempty_no_call_percent']>=90,
        data_tests=e.read(e.OUT/'tests.json')['passed'])
    summary['accepted']=all(summary['gates'].values());e.save('summary.json',summary);print(e.json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','final']);a=p.parse_args();select() if a.stage=='select' else final()
