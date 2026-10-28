"""Development-only prompt choice, then paired, question-cluster final statistics."""
import argparse
import statistics as st
import numpy as np
import chat_quality_common as e
from chat_quality_support import tool_activity
from eval_chat_quality import CONDITIONS
from review_chat_quality import key

def score(task,row):
    r=e.read(e.OUT/'content_reviews.json')[key(task,row)]
    assert r['score'] in [0,1,2]
    if not row['eligible']:assert r['score']!=2
    return r['score']

def paired(a,b):
    d=np.array(a,float)-np.array(b,float);n=len(d)
    boot=d[np.random.default_rng(20261028).integers(n,size=(10000,n))].mean(1)
    return dict(difference=float(d.mean()),se=float(d.std(ddof=1)/n**.5),ci95=np.quantile(boot,[.025,.975]).tolist(),
        n=n,better=int(sum(d>1e-10)),worse=int(sum(d< -1e-10)),same=int(sum(abs(d)<=1e-10)))

def select():
    assert not (e.OUT/'final_lock.json').exists()
    tasks=e.read(e.OUT/'dev_tasks.json');base=e.read(e.OUT/f'dev_{e.MODEL}_baseline.json')
    candidates=[]
    for backend,policy in [(m,p) for m in [e.MODEL,'MiniMind-SFT','MiniMind-Official'] for p in ['standard','concise']]:
        plain=e.read(e.OUT/f'dev_{backend}_{policy}.json');values=[];seconds=[]
        for t in tasks:
            p=plain[t['id']+'_no_tools']
            for condition in CONDITIONS:
                r=base[t['id']+'_'+condition]
                chosen=p if condition=='no_tools' or (not tool_activity(r) and r.get('failure')!='context_limit' and p['eligible'] and not tool_activity(p)) else r
                values.append(score(t,chosen))
                seconds.append(chosen['generation_seconds']+(r['generation_seconds'] if condition!='no_tools' and chosen is p else 0))
        identity_count=sum(score(t,plain[t['id']+'_no_tools'])==2 for t in tasks if t['category']=='identity')
        baseline_identity=sum(score(t,base[t['id']+'_no_tools'])==2 for t in tasks if t['category']=='identity')
        candidates.append(dict(chat_backend=backend,policy=policy,score_mean=st.mean(values),usable_percent=100*st.mean(x==2 for x in values),estimated_seconds=st.mean(seconds),identity_usable4=identity_count,dev_identity_gate=identity_count>=baseline_identity-1))
    allowed=[c for c in candidates if c['dev_identity_gate']]
    assert allowed,'No development candidate retains identity; do not proceed to final'
    selected=max(allowed,key=lambda x:(x['score_mean'],x['usable_percent'],-x['estimated_seconds']))
    diagnostic={}
    for model in [e.MODEL,'MiniMind-YuyangLU-v2']:
        rows=e.read(e.OUT/f'dev_{model}_baseline.json');diagnostic[model]={}
        for c in CONDITIONS:
            rs=[(t,rows[t['id']+'_'+c]) for t in tasks if t['id']+'_'+c in rows]
            if rs:diagnostic[model][c]=dict(mean_score=st.mean(score(t,r) for t,r in rs),usable=sum(score(t,r)==2 for t,r in rs),n=len(rs))
    e.save('selection.json',dict(selected=selected['policy'],chat_backend=selected['chat_backend'],candidates=candidates,diagnostic=diagnostic,method='development only; no formal generation seen'))
    files=[e.OUT/n for n in ['plan.json','dev_tasks.json','final_tasks.json','rubric.json','selection.json','tests.json']]
    files += [e.P/'my_eval/scripts'/n for n in ['chat_quality_common.py','eval_chat_quality.py','analyse_chat_quality.py','review_chat_quality.py','retention_chat_quality.py']]
    files += [e.P/'scripts/chat_quality_support.py']
    hashes=e.metadata();hashes.update({str(p.relative_to(e.P)):e.sha(p) for p in files})
    for name in ['model.safetensors','config.json','tokenizer.json','chat_template.jinja']:
        path=e.P/'scripts'/selected['chat_backend']/name
        if path.exists():hashes[str(path.relative_to(e.P))]=e.sha(path)
    e.save('final_lock.json',dict(selected=selected['policy'],chat_backend=selected['chat_backend'],files_sha256=hashes))
    print(e.old.json.dumps(dict(selected=selected,candidates=candidates,diagnostic=diagnostic),ensure_ascii=False,indent=2),flush=True)

def final():
    from eval_chat_quality import verify
    verify();selection=e.read(e.OUT/'selection.json')['selected'];tasks=e.read(e.OUT/'final_tasks.json')
    vectors={};summary=dict(model=e.MODEL,policy=selection,arms={},pairs={},conditions={},categories={},gates={})
    for arm in ['baseline',selection]:
        data=e.read(e.OUT/f'final_{e.MODEL}_{arm}.json');assert len(data)==len(tasks)*3
        vectors[arm]={};values=[];usable=[];times=[];tokens=[];no_calls=[]
        for t in tasks:
            rows=[data[t['id']+'_'+c] for c in CONDITIONS];scores=[score(t,r) for r in rows]
            vectors[arm][t['id']]=dict(score=st.mean(scores),usable=100*st.mean(x==2 for x in scores))
            values.extend(scores);usable.extend(x==2 for x in scores);times.extend(r['generation_seconds'] for r in rows)
            tokens.extend(r['generated_tokens'] for r in rows);no_calls.extend(r['eligible'] and not tool_activity(r) for r in rows)
        summary['arms'][arm]=dict(mean_score=st.mean(values),usable_percent=100*st.mean(usable),nonempty_no_call_percent=100*st.mean(no_calls),mean_seconds=st.mean(times),mean_generated_tokens=st.mean(tokens),responses=len(data))
        summary['conditions'][arm]={};summary['categories'][arm]={}
        for c in CONDITIONS:
            ss=[score(t,data[t['id']+'_'+c]) for t in tasks]
            summary['conditions'][arm][c]=dict(mean_score=st.mean(ss),usable_percent=100*st.mean(x==2 for x in ss))
        for category in sorted({t['category'] for t in tasks}):
            ss=[score(t,data[t['id']+'_'+c]) for t in tasks if t['category']==category for c in CONDITIONS]
            summary['categories'][arm][category]=dict(mean_score=st.mean(ss),usable_percent=100*st.mean(x==2 for x in ss))
    for metric in ['score','usable']:
        summary['pairs'][metric]=paired([vectors[selection][t['id']][metric] for t in tasks],[vectors['baseline'][t['id']][metric] for t in tasks])
    nonidentity=[t for t in tasks if t['category']!='identity']
    summary['pairs']['nonidentity_score']=paired([vectors[selection][t['id']]['score'] for t in nonidentity],[vectors['baseline'][t['id']]['score'] for t in nonidentity])
    summary['chat_backend']=e.read(e.OUT/'selection.json')['chat_backend']
    mathreviews=e.read(e.OUT/'math_reviews.json')
    from review_tool_choice_round2 import key as mathkey
    def correct(row):
        g=row['grade']['status'];g=mathreviews[mathkey(row['question'],row['final'])]['decision'] if g=='review' else g
        assert g in ['pass','fail'];return g=='pass'
    retention={}
    for domain,n in [('historical',197),('fresh_math',48)]:
        a=e.read(e.OUT/f'{domain}_candidate.json');b=e.read(e.OUT/f'{domain}_baseline.json');assert len(a)==len(b)==n and a.keys()==b.keys()
        av=[100*correct(a[k]) for k in sorted(a)];bv=[100*correct(b[k]) for k in sorted(b)]
        retention[domain]=dict(baseline=st.mean(bv),candidate=st.mean(av),paired=paired(av,bv),
            original_tool_paths_preserved=sum(r.get('chat_mode')=='tool_path' for r in a.values()))
    identity=e.read(e.OUT/'identity_tasks.json');identitycounts={}
    for arm in ['baseline',selection]:
        raw=e.read(e.OUT/f'identity_{e.MODEL}_{arm}.json');assert len(raw)==14
        identitycounts[arm]=sum(score(t,raw[t['id']])==2 for t in identity)
    summary['retention']=retention;summary['identity_usable14']=identitycounts
    base=summary['arms']['baseline'];new=summary['arms'][selection]
    summary['gates']=dict(content_gain_ge020=new['mean_score']>=base['mean_score']+.2,
        content_gain_ci_lower_gt0=summary['pairs']['score']['ci95'][0]>0,
        nonidentity_gain_ge020_and_ci_lower_gt0=summary['pairs']['nonidentity_score']['difference']>=.2 and summary['pairs']['nonidentity_score']['ci95'][0]>0,
        usable_gain_ge10pp=new['usable_percent']>=base['usable_percent']+10,
        plain_no_tools_usable_drop_le5pp=summary['conditions'][selection]['no_tools']['usable_percent']>=summary['conditions']['baseline']['no_tools']['usable_percent']-5,
        historical_math_drop_le5pp=retention['historical']['candidate']>=retention['historical']['baseline']-5,
        fresh_math_drop_le5pp=retention['fresh_math']['candidate']>=retention['fresh_math']['baseline']-5,
        identity_full_content_count_drop_le1=identitycounts[selection]>=identitycounts['baseline']-1,
        unit_tests=e.read(e.OUT/'tests.json')['passed'])
    summary['accepted']=all(summary['gates'].values());e.save('summary.json',summary)
    print(e.old.json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','final']);a=p.parse_args()
    select() if a.stage=='select' else final()

