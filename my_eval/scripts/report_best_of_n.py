"""Compute paired outcomes, including an independent strict-math endpoint."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics as st
import numpy as np
P=Path(__file__).resolve().parents[2]
OUT=P/'my_eval/results/best_of_n_20261026'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
def key(q,a):return hashlib.sha256((q+'\n'+a).encode()).hexdigest()

def paired(x,y):
    diff=np.array(x,dtype=float)-np.array(y,dtype=float)
    rng=np.random.default_rng(20261026)
    boot=diff[rng.integers(0,len(diff),size=(10000,len(diff)))].mean(axis=1)
    return dict(difference=float(diff.mean()),se=float(diff.std(ddof=1)/len(diff)**.5),
        bootstrap95=np.quantile(boot,[.025,.975]).tolist(),better=int(sum(diff>0)),worse=int(sum(diff<0)),same=int(sum(diff==0)))

def run():
    tasks=read(OUT/'tasks.json');scored=read(OUT/'scored.json')
    auto=read(OUT/'math_automatic.json');reviews=read(OUT/'math_reviews.json')
    chat=read(OUT/'chat_reviews.json')
    assert len(scored)==len(tasks)==226
    math_rows=[];chat_rows=[]
    for task in tasks:
        row=scored[task['id']];cands=row['candidates'];selected=row['selected']
        selected['eligible_first']=next((i for i,c in enumerate(cands) if c['eligible'] and c['rm'] is not None),0)
        if task['kind']=='math':
            assert len(auto[task['id']])==4
            correct=[]
            for a,c in zip(auto[task['id']],cands):
                assert a['final']==c['final']
                d=a['status']
                if d=='review':d=reviews[a['sha256']]['decision']
                assert d in ['pass','fail']
                correct.append(int(d=='pass'))
            math_rows.append(dict(id=task['id'],expression_count=len(task['gt']),correct=correct,
                selected=selected,scores={n:correct[k]*100 for n,k in selected.items()},
                random_mean=st.mean(correct)*100,oracle=max(correct)*100,
                selection_miss=bool(max(correct) and not correct[selected['4']]),
                first_seconds=cands[0]['generation_seconds'],
                four_seconds=sum(c['generation_seconds'] for c in cands)+row['score_seconds']))
        else:
            quality=[chat[key(json.dumps(task['messages'],ensure_ascii=False,sort_keys=True),c['final'])]['score'] for c in cands]
            assert all(q in (0,1,2) for q in quality)
            chat_rows.append(dict(id=task['id'],quality=quality,selected=selected,
                scores={n:quality[k] for n,k in selected.items()},
                rm={n:cands[k]['rm'] for n,k in selected.items()},
                chars={n:len(cands[k]['final']) for n,k in selected.items()},
                first_seconds=cands[0]['generation_seconds'],
                four_seconds=sum(c['generation_seconds'] for c in cands)+row['score_seconds']))
    summary={'model':'MiniMind-YuyangLU-v2','math':{},'chat':{}}
    for label,rows in [('math',math_rows),('chat',chat_rows)]:
        summary[label]['n']=len(rows)
        summary[label]['means']={n:st.mean(r['scores'][n] for r in rows) for n in ['1','2','4','eligible_first']}
        summary[label]['paired']={f'{n}_minus_{b}':paired([r['scores'][n] for r in rows],[r['scores'][b] for r in rows])
            for n,b in [('2','1'),('4','1'),('4','2'),('4','eligible_first')]}
        summary[label]['seconds']={n:st.mean(r[f'{s}_seconds'] for r in rows) for n,s in [('1','first'),('4','four')]}
        summary[label]['seconds']['ratio4to1']=summary[label]['seconds']['4']/summary[label]['seconds']['1']
    summary['math'].update(random_mean=st.mean(r['random_mean'] for r in math_rows),
        oracle=st.mean(r['oracle'] for r in math_rows),selection_misses=sum(r['selection_miss'] for r in math_rows),
        paired4_vs_random_mean=paired([r['scores']['4'] for r in math_rows],[r['random_mean'] for r in math_rows]),
        by_expression_count={str(k):{n:st.mean(r['scores'][n] for r in math_rows if r['expression_count']==k)
                                   for n in ['1','2','4']} for k in [1,2,3]},
        review_runs=sum(a['status']=='review' for aa in auto.values() for a in aa),
        review_unique=len({a['sha256'] for aa in auto.values() for a in aa if a['status']=='review'}))
    summary['chat']['rm_mean_on_common_scorable']={}
    common=[r for r in chat_rows if all(r['rm'][n] is not None for n in ['1','2','4'])]
    summary['chat']['rm_common_n']=len(common)
    for n in ['1','2','4']:
        summary['chat']['rm_mean_on_common_scorable'][n]=st.mean(r['rm'][n] for r in common)
    summary['chat']['rm4_vs_1_on_common']=paired([r['rm']['4'] for r in common],[r['rm']['1'] for r in common])
    summary['chat']['answer_chars']={n:st.mean(r['chars'][n] for r in chat_rows) for n in ['1','2','4']}
    summary['eligibility']={kind:dict(Counter(c.get('failure') or 'complete' for t in tasks if t['kind']==kind
        for c in scored[t['id']]['candidates'])) for kind in ['math','chat']}
    summary['fallback_questions']={str(n):sum(not any(c['eligible'] and c['rm'] is not None
        for c in row['candidates'][:n]) for row in scored.values()) for n in [2,4]}
    save('final_outcomes.json',dict(math=math_rows,chat=chat_rows));save('summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
