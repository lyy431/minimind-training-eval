"""Apply explicit review records, compute paired results and publish a separate report."""
from collections import Counter
import hashlib
import itertools
import json
import statistics as st
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parents[2]
OUT=P/'my_eval/results/strict_math_20261026'
OLD=P/'my_eval/results/distill_control_20261025'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')

auto=read(OUT/'automatic.json');reviews=read(OUT/'manual_reviews.json')['rows']
lookup={r['sha256']:r for r in reviews}
assert len(lookup)==len(reviews)
indices=read(OLD/'plan.json')['primary_math_indices']
prior=read(OLD/'summary.json')
final={};summary={'models':{},'pairs':{},'primary_n':197,'samples_per_question':2}
vectors={}
for name,rows in auto.items():
    assert len(rows)==400
    final[name]=[]
    for r in rows:
        h=hashlib.sha256((r['question']+'\n'+r['final']).encode()).hexdigest()
        if r['status']=='review':
            assert h in lookup,('Unreviewed answer',name,r['problem'],r['sample'])
            decision=lookup[h]['decision'];note=lookup[h]['note']
        else:decision=r['status'];note=r['reason']
        assert decision in ['pass','fail']
        final[name].append({**r,'review_key':h if r['status']=='review' else None,
            'final_pass':decision=='pass','adjudication':note})
    primary=[r for r in final[name] if r['problem'] in indices]
    vectors[name]=np.array([st.mean(r['final_pass'] for r in final[name] if r['problem']==i) for i in indices])*100
    assert len(primary)==394
    summary['models'][name]={
        'strict197':float(vectors[name].mean()),'correct_runs197':sum(r['final_pass'] for r in primary),
        'strict200':100*st.mean(r['final_pass'] for r in final[name]),
        'legacy197':100*st.mean(r['legacy_pass'] for r in primary),
        'whole_number197':prior['models'][name]['math197_diagnostics']['whole_number_accuracy_percent'],
        'old_pass_now_fail197':sum(r['legacy_pass'] and not r['final_pass'] for r in primary),
        'old_fail_now_pass197':sum(not r['legacy_pass'] and r['final_pass'] for r in primary),
        'automatic_counts200':dict(Counter(r['status'] for r in rows)),
        'reviewed_runs197':sum(r['status']=='review' for r in primary),
        'review_passes197':sum(r['status']=='review' and r['final_pass'] for r in primary),
        'automation_only_bounds197':[100*sum(r['status']=='pass' for r in primary)/394,
            100*sum(r['status']!='fail' for r in primary)/394],
        'by_expression_count':{str(k):100*st.mean(r['final_pass'] for r in primary if len(r['gt'])==k) for k in [1,2,3]}}
for a,b in itertools.combinations(auto,2):
    diff=vectors[a]-vectors[b]
    rng=np.random.default_rng(20261026)
    boot=diff[rng.integers(0,197,size=(10000,197))].mean(axis=1)
    summary['pairs'][a+'_minus_'+b]={'difference_pp':float(diff.mean()),'se':float(diff.std(ddof=1)/197**.5),
        'bootstrap95_percentile_pp':np.quantile(boot,[.025,.975]).tolist(),
        'better':int((diff>0).sum()),'worse':int((diff<0).sum()),'same':int((diff==0).sum())}
summary['ranking_by_point_estimate']=sorted(auto,key=lambda n:summary['models'][n]['strict197'],reverse=True)
summary['review']={'unique_reviewed_texts':len(reviews),'reviewed_runs':sum(r['status']=='review' for rs in auto.values() for r in rs),
    'reviewer':'Assistant, not independent human/blinded adjudication; all parser-uncertain texts reviewed;68 distinct automatic decisions spot-checked.',
    'unresolved':0}
summary['sha256']={str(p.relative_to(P)):sha(p) for p in [OLD/'math_final.json',P/'dataset/agent_math_eval200.jsonl',
    P/'my_eval/scripts/strict_math_rescore.py',P/'my_eval/scripts/test_strict_math_rescore.py',
    OUT/'manual_reviews.json',OUT/'automatic.json',OUT/'regression_tests.json',OUT/'automatic_audit_review.json']}
save('final_results.json',final);save('summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
