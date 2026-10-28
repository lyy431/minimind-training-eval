"""Independent standard-library recomputation of final clustered means and SEs."""
from pathlib import Path
import hashlib
import json
import math
import random
import statistics as st

P=Path(__file__).resolve().parents[2]
OUT=P/'my_eval/results/chat_repair_20261027'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(a,b,abs_tol=1e-10),(a,b)
def paired(a,b):
    differences=[x-y for x,y in zip(a,b)];n=len(differences)
    return st.mean(differences),st.stdev(differences)/math.sqrt(n)

def main():
    s=read('summary.json');names=[s['base'],s['candidate']];tasks=read('final_tasks.json');reviews=read('content_reviews.json')
    assert len(tasks)==48 and len({t['id'] for t in tasks})==48
    vectors={};conditions=['no_tools','six_original','six_reworded']
    for name in names:
        raw=read(f'finalchat_{name}.json');assert len(raw)==144;vectors[name]={};all_scores=[]
        for task in tasks:
            scores=[]
            for condition in conditions:
                r=raw[task['id']+'_'+condition]
                fingerprint=hashlib.sha256(json.dumps([task['messages'],r['final'],r['eligible'],r.get('failure')],ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:16]
                v=reviews[fingerprint]['score'];assert v in [0,1,2];assert r['eligible'] or v<2;scores.append(v)
            vectors[name][task['id']]=dict(score=sum(scores)/3,usable=100*sum(v==2 for v in scores)/3)
            all_scores+=scores
        close(st.mean(all_scores),s['models'][name]['mean_score'])
        close(100*sum(v==2 for v in all_scores)/144,s['models'][name]['usable_percent'])
    b,a=names
    for metric in ['score','usable']:
        diff,se=paired([vectors[a][t['id']][metric] for t in tasks],[vectors[b][t['id']][metric] for t in tasks])
        close(diff,s['pairs'][metric]['difference']);close(se,s['pairs'][metric]['se'])
    nt=[t for t in tasks if t['category']!='identity'];assert len(nt)==36
    diff,se=paired([vectors[a][t['id']]['score'] for t in nt],[vectors[b][t['id']]['score'] for t in nt])
    close(diff,s['pairs']['nonidentity_score']['difference']);close(se,s['pairs']['nonidentity_score']['se'])
    mathreviews=read('math_reviews.json')
    def correct(r):
        status=r['grade']['status']
        if status=='review':status=mathreviews[hashlib.sha256((r['question']+'\n'+r['final']).encode()).hexdigest()]['decision']
        assert status in ['pass','fail'];assert status!='pass' or r['eligible'];return 100*(status=='pass')
    for domain,n in [('historical',197),('freshmath',48)]:
        rows=[read(f'{domain}_{name}.json') for name in names];assert len(rows[0])==len(rows[1])==n and rows[0].keys()==rows[1].keys()
        vals=[[correct(row[k]) for k in sorted(row)] for row in rows]
        close(st.mean(vals[0]),s['retention'][domain]['baseline']);close(st.mean(vals[1]),s['retention'][domain]['candidate'])
        diff,se=paired(vals[1],vals[0]);close(diff,s['retention'][domain]['paired']['difference']);close(se,s['retention'][domain]['paired']['se'])
    losses=[read(f'loss_final_{name}.json') for name in names]
    assert losses[0]['source_sha256']==losses[1]['source_sha256']
    assert len(losses[0]['rows'])==len(losses[1]['rows'])==296
    assert [r['index'] for r in losses[0]['rows']]==[r['index'] for r in losses[1]['rows']]
    diff,se=paired([r['loss'] for r in losses[1]['rows']],[r['loss'] for r in losses[0]['rows']])
    close(diff,s['retention']['nll296']['paired']['difference']);close(se,s['retention']['nll296']['paired']['se'])
    if s['retention']['nll296']['paired'].get('lower_is_better'):
        deltas=[x['loss']-y['loss'] for x,y in zip(losses[1]['rows'],losses[0]['rows'])]
        assert s['retention']['nll296']['paired']['better']==sum(x<0 for x in deltas)
        assert s['retention']['nll296']['paired']['worse']==sum(x>0 for x in deltas)
    # Different RNG and sampling implementation from the report: confirm the
    # direction with an independent10000-resample, question-cluster bootstrap.
    d=[vectors[a][t['id']]['score']-vectors[b][t['id']]['score'] for t in tasks]
    rng=random.Random(20261028);boot=sorted(sum(rng.choices(d,k=48))/48 for _ in range(10000))
    interval=[boot[249],boot[9749]]
    assert (interval[0]>0)==(s['pairs']['score']['ci95'][0]>0)
    result=dict(passed=True,chat_raw_responses_checked=288,question_clusters=48,nonidentity_clusters=36,
        independent_means_and_standard_errors_match=True,math197_and48_checked=True,nll296_aligned=True,
        independent_content_gain_ci95=interval,independent_bootstrap_direction_matches=True)
    (OUT/'independent_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(result,flush=True)

if __name__=='__main__':main()
