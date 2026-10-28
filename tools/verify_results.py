"""Check the archived evidence using only the Python standard library."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'my_eval/results'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def close(a,b):assert abs(a-b)<1e-9,(a,b)
def paired_se(a,b):
    d=[x-y for x,y in zip(a,b)];assert len(a)==len(b)>1
    return st.mean(d),st.stdev(d)/len(d)**.5
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def content_key(task,row):
    return hashlib.sha256(json.dumps([task['messages'],row['final'],row['eligible'],row.get('failure')],ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:16]

def verify():
    checks={};manifest=read(ROOT/'SNAPSHOT_MANIFEST.json')
    for row in manifest['files']:
        file=ROOT/row['path'];assert file.is_file(),row['path']
        assert file.stat().st_size==row['bytes'] and sha(file)==row['sha256'],row['path']
    checks['copied_file_bytes_and_sha256']=len(manifest['files'])

    folder=RESULTS/'strict_math_20261026';summary=read(folder/'summary.json');raw=read(folder/'final_results.json')
    for model,rows in raw.items():
        chosen=[r for r in rows if r['problem'] not in [15,125,181]]
        assert len(rows)==400 and len(chosen)==394
        assert len({(r['problem'],r['sample']) for r in chosen})==394
        close(100*st.mean(r['final_pass'] for r in chosen),summary['models'][model]['strict197'])
    checks['strict_math_four_models_394_runs_each']=True

    folder=RESULTS/'best_of_n_20261026';summary=read(folder/'summary.json');raw=read(folder/'final_outcomes.json')['math'];assert len(raw)==197
    for n in ['1','2','4','eligible_first']:
        close(st.mean(r['scores'][n] for r in raw),summary['math']['means'][n])
        for r in raw:
            index=r['selected'][n]
            assert index is None or 0<=index<4
            if index is not None:assert r['scores'][n]==100*r['correct'][index]
    difference,se=paired_se([r['scores']['4'] for r in raw],[r['scores']['1'] for r in raw])
    close(difference,summary['math']['paired']['4_minus_1']['difference']);close(se,summary['math']['paired']['4_minus_1']['se'])
    checks['best_of_n_means_and_paired_standard_error']=True

    folder=RESULTS/'chat_repair_20261027';summary=read(folder/'summary.json');tasks=read(folder/'final_tasks.json')
    assert len(tasks)==48 and summary['accepted']
    reviews=read(folder/'content_reviews.json');vectors={};usable={}
    for name in [summary['base'],summary['candidate']]:
        raw=read(folder/f'finalchat_{name}.json');assert len(raw)==144;scores=[];vectors[name]=[];usable[name]=[]
        for task in tasks:
            ss=[]
            for condition in ['no_tools','six_original','six_reworded']:
                row=raw[task['id']+'_'+condition];score=reviews[content_key(task,row)]['score'];assert score in [0,1,2]
                if not row['eligible']:assert score<2
                ss.append(score)
            scores+=ss;vectors[name].append(st.mean(ss));usable[name].append(100*st.mean(s==2 for s in ss))
        close(st.mean(scores),summary['models'][name]['mean_score'])
        close(100*st.mean(s==2 for s in scores),summary['models'][name]['usable_percent'])
    for metric,values in [('score',vectors),('usable',usable)]:
        difference,se=paired_se(values[summary['candidate']],values[summary['base']])
        close(difference,summary['pairs'][metric]['difference']);close(se,summary['pairs'][metric]['se'])
    checks['v4_chat_288_raw_answers_and_question_cluster_se']=True

    math_reviews=read(folder/'math_reviews.json')
    for domain,n in [('historical',197),('freshmath',48)]:
        for name,label in [(summary['base'],'baseline'),(summary['candidate'],'candidate')]:
            raw=read(folder/f'{domain}_{name}.json');assert len(raw)==n;answers=[]
            for row in raw.values():
                status=row['grade']['status']
                if status=='review':
                    key=hashlib.sha256((row['question']+'\n'+row['final']).encode()).hexdigest();status=math_reviews[key]['decision']
                assert status in ['pass','fail'];answers.append(status=='pass')
            close(100*st.mean(answers),summary['retention'][domain][label])
    loss=[]
    for name,label in [(summary['base'],'baseline'),(summary['candidate'],'candidate')]:
        rows=read(folder/f'loss_final_{name}.json')['rows'];assert len(rows)==296;loss.append(rows)
        close(st.mean(r['loss'] for r in rows),summary['retention']['nll296'][label])
    assert [r['index'] for r in loss[0]]==[r['index'] for r in loss[1]]
    difference,se=paired_se([r['loss'] for r in loss[1]],[r['loss'] for r in loss[0]])
    close(difference,summary['retention']['nll296']['paired']['difference']);close(se,summary['retention']['nll296']['paired']['se'])
    assert difference>0,'The loss deterioration must not be described as unchanged.'
    checks['v4_retention_math_and_aligned_nll296']=True
    routing=read(RESULTS/'tool_choice_round2_20261026/summary.json');assert routing['accepted'] and all(routing['gates'].values())
    checks['v3_recorded_acceptance_gates']=True
    return checks

if __name__=='__main__':
    print(json.dumps(dict(passed=True,checks=verify()),ensure_ascii=False,indent=2))
