"""Check the archived evidence using only the Python standard library."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics as st
import unicodedata

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
def content_key(task,row,length=16):
    return hashlib.sha256(json.dumps([task['messages'],row['final'],row['eligible'],row.get('failure')],ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:length]
def math_pass(row,reviews):
    status=row['grade']['status']
    if status=='review':status=reviews[hashlib.sha256((row['question']+'\n'+row['final']).encode()).hexdigest()]['decision']
    assert status in ['pass','fail'];return status=='pass'
def only_format_differs(answer,expected):
    keep=lambda text:[c for c in text.casefold() if unicodedata.category(c)[0] not in 'PZC']
    return keep(answer)==keep(expected)

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
            raw=read(folder/f'{domain}_{name}.json');assert len(raw)==n
            close(100*st.mean(math_pass(row,math_reviews) for row in raw.values()),summary['retention'][domain][label])
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

    folder=RESULTS/'strong_teacher_20261027';summary=read(folder/'summary.json');tasks=read(folder/'final_tasks.json')
    assert len(tasks)==96 and summary['accepted'] and all(summary['gates'].values())
    assert not summary['evidence_larger_teacher_beats_matched_smaller'],'v5 passing acceptance does not show that the larger teacher is better.'
    base,weak,strong='chat_repair_id_e3','strong_teacher_weak_e2','strong_teacher_strong_e2';names=[base,weak,strong]
    assert read(folder/'selection.json')['selected']==dict(weak=weak,strong=strong)
    screening=read(folder/'teacher_screening.json');audit=read(folder/'matched_data_audit.json');accepted={}
    for arm,n in [('weak',605),('strong',874)]:
        accepted[arm]={key.split('::')[1] for key,row in screening.items() if key.startswith(arm+'::') and row['accepted']}
        assert len(accepted[arm])==n==audit['teacher_pass_counts'][arm]
    assert accepted['weak']&accepted['strong']==set(audit['common_ids']) and audit['common_cases']==561
    exact={row['id']:row['exact_answer'] for row in read(folder/'train_items.json') if row.get('exact_answer') is not None};assert len(exact)==500
    for arm,n in [('weak',151),('strong',48)]:
        rows=[json.loads(line) for line in (folder/f'teacher_{arm}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        assert sum(r['id'] in exact and r['id'] not in accepted[arm] and only_format_differs(r['answer'],exact[r['id']]) for r in rows)==n
    checks['teacher_screening_counts_and_matched_561_questions']=True

    reviews=read(folder/'content_reviews.json');vectors={};usable={}
    for name in names:
        raw=read(folder/f'finalchat_{name}.json');assert len(raw)==288;scores=[];vectors[name]=[];usable[name]=[]
        for task in tasks:
            ss=[]
            for condition in ['no_tools','six_original','six_reworded']:
                row=raw[task['id']+'_'+condition];score=reviews[content_key(task,row,64)]['score'];assert score in [0,1,2]
                if not row['eligible']:assert score<2
                ss.append(score)
            scores+=ss;vectors[name].append(st.mean(ss));usable[name].append(100*st.mean(s==2 for s in ss))
        close(st.mean(scores),summary['models'][name]['mean_score'])
        close(100*st.mean(s==2 for s in scores),summary['models'][name]['usable_percent'])
        for category,values in summary['models'][name]['categories'].items():
            close(st.mean(u for u,task in zip(usable[name],tasks) if task['category']==category),values['usable_percent'])
    for label,a,b in [('strong_minus_v4',strong,base),('strong_minus_weak',strong,weak),('weak_minus_v4',weak,base)]:
        for metric,values in [('score',vectors),('usable',usable)]:
            difference,se=paired_se(values[a],values[b])
            close(difference,summary['comparisons'][label][metric]['difference']);close(se,summary['comparisons'][label][metric]['se'])
    other=[i for i,task in enumerate(tasks) if task['category']!='instruction'];assert len(other)==72
    difference,se=paired_se([usable[strong][i] for i in other],[usable[base][i] for i in other])
    assert round(difference,2)==7.87 and round(se,2)==3.98
    checks['v5_chat_864_raw_answers_and_paired_comparisons']=True

    math_reviews=read(folder/'math_reviews.json')
    for domain,n in [('historical',197),('freshmath',48)]:
        for name in names:
            raw=read(folder/f'{domain}_{name}.json');assert len(raw)==n
            close(100*st.mean(math_pass(row,math_reviews) for row in raw.values()),summary['retention'][domain]['means'][name])
    loss={name:read(folder/f'loss_final_{name}.json')['rows'] for name in names}
    for name in names:
        assert len(loss[name])==296 and [r['index'] for r in loss[name]]==[r['index'] for r in loss[base]]
        close(st.mean(r['loss'] for r in loss[name]),summary['retention']['nll296']['means'][name])
    difference,se=paired_se([r['loss'] for r in loss[strong]],[r['loss'] for r in loss[base]])
    close(difference,summary['retention']['nll296']['strong_minus_v4']['difference']);close(se,summary['retention']['nll296']['strong_minus_v4']['se'])
    identity=read(folder/'identity_tasks.json');assert len(identity)==14
    for name in names:
        raw=read(folder/f'identity_{name}.json');assert len(raw)==14
        assert sum(reviews[content_key(task,raw[task['id']],64)]['score']==2 for task in identity)==summary['identity14'][name]
    checks['v5_retention_math_nll296_and_identity14']=True
    assert read(folder/'demo_export.json')['all_parameter_tensors_exact'] and read(folder/'demo_ui_validation.json')['passed']
    checks['v5_recorded_export_and_ui_checks']=True
    return checks

if __name__=='__main__':
    print(json.dumps(dict(passed=True,checks=verify()),ensure_ascii=False,indent=2))
