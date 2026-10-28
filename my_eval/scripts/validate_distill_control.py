"""Independent NumPy matrix check of the published task-3 means and paired SEs."""
import hashlib
import json
from pathlib import Path

import numpy as np

P=Path(__file__).resolve().parents[2]
OUT=P/'my_eval/results/distill_control_20261025'
read=lambda name:json.loads((OUT/name).read_text(encoding='utf-8'))
s=read('summary.json');plan=read('plan.json')
maths=read('math_final.json');chats=read('chat_final.json')['results'];losses=read('loss_final.json');diag=read('math_diagnostics.json')
mi=np.array(plan['primary_math_indices']);ci=np.array(plan['primary_chat_indices'])
vectors={};checks=0
for n,reported in s['models'].items():
    m=np.array([r['all'] for r in maths[n]],dtype=float).reshape(200,2).mean(axis=1)*100
    w=np.array([r['whole_numbers'] for r in diag[n]],dtype=float).reshape(200,2).mean(axis=1)*100
    t=np.array([r['strict_tool_trace'] for r in diag[n]],dtype=float).reshape(200,2).mean(axis=1)*100
    rm=np.array([r['rm'] for r in chats[n]],dtype=float).reshape(30,3).mean(axis=1)
    total=np.array([r['total'] for r in chats[n]],dtype=float).reshape(30,3).mean(axis=1)
    language=np.array(losses[n]['losses'])
    vectors[n]={'math197_pp':m[mi],'math200_pp':m,'whole_number197_pp':w[mi],'strict_tool197_pp':t[mi],
        'loss296':language,'chat29_rm':rm[ci],'chat30_rm':rm,'chat29_total':total[ci],'chat30_total':total}
    fields={'math197_percent':m[mi].mean(),'math200_percent':m.mean(),'loss296':language.mean(),
        'chat29_rm':rm[ci].mean(),'chat30_rm':rm.mean(),'chat29_total':total[ci].mean(),'chat30_total':total.mean()}
    for key,value in fields.items():
        assert abs(value-reported[key])<1e-10,(n,key,value,reported[key])
        checks+=1
paired_checks=0
for comparison,metrics in s['comparisons'].items():
    a,b=comparison.split('_minus_')
    for key,r in metrics.items():
        delta=vectors[a][key]-vectors[b][key]
        direction=-1 if key=='loss296' else 1
        expected={'n':len(delta),'difference':float(delta.mean()),'se':float(delta.std(ddof=1)/np.sqrt(len(delta))),
            'better':int(np.sum(direction*delta>1e-12)),'worse':int(np.sum(direction*delta< -1e-12)),
            'same':int(np.sum(np.abs(delta)<=1e-12))}
        for field,value in expected.items():assert abs(value-r[field])<1e-10,(comparison,key,field,value,r[field])
        paired_checks+=1
lock=read('evaluation_lock.json')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
for n,h in lock['weight_sha256'].items():assert sha(P/f'out/{n}_768.pth')==h,n
for filename,h in lock['files_sha256'].items():assert sha(P/filename)==h,filename
for filename,h in s['artifacts_sha256'].items():assert sha(P/filename)==h,filename
result={'all_passed':True,'independent_model_mean_checks':checks,'independent_paired_metric_checks':paired_checks,
    'method':'NumPy reshape200x2/30x3, sample standard deviation ddof1; all published means, SEs and direction counts agree within1e-10.',
    'weight_and_protocol_hashes_unchanged':True,'summary_sha256':sha(OUT/'summary.json')}
(OUT/'statistics_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
