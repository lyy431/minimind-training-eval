"""Evaluate the two completed controls sequentially, with restart-safe stages."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

P=Path(__file__).resolve().parents[2]
OUT=P/'my_eval/results/distill_control_20261025'
os.environ.update(PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',HF_HOME='D:/conda/hf-cache',
    HF_DATASETS_CACHE=str(P/'my_eval/runtime/datasets'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    TEMP=str(P/'my_eval/runtime'),TMP=str(P/'my_eval/runtime'),TOKENIZERS_PARALLELISM='false')

for name in ['bb_sft_replay','bb_sft_new']:
    assert (OUT/f'train_{name}.json').exists(), f'Finish both full training runs first: {name}'
record_path=OUT/'evaluation_stage_runs.json'
record=json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else []
stages=[('eval_distill_control.py',['lock']),('eval_distill_control.py',['replay']),
    ('eval_distill_control.py',['import_history'])]
stages += [('eval_distill_control.py',['loss','--model',n]) for n in ['bb_dist','bb_sft_replay','bb_sft_new','bb_rl']]
stages += [('eval_distill_control.py',['math','--model',n]) for n in ['bb_sft_replay','bb_sft_new']]
stages += [('eval_distill_control.py',['chat','--model',n]) for n in ['bb_sft_replay','bb_sft_new']]
stages += [('write_distill_control_report.py',[])]
for script,args in stages:
    print('Running '+script+' '+' '.join(args),flush=True)
    started=time.perf_counter()
    result=subprocess.run([sys.executable,'-u',str(P/'my_eval/scripts'/script),*args],cwd=P)
    record.append({'script':script,'args':args,'seconds':time.perf_counter()-started,'exit_code':result.returncode})
    record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    if result.returncode:raise SystemExit(result.returncode)
print('Both fixed controls fully evaluated; numerical report ready for interpretation.',flush=True)
