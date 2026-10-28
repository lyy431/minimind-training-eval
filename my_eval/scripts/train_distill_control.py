"""Run the unmodified full-SFT trainer for a fixed task-3 control arm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

P=Path(__file__).resolve().parents[2]
OUT=P/'my_eval/results/distill_control_20261025'
RUNTIME=P/'my_eval/runtime'
os.environ.update(HF_HOME='D:/conda/hf-cache',HF_DATASETS_CACHE=str(RUNTIME/'datasets'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TEMP=str(RUNTIME),TMP=str(RUNTIME),
    TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
sys.path.insert(0,str(P))


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('arm',choices=['bb_sft_replay','bb_sft_new'])
    ap.add_argument('--trial',action='store_true')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    assert not any(OUT.glob('*_final.json')), 'No training after final evaluation begins'
    audit=json.loads((OUT/'data_audit.json').read_text(encoding='utf-8'))
    plan=json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
    for name,digest in audit['protocol_sha256'].items(): assert sha(P/name)==digest,name
    assert sha(P/'out/bb_dist_768.pth')==audit['base_sha256']
    file='replay_mix.jsonl' if args.arm=='bb_sft_replay' else 'new_mix.jsonl'
    data=OUT/file
    assert sha(data)==audit['dataset_sha256'][args.arm]
    name=args.arm+('_trial' if args.trial else '')
    if args.trial:
        data=OUT/'sft_trial128.jsonl'
        text='\n'.join((OUT/file).read_text(encoding='utf-8').splitlines()[:128])+'\n'
        if data.exists(): assert data.read_text(encoding='utf-8')==text
        else: data.write_text(text,encoding='utf-8',newline='\n')
    assert not (P/f'out/{name}_768.pth').exists() or args.resume,'Existing weights preserved; use --resume only for this named run'
    import datasets
    import torch
    os.chdir(P/'trainer')
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    settings=plan['training']
    sys.argv=['train_full_sft.py','--from_weight','bb_dist','--save_weight',name,'--data_path',str(data),
        '--epochs','1' if args.trial else str(settings['epochs']),'--learning_rate',str(settings['learning_rate']),
        '--batch_size',str(settings['batch_size']),'--max_seq_len',str(settings['max_seq_len']),
        '--num_workers',str(settings['num_workers']),'--seed',str(settings['seed']),'--from_resume','1' if args.resume else '0',
        '--log_interval','50','--save_interval','250','--use_compile','0']
    runpy.run_path(str(P/'trainer/train_full_sft.py'),run_name='__main__')
    finished=time.perf_counter()
    saved=torch.load(P/f'out/{name}_768.pth',map_location='cpu',weights_only=True)
    initial=torch.load(P/'out/bb_dist_768.pth',map_location='cpu',weights_only=True)
    assert saved.keys()==initial.keys()
    assert all(torch.isfinite(t).all().item() for t in saved.values()), 'Non-finite saved parameters'
    changed=sum(not torch.equal(t,initial[k]) for k,t in saved.items())
    assert changed>0, 'Training did not change any parameters'
    record={'arm':args.arm,'name':name,'trial':args.trial,'resume':args.resume,'settings':settings,
        'data':str(data),'data_sha256':sha(data),'weight_sha256':sha(P/f'out/{name}_768.pth'),
        'seconds':finished-started,'all_saved_parameters_finite':True,'changed_state_tensors':changed,
        'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_gpu_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'argv':sys.argv}
    (OUT/f'train_{name}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
