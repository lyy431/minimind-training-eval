"""Portable, resumable Qwen generator. No downloads; accepts a local model directory.

The same source/settings are used for the local1.5B and remote7B arms. JSONL is
flushed after each completed batch. Partial or instruction-leaking outputs remain
in raw records and are never silently converted into training labels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n');tmp.replace(path)

def main(args):
    root=Path(args.output).resolve().parent;root.mkdir(parents=True,exist_ok=True)
    cache=root/'runtime';cache.mkdir(exist_ok=True)
    os.environ.update(HF_HOME=str(cache/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',TEMP=str(cache),TMP=str(cache))
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    settings=read(args.settings);requests=read(args.requests);model_dir=Path(args.model_dir).resolve()
    assert all(set(r)=={'id','question','messages','language','category'} for r in requests),'Unexpected private/reference fields in teacher request'
    assert len({r['id'] for r in requests})==len(requests)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert settings['batch']==2 and settings['do_sample'] is False and settings['dtype']=='bfloat16'
    files=sorted(p for p in model_dir.iterdir() if p.is_file() and (p.suffix in ['.safetensors','.json','.txt'] and p.name not in ['README.md']))
    assert any(p.suffix=='.safetensors' for p in files)
    model_hashes={p.name:sha(p) for p in files}
    fingerprint=dict(settings_sha256=sha(args.settings),requests_sha256=sha(args.requests),source_sha256=sha(__file__),model_sha256=model_hashes)
    manifest=Path(args.output).with_suffix('.manifest.json')
    if manifest.exists():assert read(manifest)['fingerprint']==fingerprint,'Do not mix runs with changed model/input/code/settings'
    else:atomic(manifest,dict(fingerprint=fingerprint,arm=args.arm,python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),gpu_gib=torch.cuda.get_device_properties(0).total_memory/2**30))
    output=Path(args.output);saved={}
    if output.exists():
        for line in output.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row=json.loads(line);assert row['id'] not in saved;saved[row['id']]=row
    request_map={r['id']:r for r in requests}
    assert set(saved)<=set(request_map)
    assert all(r['question']==request_map[k]['question'] for k,r in saved.items())
    target=requests[:args.limit] if args.limit else requests
    todo=[r for r in target if r['id'] not in saved]
    if not todo:print('Already complete:',len(target),flush=True);return
    torch.manual_seed(settings['seed']);torch.cuda.manual_seed_all(settings['seed'])
    start=time.perf_counter();tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,padding_side='left')
    model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').cuda().eval()
    assert tok.eos_token_id is not None and tok.pad_token_id is not None
    torch.cuda.reset_peak_memory_stats();load_seconds=time.perf_counter()-start;run_start=time.perf_counter();batch_size=settings['batch']
    with output.open('a',encoding='utf-8',newline='\n') as handle:
        for offset in range(0,len(todo),batch_size):
            if time.perf_counter()-start>args.max_seconds:print('Generation time limit; partial records retained.',flush=True);break
            batch=todo[offset:offset+batch_size]
            texts=[tok.apply_chat_template([dict(role='system',content=settings['system'])]+r['messages'],tokenize=False,add_generation_prompt=True) for r in batch]
            encoded=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda');width=encoded.input_ids.shape[1]
            assert width<=settings['max_input_tokens'],'Refuse silent prompt truncation'
            t=time.perf_counter()
            with torch.inference_mode():
                out=model.generate(**encoded,max_new_tokens=settings['max_new_tokens'],do_sample=False,
                    eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,use_cache=True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            for batch_index,(request,tokens) in enumerate(zip(batch,out[:,width:])):
                ids=tokens.tolist();complete=tok.eos_token_id in ids
                used=ids[:ids.index(tok.eos_token_id)+1] if complete else ids
                answer=tok.decode(used,skip_special_tokens=True).strip()
                structural=bool(complete and answer and not any(x in answer.casefold() for x in ['<tool_call','<think','chatgpt','qwen','通义千问']))
                row=dict(**request,answer=answer,complete=complete,structural_ok=structural,
                    generated_tokens=len(used),generation_seconds_batch=elapsed,input_tokens=int(encoded.attention_mask[batch_index].sum().item()))
                handle.write(json.dumps(row,ensure_ascii=False)+'\n');saved[row['id']]=row
            handle.flush();os.fsync(handle.fileno())
            if (offset//batch_size+1)%8==0 or offset+len(batch)==len(todo):
                print(json.dumps(dict(recorded=len(saved),target=len(target),seconds=round(time.perf_counter()-run_start,2),peak_gib=round(torch.cuda.max_memory_allocated()/2**30,3))),flush=True)
    atomic(output.with_suffix('.runtime.json'),dict(arm=args.arm,requested=len(target),recorded=len(saved),
        completed=sum(r['id'] in saved for r in target)==len(target),load_seconds=load_seconds,
        generation_seconds=time.perf_counter()-run_start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,settings=settings))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--requests',required=True);p.add_argument('--settings',required=True)
    p.add_argument('--model-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--arm',choices=['weak','strong'],required=True)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--max-seconds',type=int,default=7200)
    main(p.parse_args())
