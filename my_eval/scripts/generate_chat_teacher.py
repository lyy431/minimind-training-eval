"""Greedy, local Qwen labels; restartable batches, explicit incomplete-output audit."""
import argparse
import time
import chat_repair_common as e
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

PROMPT='你在为小型文本助手编写优质训练答案。直接回答用户，不要分析过程、不要复述问题、不要写客套开头。使用用户所用语言，通常一到两句，最多三句；如果问题过于复杂，给最重要的结论。信息不足时直说需要什么信息，不编造。不要调用工具，不使用XML标签，不自称Qwen或ChatGPT。'

def main(trial=False):
    plan=e.read(e.OUT/'plan.json');assert not (e.OUT/'final_lock.json').exists()
    requests=e.read(e.OUT/'teacher_requests.json');saved=e.read(e.OUT/'teacher_answers.json') if (e.OUT/'teacher_answers.json').exists() else {}
    path=plan['teacher'];tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left')
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16).cuda().eval()
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter();batch_size=8
    target=requests[:8] if trial else requests
    for start in range(0,len(target),batch_size):
        batch=target[start:start+batch_size]
        if all(r['id'] in saved for r in batch):continue
        texts=[tok.apply_chat_template([dict(role='system',content=PROMPT),dict(role='user',content=r['question'])],tokenize=False,add_generation_prompt=True) for r in batch]
        encoded=tok(texts,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda');n=encoded.input_ids.shape[1]
        assert n<=512,'Do not silently truncate teacher prompts'
        with torch.inference_mode():out=model.generate(**encoded,max_new_tokens=128,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        for row,tokens in zip(batch,out[:,n:]):
            ids=tokens.tolist();complete=tok.eos_token_id in ids
            text=tok.decode(tokens,skip_special_tokens=True).strip()
            valid=bool(complete and text and '<tool_call' not in text and '<think' not in text and not any(n in text.lower() for n in ['chatgpt','qwen','通义千问']))
            saved[row['id']]=dict(**row,answer=text,complete=complete,accepted=valid,generated_tokens=(ids.index(tok.eos_token_id)+1 if complete else len(ids)))
        e.save('teacher_answers.json',saved)
        print('teacher',min(start+len(batch),len(target)),'/',len(target),f'{time.perf_counter()-started:.1f}s','accepted',sum(x['accepted'] for x in saved.values()),flush=True)
    result=dict(trial=trial,seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        total_recorded=len(saved),accepted=sum(x['accepted'] for x in saved.values()),settings=dict(batch=8,max_new_tokens=128,do_sample=False,prompt=PROMPT),
        teacher_weight_sha256=e.sha(e.P.parent/'Qwen2.5-1.5B-Instruct/model.safetensors'),source_sha256=e.sha(__file__))
    e.save('teacher_trial.json' if trial else 'teacher_run.json',result);print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trial',action='store_true');a=p.parse_args();main(a.trial)
