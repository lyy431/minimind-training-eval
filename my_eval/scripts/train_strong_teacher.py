"""Matched full-parameter student runs from exact v4; frozen-v4 KL retention."""
import argparse
import copy
import json
import math
import time
import strong_teacher_common as e
e.configure_local()
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def verify(arm):
    assert not (e.OUT/'final_lock.json').exists(),'Training is forbidden after formal freeze'
    lock=e.read(e.OUT/'training_lock.json');assert e.sha(e.P/f'out/{e.BASE}_768.pth')==lock['base_sha256']
    assert e.sha(e.OUT/f'train_{arm}.jsonl')==lock['train_sha256'][arm]
    for name,h in lock['files_sha256'].items():assert e.sha(e.OUT/name)==h,name
    for name,h in lock['source_sha256'].items():assert e.sha(e.P/'my_eval/scripts'/name)==h,name

def tokenize(arm):
    verify(arm);tok=e.tokenizer();limit=e.read(e.OUT/'plan.json')['training']['max_seq'];data=[]
    bos=tok('<|im_start|>assistant\n',add_special_tokens=False).input_ids;eos=tok('<|im_end|>\n',add_special_tokens=False).input_ids
    for i,r in enumerate(e.rows(e.OUT/f'train_{arm}.jsonl')):
        messages=copy.deepcopy(r['messages'])
        for m in messages:
            if isinstance(m.get('tool_calls'),str) and m['tool_calls']:m['tool_calls']=json.loads(m['tool_calls'])
        ids=tok(tok.apply_chat_template(messages,tools=r['tools'],tokenize=False,add_generation_prompt=False),add_special_tokens=False).input_ids
        assert len(ids)<=limit,(i,len(ids),'No silent truncation')
        labels=[-100]*len(ids);j=0;spans=0
        while j<len(ids):
            if ids[j:j+len(bos)]==bos:
                start=j+len(bos);end=start
                while end<len(ids) and ids[end:end+len(eos)]!=eos:end+=1
                assert end<len(ids),(i,'Missing assistant end')
                labels[start:end+len(eos)]=ids[start:end+len(eos)];j=end+len(eos);spans+=1
            else:j+=1
        assert spans==sum(m['role']=='assistant' for m in messages)
        assert any(v!=-100 for v in labels)
        data.append(dict(ids=ids,labels=labels,retain=r['kind']=='retain',math=r['kind']=='supervised' and r['source_kind']=='math',identity=r['source_kind']=='identity'))
    torch.save(data,e.OUT/f'tokenized_{arm}.pt')
    e.save(f'token_audit_{arm}.json',dict(rows=len(data),max_length=max(len(r['ids']) for r in data),answer_tokens=sum(sum(v!=-100 for v in r['labels']) for r in data),no_truncation=True,prompt_and_tool_results_masked=True,train_sha256=e.sha(e.OUT/f'train_{arm}.jsonl')))
    return data

def collate(batch):
    x=torch.zeros(len(batch),max(len(r['ids']) for r in batch),dtype=torch.long);y=torch.full_like(x,-100)
    for i,r in enumerate(batch):x[i,:len(r['ids'])]=torch.tensor(r['ids']);y[i,:len(r['labels'])]=torch.tensor(r['labels'])
    return x,y,*[torch.tensor([r[k] for r in batch]) for k in ['retain','math','identity']]

def train(arm,trial):
    verify(arm);s=e.read(e.OUT/'plan.json')['training'];data=tokenize(arm)
    torch.manual_seed(s['seed']);torch.cuda.manual_seed_all(s['seed'])
    student=e.load(fp32=True);anchor=e.load(fp32=True).requires_grad_(False)
    params=list(student.parameters());optimizer=torch.optim.AdamW(params,lr=s['lr'])
    steps=0;history=[];start=time.perf_counter();batches=math.ceil(len(data)/s['batch']);torch.cuda.reset_peak_memory_stats()
    for epoch in range(1 if trial else s['epochs']):
        loader=DataLoader(data,batch_size=s['batch'],shuffle=True,num_workers=0,pin_memory=True,collate_fn=collate,generator=torch.Generator().manual_seed(s['seed']+epoch))
        student.train()
        for i,batch in enumerate(loader,1):
            x,y,retain,maths,identity=[v.cuda() for v in batch];mask=y[:,1:]!=-100;labels=y[:,1:]
            optimizer.zero_grad(set_to_none=True);progress=(epoch*batches+i)/(s['epochs']*batches)
            lr=s['lr']*(.1+.9*.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:group['lr']=lr
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):t=anchor(x[retain],use_cache=False).logits[:,:-1] if retain.any() else None
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=student(x,use_cache=False).logits[:,:-1]
            ce=logits[0,0,0].float()*0;kl=ce
            if (~retain).any():
                m=mask[~retain];target=labels[~retain];lp=logits[~retain].float()
                losses=F.cross_entropy(lp.reshape(-1,lp.shape[-1]),target.reshape(-1),ignore_index=-100,reduction='none').reshape_as(target)
                weights=m.float()*(1+(m.cumsum(-1)<=s['math_first_tokens'])*maths[~retain,None])
                row_weights=1+(s['identity_weight']-1)*identity[~retain].float()
                ce=(((losses*weights).sum(-1)/weights.sum(-1).clamp_min(1))*row_weights).mean()
            if t is not None:
                m=mask[retain];sl=F.log_softmax(logits[retain][m].float(),-1);tp=F.softmax(t[m].float(),-1)
                losses=F.kl_div(sl,tp,reduction='none').sum(-1);which=torch.arange(m.shape[0],device='cuda')[:,None].expand_as(m)[m]
                kl=(torch.zeros(m.shape[0],device='cuda').scatter_add(0,which,losses)/m.sum(-1).clamp_min(1)).mean()
            loss=ce+s['kl']*kl;assert torch.isfinite(loss);loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(params,1.));optimizer.step();steps+=1
            if steps%50==0 or i==batches or (trial and steps==16):
                rec=dict(epoch=epoch+1,step=steps,ce=ce.item(),kl=kl.item(),lr=lr);history.append(rec);print(arm,rec,flush=True)
            del logits,t,loss,ce,kl,x,y
            if trial and steps>=16:break
        name=f'strong_teacher_{arm}_'+('trial' if trial else f'e{epoch+1}')
        path=e.P/f'out/{name}_768.pth';assert not path.exists(),'Preserve all existing checkpoints'
        state={k:v.detach().cpu().half().clone() for k,v in student.state_dict().items()};assert all(torch.isfinite(v).all() for v in state.values())
        torch.save(state,path);assert all(v.grad is None for v in anchor.parameters())
        e.save(f'train_{name}.json',dict(base=e.BASE,base_sha256=e.BASE_SHA,arm=arm,epoch=epoch+1,steps=steps,trial=trial,settings=s,
            weight_sha256=e.sha(path),seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            train_sha256=e.sha(e.OUT/f'train_{arm}.jsonl'),source_sha256=e.sha(__file__),history=history))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['weak','strong']);p.add_argument('stage',choices=['tokens','trial','train']);a=p.parse_args()
    tokenize(a.arm) if a.stage=='tokens' else train(a.arm,a.stage=='trial')
