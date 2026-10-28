"""Single LoRA training recipe on the exact v2 base, with KL retention and audited tokens."""
import argparse
import math
import time
import tool_choice_round2 as e
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from model.model_lora import apply_lora


def tokenize_rows():
    tok=e.load_tok();plan=e.read(e.OUT/'plan.json');maximum=plan['training']['max_seq']
    assert e.sha(e.OUT/'train.jsonl')==plan['data_sha256']['train.jsonl']
    bos=tok('<|im_start|>assistant\n',add_special_tokens=False).input_ids
    eos=tok('<|im_end|>\n',add_special_tokens=False).input_ids
    data=[];excluded=[];lengths=[]
    for i,row in enumerate(e.rows(e.OUT/'train.jsonl')):
        messages=e.copy.deepcopy(row['messages'])
        for message in messages:
            if isinstance(message.get('tool_calls'),str) and message['tool_calls']:
                message['tool_calls']=e.json.loads(message['tool_calls'])
        text=tok.apply_chat_template(messages,tools=row['tools'],tokenize=False,add_generation_prompt=False)
        ids=tok(text,add_special_tokens=False).input_ids
        if len(ids)>maximum:
            excluded.append(dict(index=i,kind=row['kind'],tokens=len(ids)));continue
        labels=[-100]*len(ids);j=0;spans=0
        while j<len(ids):
            if ids[j:j+len(bos)]==bos:
                start=j+len(bos);end=start
                while end<len(ids) and ids[end:end+len(eos)]!=eos:end+=1
                assert end<len(ids),('Missing assistant end',i)
                labels[start:end+len(eos)]=ids[start:end+len(eos)];j=end+len(eos);spans+=1
            else:j+=1
        assert spans==sum(m['role']=='assistant' for m in row['messages'])
        assert sum(x!=-100 for x in labels)>0
        data.append(dict(ids=ids,labels=labels,retain=row['kind']=='retain',source_index=i,math=row['kind']=='supervised' and row['source_kind']=='math'));lengths.append(len(ids))
    e.save('token_audit.json',dict(total_rows=len(lengths)+len(excluded),retained_rows=len(data),excluded=excluded,
        max_length=max(lengths),mean_length=sum(lengths)/len(lengths),no_silent_truncation=True,
        all_assistant_spans_supervised=True,train_sha256=e.sha(e.OUT/'train.jsonl')))
    torch.save(data,e.OUT/'tokenized.pt')
    print('Token audit:',len(data),'retained;',len(excluded),'too long; max',max(lengths),flush=True)


class Tokens(Dataset):
    def __init__(self):self.data=torch.load(e.OUT/'tokenized.pt',weights_only=True)
    def __len__(self):return len(self.data)
    def __getitem__(self,i):return self.data[i]


def collate(batch):
    n=max(len(r['ids']) for r in batch)
    x=torch.zeros(len(batch),n,dtype=torch.long);y=torch.full_like(x,-100)
    for i,row in enumerate(batch):
        x[i,:len(row['ids'])]=torch.tensor(row['ids']);y[i,:len(row['labels'])]=torch.tensor(row['labels'])
    return x,y,torch.tensor([r['retain'] for r in batch]),torch.tensor([r['math'] for r in batch])


def run(trial=False):
    assert not (e.OUT/'final_lock.json').exists(),'No training after final evaluation begins'
    plan=e.read(e.OUT/'plan.json');s=plan['training'];assert e.sha(e.P/f'out/{e.BASE}_768.pth')==plan['base_sha256']
    if not (e.OUT/'tokenized.pt').exists():tokenize_rows()
    torch.manual_seed(s['seed']);torch.cuda.manual_seed_all(s['seed'])
    student=e.load_model(fp32=True);teacher=e.load_model(fp32=True).requires_grad_(False)
    reference=torch.load(e.P/f'out/{e.BASE}_768.pth',map_location='cpu',weights_only=True)
    assert student.state_dict().keys()==reference.keys()
    assert all(torch.equal(v.detach().cpu().half(),reference[k].half()) for k,v in student.state_dict().items()),'HF export differs from intended exact LoRA base'
    del reference
    apply_lora(student,rank=s['rank'])
    params=[]
    for name,p in student.named_parameters():
        p.requires_grad_('.lora.' in name)
        if p.requires_grad:params.append(p)
    assert sum(p.numel() for p in params)==393216
    data=Tokens();epochs=1 if trial else s['epochs'];batches=math.ceil(len(data)/s['batch'])
    optimizer=torch.optim.AdamW(params,lr=s['lr']);started=time.perf_counter();steps=0;history=[]
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        generator=torch.Generator().manual_seed(s['seed']+epoch)
        loader=DataLoader(data,batch_size=s['batch'],shuffle=True,num_workers=2,pin_memory=True,
            collate_fn=collate,generator=generator)
        student.train()
        for step,(x,y,retain,math_rows) in enumerate(loader,1):
            x,y,retain,math_rows=x.cuda(),y.cuda(),retain.cuda(),math_rows.cuda()
            mask=y[:,1:]!=-100;labels=y[:,1:]
            optimizer.zero_grad(set_to_none=True)
            progress=(epoch*batches+step)/(s['epochs']*batches)
            lr=s['lr']*(.1+.9*.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:group['lr']=lr
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                t=teacher(x[retain],use_cache=False).logits[:,:-1] if retain.any() else None
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=student(x,use_cache=False).logits[:,:-1]
            ce=logits[0,0,0].float()*0;kl=ce
            if (~retain).any():
                lp=logits[~retain].float();target=labels[~retain]
                token_ce=F.cross_entropy(lp.reshape(-1,lp.shape[-1]),target.reshape(-1),ignore_index=-100,reduction='none').reshape_as(target)
                valid_ce=mask[~retain]
                weights_ce=(1+2*(valid_ce.cumsum(-1)<=64)*math_rows[~retain,None]).float()*valid_ce
                ce=((token_ce*weights_ce).sum(-1)/weights_ce.sum(-1).clamp_min(1)).mean()
            if t is not None:
                valid=mask[retain]
                slog=F.log_softmax(logits[retain][valid].float(),-1)
                tprob=F.softmax(t[valid].float(),-1)
                # Per-example KL, first32 answer tokens weighted3 as in identity retention.
                token_kl=F.kl_div(slog,tprob,reduction='none').sum(-1)
                weights=(1+2*(valid.cumsum(-1)<=32)).float()*valid
                which=torch.arange(valid.shape[0],device='cuda')[:,None].expand_as(valid)[valid]
                per_row=torch.zeros(valid.shape[0],device='cuda').scatter_add(0,which,token_kl*weights[valid])
                kl=(per_row/weights.sum(-1).clamp_min(1)).mean()
            loss=ce+s['kl']*kl
            assert torch.isfinite(loss),'Nonfinite loss'
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.)
            assert torch.isfinite(grad),'Nonfinite gradients'
            optimizer.step();steps+=1
            if steps%50==0 or step==batches or (trial and steps==16):
                record=dict(epoch=epoch+1,step=steps,ce=ce.item(),kl=kl.item(),loss=loss.item(),lr=lr)
                history.append(record);print(record,flush=True)
            del x,y,logits,t,loss,ce,kl
            if trial and steps>=16:break
        name=f'tool_choice2_trial' if trial else f'tool_choice2_e{epoch+1}'
        path=e.P/f'out/{name}_768.pth';assert not path.exists(),f'Preserve existing {name}'
        adapter={k:p.detach().cpu().half() for k,p in student.state_dict().items() if '.lora.' in k}
        merged={k:p.detach().cpu().half().clone() for k,p in student.state_dict().items() if '.lora.' not in k}
        for key in list(merged):
            if key.endswith('.weight'):
                prefix=key[:-7]
                if prefix+'.lora.A.weight' in adapter:
                    delta=adapter[prefix+'.lora.B.weight'].float()@adapter[prefix+'.lora.A.weight'].float()
                    merged[key]=(merged[key].float()+delta).half()
        assert all(torch.isfinite(p).all() for p in merged.values())
        torch.save(merged,path);torch.save(adapter,e.P/f'out/{name}_lora_768.pth')
        assert all(p.grad is None for p in teacher.parameters())
        assert all(p.grad is None for k,p in student.named_parameters() if '.lora.' not in k)
        e.save(f'train_{name}.json',dict(name=name,base=e.BASE,base_sha256=plan['base_sha256'],trial=trial,epoch=epoch+1,
            steps=steps,seconds=time.perf_counter()-started,settings=s,history=history,
            trainable_parameters=sum(p.numel() for p in params),all_saved_parameters_finite=True,
            teacher_and_base_frozen=True,weight_sha256=e.sha(path),adapter_base=e.BASE,
            base_export_parameters_exact=True,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            source_sha256=e.sha(__file__),tokenized_sha256=e.sha(e.OUT/'tokenized.pt')))
    print('Training completed',time.perf_counter()-started,'seconds',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['tokens','trial','train']);a=ap.parse_args()
    if a.stage=='tokens':tokenize_rows()
    else:run(a.stage=='trial')
