"""Small full-parameter repair from exact v3, with frozen-v3 KL retention."""
import argparse
import math
import time
import chat_repair_common as e
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset

def tokenize():
    tok=e.tok();limit=e.read(e.OUT/'plan.json')['training']['max_seq'];data=[];excluded=[]
    bos=tok('<|im_start|>assistant\n',add_special_tokens=False).input_ids
    eos=tok('<|im_end|>\n',add_special_tokens=False).input_ids
    for i,r in enumerate(e.rows(e.OUT/'train.jsonl')):
        messages=e.copy.deepcopy(r['messages'])
        for message in messages:
            if isinstance(message.get('tool_calls'),str) and message['tool_calls']:
                message['tool_calls']=e.json.loads(message['tool_calls'])
        text=tok.apply_chat_template(messages,tools=r['tools'],tokenize=False,add_generation_prompt=False)
        ids=tok(text,add_special_tokens=False).input_ids
        if len(ids)>limit:excluded.append(dict(index=i,tokens=len(ids),kind=r['kind'],source_kind=r['source_kind']));continue
        labels=[-100]*len(ids);j=0;spans=0
        while j<len(ids):
            if ids[j:j+len(bos)]==bos:
                start=j+len(bos);end=start
                while end<len(ids) and ids[end:end+len(eos)]!=eos:end+=1
                assert end<len(ids),(i,'missing assistant end')
                labels[start:end+len(eos)]=ids[start:end+len(eos)];j=end+len(eos);spans+=1
            else:j+=1
        assert spans==sum(m['role']=='assistant' for m in r['messages'])
        assert any(y!=-100 for y in labels)
        data.append(dict(ids=ids,labels=labels,retain=r['kind']=='retain',math=r['kind']=='supervised' and r['source_kind']=='math',source_index=i))
    assert not excluded,('Explicitly resolve overlength rows rather than silently truncating',excluded[:10])
    torch.save(data,e.OUT/'tokenized.pt');e.save('token_audit.json',dict(rows=len(data),excluded=excluded,max_length=max(len(r['ids']) for r in data),no_silent_truncation=True,assistant_spans_only=True,train_sha256=e.sha(e.OUT/'train.jsonl')))
    print(e.read(e.OUT/'token_audit.json'),flush=True)

class Tokens(Dataset):
    def __init__(self):self.data=torch.load(e.OUT/'tokenized.pt',weights_only=True)
    def __len__(self):return len(self.data)
    def __getitem__(self,i):return self.data[i]

def collate(batch):
    x=torch.zeros(len(batch),max(len(r['ids']) for r in batch),dtype=torch.long);y=torch.full_like(x,-100)
    for i,r in enumerate(batch):x[i,:len(r['ids'])]=torch.tensor(r['ids']);y[i,:len(r['labels'])]=torch.tensor(r['labels'])
    return x,y,torch.tensor([r['retain'] for r in batch]),torch.tensor([r['math'] for r in batch])

def run(trial=False):
    assert not (e.OUT/'final_lock.json').exists(),'No training after final freeze'
    plan=e.read(e.OUT/'plan.json');s=plan['training'];assert e.sha(e.OUT/'train.jsonl')==plan['train_sha256']
    assert e.sha(e.P/f'out/{e.BASE}_768.pth')==plan['baseline_sha256']
    if not (e.OUT/'tokenized.pt').exists():tokenize()
    torch.manual_seed(s['seed']);torch.cuda.manual_seed_all(s['seed'])
    student=e.load(fp32=True);teacher=e.load(fp32=True).requires_grad_(False)
    original=torch.load(e.P/f'out/{e.BASE}_768.pth',map_location='cpu',weights_only=True)
    assert student.state_dict().keys()==original.keys()
    assert all(torch.equal(p.detach().cpu().half(),original[k].half()) for k,p in student.state_dict().items());del original
    params=list(student.parameters());assert all(p.requires_grad for p in params)
    optimizer=torch.optim.AdamW(params,lr=s['lr']);data=Tokens();batches=math.ceil(len(data)/s['batch'])
    started=time.perf_counter();steps=0;history=[];torch.cuda.reset_peak_memory_stats()
    for epoch in range(1 if trial else s['epochs']):
        loader=DataLoader(data,batch_size=s['batch'],shuffle=True,num_workers=s['num_workers'],pin_memory=True,
            collate_fn=collate,generator=torch.Generator().manual_seed(s['seed']+epoch))
        student.train()
        for i,(x,y,retain,maths) in enumerate(loader,1):
            x,y,retain,maths=x.cuda(),y.cuda(),retain.cuda(),maths.cuda();valid=y[:,1:]!=-100;labels=y[:,1:]
            optimizer.zero_grad(set_to_none=True);progress=(epoch*batches+i)/(s['epochs']*batches)
            lr=s['lr']*(.1+.9*.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups:group['lr']=lr
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):t=teacher(x[retain],use_cache=False).logits[:,:-1] if retain.any() else None
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=student(x,use_cache=False).logits[:,:-1]
            ce=logits[0,0,0].float()*0;kl=ce
            if (~retain).any():
                lp=logits[~retain].float();target=labels[~retain];mask=valid[~retain]
                losses=F.cross_entropy(lp.reshape(-1,lp.shape[-1]),target.reshape(-1),ignore_index=-100,reduction='none').reshape_as(target)
                weights=mask.float()*(1+(mask.cumsum(-1)<=64)*maths[~retain,None])
                ce=((losses*weights).sum(-1)/weights.sum(-1).clamp_min(1)).mean()
            if t is not None:
                mask=valid[retain];sl=F.log_softmax(logits[retain][mask].float(),-1);tp=F.softmax(t[mask].float(),-1)
                losses=F.kl_div(sl,tp,reduction='none').sum(-1);which=torch.arange(mask.shape[0],device='cuda')[:,None].expand_as(mask)[mask]
                perrow=torch.zeros(mask.shape[0],device='cuda').scatter_add(0,which,losses)
                kl=(perrow/mask.sum(-1).clamp_min(1)).mean()
            loss=ce+s['kl']*kl;assert torch.isfinite(loss)
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad)
            optimizer.step();steps+=1
            if steps%50==0 or i==batches or (trial and steps==16):
                rec=dict(epoch=epoch+1,step=steps,ce=ce.item(),kl=kl.item(),loss=loss.item(),lr=lr);history.append(rec);print(rec,flush=True)
            del logits,t,loss,ce,kl,x,y
            if trial and steps>=16:break
        name='chat_repair_trial' if trial else f'chat_repair_e{epoch+1}';path=e.P/f'out/{name}_768.pth'
        assert not path.exists(),'Retain all earlier checkpoints'
        state={k:p.detach().cpu().half().clone() for k,p in student.state_dict().items()};assert all(torch.isfinite(p).all() for p in state.values())
        torch.save(state,path);assert all(p.grad is None for p in teacher.parameters())
        e.save(f'train_{name}.json',dict(name=name,base=e.BASE,base_sha256=plan['baseline_sha256'],full_parameter_training=True,
            trainable_parameters=sum(p.numel() for p in params),steps=steps,epoch=epoch+1,trial=trial,seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,settings=s,history=history,all_weights_finite=True,kl_anchor_frozen=True,
            weight_sha256=e.sha(path),train_sha256=plan['train_sha256'],source_sha256=e.sha(__file__)))
    print('Full-parameter training completed',time.perf_counter()-started,'seconds',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['tokens','trial','train']);a=p.parse_args()
    tokenize() if a.stage=='tokens' else run(a.stage=='trial')
