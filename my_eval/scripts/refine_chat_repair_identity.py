"""Identity retention on exact repaired-e2 base; never merge into a different base."""
import argparse
import copy
import math
import random
import time
import chat_repair_common as e
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from model.model_lora import apply_lora
from train_chat_repair import collate

BASE='chat_repair_e2'

def prepare():
    assert not (e.OUT/'identity_refinement_train.jsonl').exists()
    rows=e.rows(e.OUT/'train.jsonl');rng=random.Random(202610277);ordinary=[]
    for category in ['general','knowledge','writing','retrieval']:
        candidates=[r for r in rows if r['kind']=='supervised' and r['source_kind']==category];rng.shuffle(candidates);ordinary+=candidates[:80]
    identities=[]
    for c in e.read(e.OUT/'fixed_supervision.json'):
        if c['source_kind']!='identity':continue
        english=c['question'][0].isascii()
        answer=('I am MiniMind, customized by Yuyang LU based on jingyaogong\'s original MiniMind project.' if english else '我是MiniMind，由Yuyang LU基于jingyaogong的原始MiniMind项目进行本地定制。')
        for tools in [None,e.old.menu(6,0,730000+len(identities)),e.old.menu(6,rng.randint(1,5),731000+len(identities))]:
            messages=([dict(role='system',content=e.SYSTEM)] if not tools else [])+[dict(role='user',content=c['question']),dict(role='assistant',content=answer)]
            identities.append(dict(messages=messages,tools=tools,kind='supervised',source_kind='identity'))
    mathrows=e.read(e.OUT/'math_supervision640.json');rng.shuffle(mathrows)
    result=identities+ordinary+mathrows[:320]+e.read(e.OUT/'retention600.json');rng.shuffle(result)
    (e.OUT/'identity_refinement_train.jsonl').write_text(''.join(e.json.dumps(r,ensure_ascii=False)+'\n' for r in result),encoding='utf-8',newline='\n')
    tok=e.tok();bos=tok('<|im_start|>assistant\n',add_special_tokens=False).input_ids;eos=tok('<|im_end|>\n',add_special_tokens=False).input_ids;data=[]
    held={e.old.norm(t['question']) for t in e.read(e.OUT/'dev_tasks.json')+e.read(e.OUT/'final_tasks.json')}
    for row in result:
        assert e.old.norm('\n'.join(m['content'] for m in row['messages'] if m['role']=='user')) not in held
        messages=copy.deepcopy(row['messages'])
        for m in messages:
            if isinstance(m.get('tool_calls'),str) and m['tool_calls']:m['tool_calls']=e.json.loads(m['tool_calls'])
        text=tok.apply_chat_template(messages,tools=row['tools'],tokenize=False,add_generation_prompt=False);ids=tok(text,add_special_tokens=False).input_ids
        assert len(ids)<=1536;labels=[-100]*len(ids);j=0;spans=0
        while j<len(ids):
            if ids[j:j+len(bos)]==bos:
                a=j+len(bos);b=a
                while b<len(ids) and ids[b:b+len(eos)]!=eos:b+=1
                assert b<len(ids);labels[a:b+len(eos)]=ids[a:b+len(eos)];j=b+len(eos);spans+=1
            else:j+=1
        assert spans==sum(m['role']=='assistant' for m in messages)
        data.append(dict(ids=ids,labels=labels,retain=row['kind']=='retain',identity=row['source_kind']=='identity'))
    torch.save(data,e.OUT/'identity_refinement_tokenized.pt')
    e.save('identity_refinement_plan.json',dict(base=BASE,base_sha256=e.sha(e.P/f'out/{BASE}_768.pth'),
        epochs=3,lr=.0002,kl=4.,rank=16,batch=4,seed=42,identity_row_weight=3,
        counts=dict(identity_ce=len(identities),ordinary_ce=len(ordinary),math_ce=320,kl=600),rows=len(data),max_tokens=max(len(r['ids']) for r in data),
        identity_target='Canonical complete name+customizer+upstream relation in both languages, same for all identity variants',
        train_sha256=e.sha(e.OUT/'identity_refinement_train.jsonl'),no_truncation=True,known_chat_heldout_question_overlap=0))
    print(e.read(e.OUT/'identity_refinement_plan.json'),flush=True)

class Tokens(Dataset):
    def __init__(self):self.data=torch.load(e.OUT/'identity_refinement_tokenized.pt',weights_only=True)
    def __len__(self):return len(self.data)
    def __getitem__(self,i):return self.data[i]

def batch(rows):
    n=max(len(r['ids']) for r in rows);x=torch.zeros(len(rows),n,dtype=torch.long);y=torch.full_like(x,-100)
    for i,r in enumerate(rows):x[i,:len(r['ids'])]=torch.tensor(r['ids']);y[i,:len(r['labels'])]=torch.tensor(r['labels'])
    return x,y,torch.tensor([r['retain'] for r in rows]),torch.tensor([r['identity'] for r in rows])

def run(trial=False):
    assert not (e.OUT/'final_lock.json').exists();s=e.read(e.OUT/'identity_refinement_plan.json')
    assert e.sha(e.P/f'out/{BASE}_768.pth')==s['base_sha256'];assert e.sha(e.OUT/'identity_refinement_train.jsonl')==s['train_sha256']
    torch.manual_seed(42);student=e.load(BASE,fp32=True);teacher=e.load(BASE,fp32=True).requires_grad_(False)
    apply_lora(student,rank=16);params=[]
    for name,p in student.named_parameters():
        p.requires_grad_('.lora.' in name)
        if p.requires_grad:params.append(p)
    assert sum(p.numel() for p in params)==393216
    data=Tokens();batches=math.ceil(len(data)/4);optim=torch.optim.AdamW(params,lr=s['lr']);steps=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1 if trial else 3):
        loader=DataLoader(data,batch_size=4,shuffle=True,num_workers=2,pin_memory=True,collate_fn=batch,generator=torch.Generator().manual_seed(42+epoch));student.train()
        for i,(x,y,retain,identity) in enumerate(loader,1):
            x,y,retain,identity=x.cuda(),y.cuda(),retain.cuda(),identity.cuda();mask=y[:,1:]!=-100;labels=y[:,1:];optim.zero_grad(set_to_none=True)
            lr=s['lr']*(.1+.9*.5*(1+math.cos(math.pi*(epoch*batches+i)/(3*batches))))
            for g in optim.param_groups:g['lr']=lr
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):t=teacher(x[retain],use_cache=False).logits[:,:-1] if retain.any() else None
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=student(x,use_cache=False).logits[:,:-1]
            ce=logits[0,0,0].float()*0;kl=ce
            if (~retain).any():
                lp=logits[~retain].float();target=labels[~retain];valid=mask[~retain]
                losses=F.cross_entropy(lp.reshape(-1,lp.shape[-1]),target.reshape(-1),ignore_index=-100,reduction='none').reshape_as(target)
                ce=(((losses*valid).sum(-1)/valid.sum(-1).clamp_min(1))*(1+2*identity[~retain])).mean()
            if t is not None:
                valid=mask[retain];weights=valid*(1+2*(valid.cumsum(-1)<=32));slog=F.log_softmax(logits[retain][valid].float(),-1);tp=F.softmax(t[valid].float(),-1)
                losses=F.kl_div(slog,tp,reduction='none').sum(-1);which=torch.arange(valid.shape[0],device='cuda')[:,None].expand_as(valid)[valid]
                sums=torch.zeros(valid.shape[0],device='cuda').scatter_add(0,which,losses*weights[valid]);kl=(sums/weights.sum(-1).clamp_min(1)).mean()
            loss=ce+4*kl;assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(grad);optim.step();steps+=1
            if steps%50==0 or i==batches or (trial and steps==16):print(dict(epoch=epoch+1,step=steps,ce=ce.item(),kl=kl.item(),lr=lr),flush=True)
            del x,y,logits,t,loss,ce,kl
            if trial and steps>=16:break
        name='chat_repair_id_trial' if trial else f'chat_repair_id_e{epoch+1}';path=e.P/f'out/{name}_768.pth';assert not path.exists()
        state=student.state_dict();adapter={k:p.detach().cpu().float().clone() for k,p in state.items() if '.lora.' in k}
        merged={k:p.detach().cpu().float().clone() for k,p in state.items() if '.lora.' not in k}
        for k in merged:
            prefix=k[:-7] if k.endswith('.weight') else ''
            if prefix+'.lora.A.weight' in adapter:merged[k]+=adapter[prefix+'.lora.B.weight']@adapter[prefix+'.lora.A.weight']
        merged={k:p.half() for k,p in merged.items()};assert all(torch.isfinite(p).all() for p in merged.values())
        assert all(p.grad is None for p in teacher.parameters());assert all(p.grad is None for k,p in student.named_parameters() if '.lora.' not in k)
        torch.save(merged,path);torch.save(adapter,e.P/f'out/{name}_lora_768.pth')
        e.save(f'train_{name}.json',dict(name=name,base=BASE,base_sha256=s['base_sha256'],trial=trial,epoch=epoch+1,steps=steps,
            seconds=time.perf_counter()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            exact_base_only=True,base_and_kl_anchor_frozen=True,trainable_parameters=393216,settings=s,all_weights_finite=True,
            weight_sha256=e.sha(path),source_sha256=e.sha(__file__)))
    print('Identity refinement completed',time.perf_counter()-started,'seconds',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','trial','train']);a=p.parse_args();prepare() if a.stage=='prepare' else run(a.stage=='trial')
