"""Batched first-reply routing, with question clusters kept intact."""
import statistics as st
import time
import torch
import tool_choice_round2 as e
import eval_tool_choice_round2 as v

@torch.inference_mode()
def run(stage,name):
    if stage=='final':v.verify_final(name)
    tasks=e.read(e.OUT/('dev_tasks.json' if stage=='dev' else 'test_tasks.json'));n=4 if stage=='dev' else 12
    path=e.OUT/f'{stage}_{name}.json';done=e.read(path) if path.exists() else {}
    tok=e.load_tok();tok.padding_side='left';model=e.load_model(name);started=time.perf_counter()
    def texts(ts):return [tok.apply_chat_template([dict(role='user',content=t['question'])],tools=t['tools'],tokenize=False,add_generation_prompt=True) for t in ts]
    dev=e.read(e.OUT/'dev_tasks.json')[:4];probe=[dev[0],dev[-1]];render=texts(probe)
    ids=tok(render,padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
    batch=model.generate(input_ids=ids.input_ids,attention_mask=ids.attention_mask,max_new_tokens=24,do_sample=False)
    def trim(t):
        xs=t.tolist();return xs[:xs.index(tok.eos_token_id)+1] if tok.eos_token_id in xs else xs
    for i,text in enumerate(render):
        one=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda')
        out=model.generate(input_ids=one.input_ids,attention_mask=one.attention_mask,max_new_tokens=24,do_sample=False)
        assert trim(batch[i,ids.input_ids.shape[1]:])==trim(out[0,one.input_ids.shape[1]:])
    e.save(f'{stage}_{name}_padding.json',dict(passed=True,greedy24_equal=True))
    for start in range(0,len(tasks),n):
        group=tasks[start:start+n]
        assert len({t['id'] for t in group})==1
        if all(t['task_id'] in done for t in group):continue
        ids=tok(texts(group),padding=True,return_tensors='pt',add_special_tokens=False).to('cuda')
        torch.manual_seed((940000 if stage=='dev' else 950000)+start//n)
        output=model.generate(input_ids=ids.input_ids,attention_mask=ids.attention_mask,max_new_tokens=384,
            do_sample=True,temperature=.9,top_p=.85,top_k=50,repetition_penalty=1.,
            pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        for t,tokens in zip(group,output[:,ids.input_ids.shape[1]:]):
            reply=tok.decode(tokens,skip_special_tokens=True)
            done[t['task_id']]=dict(task_id=t['task_id'],question=t['question'],kind=t['kind'],turns=[reply],final=reply,
                **e.first_call_metrics(dict(turns=[reply]),t),name_hit='minimind' in reply.casefold() and 'yuyang' in reply.casefold())
        e.save(path.name,done)
        if (start//n+1)%12==0:print(stage,name,start//n+1,'/',len(tasks)//n,f'{time.perf_counter()-started:.1f}s',flush=True)
    ms=[r for r in done.values() if r['kind']=='math'];ns=[r for r in done.values() if r['kind']=='no_tool']
    result=dict(math_name=100*st.mean(r['route_ok'] for r in ms),math_arguments=100*st.mean(r['arguments_ok'] for r in ms),
        no_tool=100*st.mean(r['route_ok'] for r in ns))
    e.save(f'{stage}_{name}_routes.json',result);print(result,flush=True)
    e.save(f'{stage}_{name}_runtime.json',dict(seconds=time.perf_counter()-started,base_questions=len(tasks)//n,conditions=len(tasks)))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['dev','final']);p.add_argument('--model',default=e.BASE);a=p.parse_args();run(a.stage,a.model)
