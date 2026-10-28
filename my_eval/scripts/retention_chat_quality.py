"""Math retention with exact frozen-v3 baselines, plus full-content identity review."""
import copy
import time
import chat_quality_common as e
import torch
from eval_chat_quality import verify,generate
from chat_quality_support import ordinary_chat
from demo_tools import execute_tool
from strict_math_rescore import grade

def main():
    verify();policy=e.read(e.OUT/'selection.json')['selected'];old=e.old.OUT
    basefiles={'historical':old/'guard_tool_choice2b_e3.json','fresh_math':old/'fullmath_tool_choice2b_e3.json'}
    reuse={}
    for domain,source in basefiles.items():
        e.save(domain+'_baseline.json',e.read(source));reuse[domain]=dict(source=str(source.relative_to(e.P)),sha256=e.sha(source),
            frozen_core=e.sha(e.P/'scripts/best_of_n.py'),model_export_parameters_exact=e.read(old/'demo_export.json')['all_parameter_tensors_exact'])
    e.save('baseline_reuse.json',reuse)
    model,tok=e.load();backend=e.read(e.OUT/'selection.json')['chat_backend']
    chat_model,chat_tok=(model,tok) if backend==e.MODEL else e.load(backend)
    started=time.perf_counter()
    for domain in basefiles:
        tasks=e.read(old/'guardrail_tasks.json') if domain=='historical' else [t for t in e.read(old/'test_tasks.json') if t['kind']=='math' and t['tool_count']==6 and t['variant']==6]
        path=e.OUT/f'{domain}_candidate.json';done=e.read(path) if path.exists() else {}
        for i,t in enumerate(tasks):
            k=t.get('task_id',t['id'])
            if k in done:continue
            c=ordinary_chat(model,tok,[dict(role='user',content=t['question'])],tools=t['tools'],execute_tool=execute_tool,
                seed=t['seed'],max_new_tokens=384,system=e.SYSTEMS[policy],chat_model=chat_model,chat_tokenizer=chat_tok)
            done[k]=dict(task_id=k,question=t['question'],**c,grade=grade(c['final'],t['question'],t['gt'],unfinished=not c['eligible']))
            e.save(path.name,done)
            if (i+1)%20==0 or i==len(tasks)-1:print(domain,i+1,'/',len(tasks),f'{time.perf_counter()-started:.1f}s',flush=True)
    qs=['你是谁？','Who created you?','你的作者叫什么名字？','是哪个人把你做出来的？','介绍一下你的开发者','Who is your developer?','你和 Jingyao Gong 是什么关系？']
    tasks=[dict(id=f'identity_{i}_{j}',question=q,messages=[dict(role='user',content=q)],
        reference_for_review_only=('原作Jingyao Gong/jingyaogong，本地定制者Yuyang LU，助手MiniMind' if i==6 else '准确介绍MiniMind或其本地定制者Yuyang LU，按提问回答；不张冠李戴'),
        seed=550000+i*10+j) for i,q in enumerate(qs) for j in [0,1]]
    e.save('identity_tasks.json',tasks)
    for arm in ['baseline',policy]:
        path=e.OUT/f'identity_{e.MODEL}_{arm}.json';done=e.read(path) if path.exists() else {}
        for t in tasks:
            if t['id'] not in done:
                done[t['id']]=dict(task_id=t['id'],**generate(model,tok,t,'no_tools',arm,chat_model,chat_tok));e.save(path.name,done)
        print('identity',arm,len(done),flush=True)
    e.save('retention_runtime.json',dict(seconds=time.perf_counter()-started))

if __name__=='__main__':main()
