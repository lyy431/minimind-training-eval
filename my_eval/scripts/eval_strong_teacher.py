"""Reuse the frozen v4 evaluator, changing only experiment directory and task sets."""
import argparse
import copy
import strong_teacher_common as e
e.configure_local()
import torch
import tool_choice_round2 as old
import eval_chat_repair as legacy
e.old=old;e.copy=copy
legacy.e=e

def verify_final(name):
    lock=e.read(e.OUT/'final_lock.json');assert name in lock['weights']
    for n,h in lock['weights'].items():assert e.sha(e.P/f'out/{n}_768.pth')==h,n
    for n,h in lock['files_sha256'].items():assert e.sha(e.P/n)==h,n

def main(a):
    if a.stage=='final':verify_final(a.model)
    else:assert not (e.OUT/'final_lock.json').exists(),'Development is closed after formal freeze'
    model=e.load(a.model);tok=e.tokenizer();torch.cuda.reset_peak_memory_stats()
    legacy.generate_rows('devchat' if a.stage=='dev' else 'finalchat',a.model,model,tok,legacy.chat_tasks(a.stage))
    if a.stage=='dev':
        legacy.generate_rows('devmath',a.model,model,tok,legacy.math_tasks('devmath'))
        legacy.nll('loss_dev',a.model,model,tok)
        # These are already known v4 identity development questions, not new tests.
        tasks=[t for t in e.read(e.P/'my_eval/results/chat_repair_20261027/dev_tasks.json') if t['category']=='identity']
        for t in tasks:t.update(task_id=t['id'],kind='chat',tools=None,input_messages=[dict(role='system',content=e.SYSTEM)]+t['messages'])
        e.save('identity_dev_tasks.json',tasks,immutable=True)
        legacy.generate_rows('devidentity',a.model,model,tok,tasks)
    else:
        for domain in ['historical','freshmath']:legacy.generate_rows(domain,a.model,model,tok,legacy.math_tasks(domain))
        legacy.nll('loss_final',a.model,model,tok)
        tasks=legacy.identity_tasks();e.save('identity_tasks.json',tasks,immutable=True)
        legacy.generate_rows('identity',a.model,model,tok,tasks)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['dev','final']);p.add_argument('--model',default=e.BASE);main(p.parse_args())
