"""Prepare content-only review queues without showing reward scores or choices."""
from pathlib import Path
import hashlib
import json
import random
import sys
P=Path(__file__).resolve().parents[2]
OUT=P/'my_eval/results/best_of_n_20261026'
sys.path.insert(0,str(P/'my_eval/scripts'))
from strict_math_rescore import grade

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
def key(q,a):return hashlib.sha256((q+'\n'+a).encode()).hexdigest()

def prepare():
    generated=read(OUT/'generated.json')
    tasks=read(OUT/'tasks.json')
    old={r['sha256']:r for r in read(P/'my_eval/results/strict_math_20261026/manual_reviews.json')['rows']}
    current=read(OUT/'math_reviews.json') if (OUT/'math_reviews.json').exists() else {}
    automatic={};pending={};chat={}
    for task in tasks:
        pool=generated.get(task['id'],[])
        if len(pool)!=4:continue
        if task['kind']=='math':
            automatic[task['id']]=[]
            for c in pool:
                h=key(task['question'],c['final'])
                g=grade(c['final'],task['question'],task['gt'],unfinished=not c['eligible'])
                row=dict(sha256=h,question=task['question'],final=c['final'],gt=task['gt'],**g)
                automatic[task['id']].append(row)
                if g['status']=='review':
                    if h not in current and h in old:
                        current[h]=dict(decision=old[h]['decision'],note=old[h]['note'],source='previous exact-text adjudication')
                    if h not in current:pending[h]=row
        else:
            for c in pool:
                h=key(json.dumps(task['messages'],ensure_ascii=False,sort_keys=True),c['final'])
                chat[h]=dict(sha256=h,question=task['question'],context=task['messages'],final=c['final'])
    save('math_automatic.json',automatic);save('math_review_queue.json',list(pending.values()));save('math_reviews.json',current)
    chat_rows=list(chat.values());random.Random(20261026).shuffle(chat_rows)
    save('chat_content_queue.json',chat_rows)
    print(f'Complete math questions {len(automatic)}; pending unique math {len(pending)}; chat texts {len(chat)}')

if __name__=='__main__':prepare()
