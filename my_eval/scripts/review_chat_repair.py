"""Content review hides arm names; reuse only exact earlier question+answer decisions."""
import argparse
import hashlib
import json
import random
import chat_repair_common as e
from review_tool_choice_round2 import key as mathkey

def key(task,row):
    return hashlib.sha256(json.dumps([task['messages'],row['final'],row['eligible'],row.get('failure')],ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:16]

def queue(split):
    tasks={t['id']:t for t in e.read(e.OUT/('dev_tasks.json' if split=='dev' else 'final_tasks.json'))}
    ids={t['id']:t for t in e.read(e.OUT/'identity_tasks.json')}
    oldreviews=e.read(e.diagnostic.OUT/'content_reviews.json')
    reviews=e.read(e.OUT/'content_reviews.json') if (e.OUT/'content_reviews.json').exists() else {}
    unique={};mapping={}
    files=sorted(e.OUT.glob(('devchat_' if split=='dev' else 'finalchat_')+'*.json'))
    if split=='final':files+=sorted(e.OUT.glob('identity_*.json'))
    for f in files:
        if f.stem.endswith('_runtime') or f.name=='identity_tasks.json':continue
        lookup=ids if f.name.startswith('identity_') else tasks
        for name,row in e.read(f).items():
            task=lookup[row.get('id') or row.get('task_id')];k=key(task,row);mapping[f.name+'::'+name]=k
            if k not in reviews and k in oldreviews:reviews[k]=dict(oldreviews[k],source='exact prior context/answer/eligibility decision')
            if k not in reviews:unique[k]=dict(review_id=k,question=task['question'],context=task['messages'][:-1],reference=task['reference_for_review_only'],answer=row['final'],eligible=row['eligible'],failure=row.get('failure'))
    values=list(unique.values());random.Random(202610274).shuffle(values)
    e.save('content_reviews.json',reviews);e.save(split+'_review_map.json',mapping);e.save(split+'_review_queue.json',values)
    current=e.read(e.OUT/'math_reviews.json') if (e.OUT/'math_reviews.json').exists() else {};prior={}
    for folder in ['tool_choice_round2_20261026','tool_choice_20261026','best_of_n_20261026']:
        prior.update(e.read(e.P/'my_eval/results'/folder/'math_reviews.json'))
    pending={}
    for prefix in (['devmath_'] if split=='dev' else ['historical_','freshmath_']):
        for f in e.OUT.glob(prefix+'*.json'):
            if f.stem.endswith('_runtime'):continue
            for row in e.read(f).values():
                if row['grade']['status']!='review':continue
                k=mathkey(row['question'],row['final'])
                if k not in current and k in prior:current[k]=dict(prior[k],source='exact earlier question and answer')
                if k not in current:pending[k]=dict(sha256=k,question=row['question'],answer=row['final'],reason=row['grade']['reason'])
    e.save('math_reviews.json',current);e.save(split+'_math_review_queue.json',list(pending.values()))
    print('Pending content',len(values),'math',len(pending),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['dev','final']);a=p.parse_args();queue(a.split)
