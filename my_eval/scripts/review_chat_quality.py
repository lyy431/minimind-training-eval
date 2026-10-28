"""Deduplicate exact answers and hide condition/model labels during content review."""
import argparse
import hashlib
import json
import random
import chat_quality_common as e

def key(task,row):
    return hashlib.sha256(json.dumps([task['messages'],row['final'],row['eligible'],row.get('failure')],ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:16]

def queue(split):
    tasks={t['id']:t for t in e.read(e.OUT/('dev_tasks.json' if split=='dev' else 'final_tasks.json'))}
    reviews=e.read(e.OUT/'content_reviews.json') if (e.OUT/'content_reviews.json').exists() else {}
    unique={};mapping={}
    for file in sorted(e.OUT.glob(split+'_MiniMind*.json')):
        if file.stem.endswith('_runtime'):continue
        for item,row in e.read(file).items():
            task=tasks[row['task_id']];k=key(task,row);mapping[file.name+'::'+item]=k
            if k in reviews:continue
            unique[k]=dict(review_id=k,question=task['question'],context=task['messages'][:-1],reference=task['reference_for_review_only'],
                answer=row['final'],eligible=row['eligible'],failure=row.get('failure'))
    values=list(unique.values());random.Random(202610271).shuffle(values)
    e.save(split+'_review_map.json',mapping);e.save(split+'_review_queue.json',values)
    print('Unreviewed unique answers',len(values),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['dev','final']);a=p.parse_args();queue(a.split)
