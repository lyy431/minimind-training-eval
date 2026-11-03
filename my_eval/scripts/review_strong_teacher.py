"""Review content with hidden arm labels and strict whole-answer matching."""
import argparse
import hashlib
import json
import random
import strong_teacher_common as e
e.configure_local()
from review_tool_choice_round2 import key as math_key

def content_key(task,row):
    return hashlib.sha256(json.dumps([task['messages'],row['final'],row['eligible'],row.get('failure')],ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def names(split):
    if split=='final':return list(e.read(e.OUT/'final_lock.json')['weights'])
    return [e.BASE]+[f'strong_teacher_{arm}_e{i}' for arm in ['weak','strong'] for i in [1,2]]

def queue(split):
    tasks={t['id']:t for t in e.read(e.OUT/f'{split}_tasks.json')}
    reviews=e.read(e.OUT/'content_reviews.json') if (e.OUT/'content_reviews.json').exists() else {}
    pending={};mapping={}
    domains=[('devchat' if split=='dev' else 'finalchat',tasks)]
    idfile=e.OUT/('identity_dev_tasks.json' if split=='dev' else 'identity_tasks.json')
    if idfile.exists():domains.append(('devidentity' if split=='dev' else 'identity',{t['id']:t for t in e.read(idfile)}))
    for name in names(split):
        for domain,lookup in domains:
            path=e.OUT/f'{domain}_{name}.json'
            if not path.exists():continue
            for record_id,row in e.read(path).items():
                task=lookup[row['id'] or row['task_id']];k=content_key(task,row);mapping[path.name+'::'+record_id]=k
                if k in reviews:continue
                if 'exact_answer' in task and row['eligible'] and row['final'].strip().replace('\r\n','\n')==task['exact_answer'] and not any('<tool_call' in t for t in row['turns']):
                    reviews[k]=dict(score=2,reason='Complete exact requested output; whole answer equals reference',source='strict deterministic exact match')
                else:
                    pending[k]=dict(review_id=k,question=task['question'],context=task['messages'][:-1],reference=task['reference_for_review_only'],exact_answer=task.get('exact_answer'),answer=row['final'],eligible=row['eligible'],failure=row.get('failure'))
    todo=list(pending.values());random.Random(202610281).shuffle(todo)
    e.save('content_reviews.json',reviews);e.save(f'{split}_review_map.json',mapping);e.save(f'{split}_review_queue.json',todo)
    mathreviews=e.read(e.OUT/'math_reviews.json') if (e.OUT/'math_reviews.json').exists() else {};prior={};mathpending={}
    for folder in ['tool_choice_round2_20261026','chat_repair_20261027']:prior.update(e.read(e.P/'my_eval/results'/folder/'math_reviews.json'))
    for name in names(split):
        for domain in (['devmath'] if split=='dev' else ['historical','freshmath']):
            path=e.OUT/f'{domain}_{name}.json'
            if not path.exists():continue
            for row in e.read(path).values():
                if row['grade']['status']!='review':continue
                k=math_key(row['question'],row['final'])
                if k not in mathreviews and k in prior:mathreviews[k]=dict(prior[k],source='identical earlier question and answer')
                if k not in mathreviews:mathpending[k]=dict(review_id=k,question=row['question'],answer=row['final'],reason=row['grade']['reason'])
    e.save('math_reviews.json',mathreviews);e.save(f'{split}_math_review_queue.json',list(mathpending.values()))
    print('Pending content',len(todo),'math',len(mathpending))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['dev','final']);queue(p.parse_args().split)
