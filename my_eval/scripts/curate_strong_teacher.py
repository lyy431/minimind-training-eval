"""Screen both teachers without rewriting labels; construct matched training arms."""
from collections import Counter
import copy
import hashlib
import json
import random
import strong_teacher_common as e

def key(task,row):
    return hashlib.sha256(json.dumps([task['messages'],row['answer'],row['complete']],ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def teacher_queue():
    items={r['id']:r for r in e.read(e.OUT/'train_items.json')};queue={};mapping={};status={}
    reviews=e.read(e.OUT/'teacher_content_reviews.json') if (e.OUT/'teacher_content_reviews.json').exists() else {}
    for arm in ['weak','strong']:
        file=e.OUT/f'teacher_{arm}.jsonl'
        if not file.exists():continue
        raw=e.rows(file);assert len({r['id'] for r in raw})==len(raw)
        for r in raw:
            t=items[r['id']];assert r['question']==t['question'];k=key(t,r);mapping[f'{arm}::{r["id"]}']=k
            if not r['structural_ok']:decision=dict(accepted=False,reason='Incomplete/empty/forbidden teacher format')
            elif 'exact_answer' in t:
                accepted=r['answer'].strip().replace('\r\n','\n')==t['exact_answer']
                decision=dict(accepted=accepted,reason='Exact complete answer' if accepted else 'Not exactly the requested output; no keyword-only acceptance')
            elif k in reviews:
                review=reviews[k];assert review['score'] in [0,1,2] and review.get('reason')
                decision=dict(accepted=review['score']==2,reason=review['reason'])
            else:
                decision=dict(accepted=None,reason='Pending content review')
                queue[k]=dict(review_id=k,question=t['question'],language=t['language'],category=t['category'],reference=t['reference_for_review_only'],answer=r['answer'])
            status[f'{arm}::{r["id"]}']=decision
    todo=list(queue.values());random.Random(202610278).shuffle(todo)
    e.save('teacher_review_map.json',mapping);e.save('teacher_review_queue.json',todo);e.save('teacher_screening.json',status)
    e.save('teacher_content_reviews.json',reviews)
    print('Pending open-answer judgments:',len(todo),'screened',Counter(str(v['accepted']) for v in status.values()))

def build():
    assert not (e.OUT/'training_lock.json').exists(),'Data are frozen once built'
    teacher_queue();status=e.read(e.OUT/'teacher_screening.json')
    raw={a:{r['id']:r for r in e.rows(e.OUT/f'teacher_{a}.jsonl')} for a in ['weak','strong']}
    items={r['id']:r for r in e.read(e.OUT/'train_items.json')}
    assert all(set(raw[a])==set(items) for a in raw),'Both teachers must finish all1000 requests'
    assert all(s['accepted'] is not None for s in status.values()),'Complete all open-answer reviews first'
    common=sorted(k for k in items if all(status[f'{a}::{k}']['accepted'] for a in raw))
    counts=Counter(items[k]['category'] for k in common)
    assert len(common)>=400 and counts['writing']>=80 and counts['explanation']>=40,('Insufficient matched data',counts)
    e.configure_local();import tool_choice_round2 as old
    cells=[(count,variant) for count in range(1,7) for variant in range(6)]
    cells=(cells*((len(common)+35)//36))[:len(common)];random.Random(202610279).shuffle(cells)
    blocked={e.norm(r['question']) for split in ['dev','final'] for r in e.read(e.OUT/f'{split}_tasks.json')}
    retention=[]
    for name in ['retention_math640.json','retention_kl600.json','retention_identity240.json']:retention+=e.read(e.OUT/name)
    assert all(e.norm(m['content']) not in blocked for r in retention for m in r['messages'] if m['role']=='user')
    identities=[r for r in retention if r['source_kind']=='identity'];assert len(identities)==240
    train_hashes={};token_counts={}
    for arm in raw:
        data=[]
        for i,k in enumerate(common):
            t=items[k];answer=raw[arm][k]['answer'];count,variant=cells[i]
            for tools in [None,old.menu(count,variant,850000+i)]:
                messages=([dict(role='system',content=e.SYSTEM)] if tools is None else [])+copy.deepcopy(t['messages'])+[dict(role='assistant',content=answer)]
                data.append(dict(messages=messages,tools=tools,kind='supervised',source_kind=t['category'],case_id=k))
        data+=copy.deepcopy(retention);random.Random(202610280).shuffle(data)
        path=e.OUT/f'train_{arm}.jsonl';assert not path.exists()
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8',newline='\n')
        train_hashes[arm]=e.sha(path);token_counts[arm]=sum(raw[arm][k]['generated_tokens'] for k in common)
    audit=dict(common_cases=len(common),common_by_category=dict(counts),common_ids=common,rows_per_arm=2*len(common)+len(retention),
        teacher_pass_counts={a:sum(status[f'{a}::{k}']['accepted'] for k in items) for a in raw},
        teacher_tokens_in_common_cases=token_counts,same_prompts_same_updates=True,equal_answer_tokens=False,
        tool_cells={str(k):v for k,v in Counter(cells).items()},no_rewritten_teacher_answers=True,heldout_exact_overlap=0)
    e.save('matched_data_audit.json',audit,immutable=True)
    e.save('training_lock.json',dict(base=e.BASE,base_sha256=e.BASE_SHA,train_sha256=train_hashes,
        files_sha256={n:e.sha(e.OUT/n) for n in ['plan.json','teacher_weak.jsonl','teacher_strong.jsonl','teacher_content_reviews.json','teacher_screening.json','matched_data_audit.json']},
        source_sha256={n:e.sha(e.P/'my_eval/scripts'/n) for n in ['strong_teacher_common.py','train_strong_teacher.py','curate_strong_teacher.py']}),immutable=True)
    print(audit)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['queue','build']);a=p.parse_args()
    teacher_queue() if a.stage=='queue' else build()
