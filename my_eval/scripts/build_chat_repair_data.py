"""Assemble audited, short supervision; do not truncate incomplete teacher answers."""
import copy
import random
import collections
import chat_repair_common as e

def main():
    assert not (e.OUT/'train.jsonl').exists(),'Training data are frozen once assembled'
    req=e.read(e.OUT/'teacher_requests.json');raw=e.read(e.OUT/'teacher_answers.json');assert len(raw)==len(req)
    curation=e.read(e.OUT/'teacher_curation.json');labels=[]
    for r in req:
        if not raw[r['id']]['accepted'] or r['id'] in curation['excluded']:continue
        label=copy.deepcopy(raw[r['id']])
        if r['id'] in curation['overrides']:label['answer']=curation['overrides'][r['id']]['answer']
        labels.append(label)
    cases=e.read(e.OUT/'fixed_supervision.json')+labels
    blocked={e.old.norm(t['question']) for t in e.read(e.OUT/'dev_tasks.json')+e.read(e.OUT/'final_tasks.json')}
    for f in [e.P/'dataset/dpo_eval300.jsonl',e.P/'my_eval/results/identity_repair_20261024/dev_language100.jsonl']:
        blocked.update(e.old.norm(e.old.question(r)) for r in e.rows(f))
    rng=random.Random(202610273);rows=[]
    for i,c in enumerate(cases):
        assert e.old.norm(c['question']) not in blocked
        for tools in [None,e.old.menu(1+i%6,i%6,720000+i)]:
            messages=([dict(role='system',content=e.SYSTEM)] if not tools else [])+[dict(role='user',content=c['question']),dict(role='assistant',content=c['answer'])]
            rows.append(dict(messages=messages,tools=tools,kind='supervised',source_kind=c['source_kind'],case_id=f'ordinary_{i}'))
    retained=[];excluded=[]
    for row in e.read(e.OUT/'math_supervision640.json')+e.read(e.OUT/'retention600.json'):
        q='\n'.join(m['content'] for m in row['messages'] if m['role']=='user')
        if e.old.norm(q) in blocked:excluded.append(q);continue
        rows.append(row)
    rng.shuffle(rows);text=''.join(e.json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
    (e.OUT/'train.jsonl').write_text(text,encoding='utf-8',newline='\n')
    counts=collections.Counter((r['kind'],r['source_kind']) for r in rows)
    e.save('data_audit.json',dict(rows=len(rows),ordinary_cases=len(cases),teacher_accepted=len(labels),teacher_format_accepted=sum(r['accepted'] for r in raw.values()),
        teacher_content_scope_excluded=len(curation['excluded']),teacher_answers_corrected=len(curation['overrides']),
        teacher_incomplete=sum(not r['complete'] for r in raw.values()),teacher_other_rejected=sum(r['complete'] and not r['accepted'] for r in raw.values()),
        excluded_known_heldout_questions=excluded,counts={str(k):v for k,v in counts.items()},normalized_exact_overlap_with_heldout=0,
        train_sha256=e.sha(e.OUT/'train.jsonl'),accepted_teacher_answers_only=True))
    plan=e.read(e.OUT/'plan.json');plan['train_sha256']=e.sha(e.OUT/'train.jsonl');plan['actual_rows']=len(rows)
    plan['preparation_complete_before_training']=True;e.save('plan.json',plan)
    print(e.read(e.OUT/'data_audit.json'),flush=True)

if __name__=='__main__':main()

