"""Reproducible correction: decorrelate arithmetic count and available-tool count."""
import random
import tool_choice_round2 as e

def main():
    assert not (e.OUT/'final_lock.json').exists()
    data=e.rows(e.OUT/'train.jsonl');rng=random.Random(202610268)
    counts={k:[n for n in range(1,7) for _ in range(80)] for k in [1,2,3]}
    for ns in counts.values():rng.shuffle(ns)
    for i,row in enumerate(data):
        if row['kind']=='retain':continue
        if row['source_kind']=='math':n=counts[len(e.expressions(row['messages'][0]['content']))].pop()
        else:n=rng.randint(1,6)
        row['tools']=e.menu(n,rng.randrange(4),960000+i)
    assert all(not ns for ns in counts.values())
    text=''.join(e.json.dumps(r,ensure_ascii=False)+'\n' for r in data)
    (e.OUT/'train.jsonl').write_text(text,encoding='utf-8')
    plan=e.read(e.OUT/'plan.json');plan['data_sha256']['train.jsonl']=e.sha(e.OUT/'train.jsonl')
    plan['development_correction']='Before any development/final inference, unused first round2 training retained. Corrected independent shuffled/balanced tool counts across each1/2/3-expression group. Dev/final tasks unchanged.'
    e.save('plan.json',plan)
    cross=e.Counter((len(e.expressions(r['messages'][0]['content'])),len(r['tools'])) for r in data if r['kind']=='supervised' and r['source_kind']=='math')
    assert len(cross)==18 and set(cross.values())=={80}
    e.save('count_balance_audit.json',dict(passed=True,cells={str(k):v for k,v in cross.items()},train_sha256=e.sha(e.OUT/'train.jsonl')))
    print('18 count/complexity cells balanced at80; no test inputs changed.',flush=True)

if __name__=='__main__':main()
