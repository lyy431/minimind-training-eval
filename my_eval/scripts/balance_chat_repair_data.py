"""Balance tool count independently of description before the first training update."""
import collections
import random
import shutil
import chat_repair_common as e

def main():
    assert not list(e.OUT.glob('train_chat_repair*.json')),'Balance before any parameter update'
    assert not (e.OUT/'menu_balance_audit.json').exists()
    backup=e.OUT/'development_prebalance';backup.mkdir()
    for name in ['train.jsonl','data_audit.json','plan.json']:shutil.copy2(e.OUT/name,backup/name)
    rows=e.rows(e.OUT/'train.jsonl');toolrows=[r for r in rows if r.get('case_id') and r['tools']]
    rng=random.Random(202610276);cells=[(n,v) for n in range(1,7) for v in range(6)]
    assignments=cells*(len(toolrows)//36)+rng.sample(cells,len(toolrows)%36);rng.shuffle(assignments)
    counts=collections.Counter()
    for i,(row,(n,v)) in enumerate(zip(toolrows,assignments)):
        row['tools']=e.old.menu(n,v,725000+i);counts[n,v]+=1
    assert len(counts)==36 and max(counts.values())-min(counts.values())<=1
    (e.OUT/'train.jsonl').write_text(''.join(e.json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8',newline='\n')
    h=e.sha(e.OUT/'train.jsonl');audit=e.read(e.OUT/'data_audit.json');audit['train_sha256']=h;e.save('data_audit.json',audit)
    plan=e.read(e.OUT/'plan.json');plan['train_sha256']=h
    plan['pretraining_correction']='Before any parameter updates, ordinary menus made jointly balanced over1-6 tools x6 descriptions (21/22 rows per cell), shuffled independently of source categories. Prebalance data preserved. Tokenization also normalizes legacy JSON-string tool_calls.'
    e.save('plan.json',plan);e.save('menu_balance_audit.json',dict(passed=True,cells={str(k):v for k,v in counts.items()},tool_rows=len(toolrows),train_sha256=h,corrected_before_training=True))
    print('Balanced36 independent tool-count/description cells before training.',flush=True)

if __name__=='__main__':main()
