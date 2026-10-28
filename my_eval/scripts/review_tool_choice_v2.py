"""Offline strict grading review queues; raw generations and frozen rules are preserved."""
import hashlib
import tool_choice_v2 as e
def key(q,a):return hashlib.sha256((q+'\n'+a).encode()).hexdigest()

def main():
    lookup={}
    for path in [e.P/'my_eval/results/strict_math_20261026/manual_reviews.json',
                 e.P/'my_eval/results/best_of_n_20261026/math_reviews.json']:
        data=e.read(path)
        for h,r in ([(r['sha256'],r) for r in data['rows']] if 'rows' in data else data.items()):lookup[h]=r
    current=e.read(e.OUT/'math_reviews.json') if (e.OUT/'math_reviews.json').exists() else {}
    pending={};count=0
    for path in sorted(e.OUT.glob('*.json')):
        if not path.name.startswith(('final_bb_','final_tool_choice_','guard_bb_','guard_tool_choice_')) or path.name.endswith(('_routes.json','_runtime.json')):continue
        for r in e.read(path).values():
            if r.get('grade',{} ) is None or r.get('grade',{}).get('status')!='review':continue
            count+=1;h=key(r['question'],r['final'])
            if h not in current and h in lookup:
                current[h]=dict(decision=lookup[h]['decision'],note=lookup[h]['note'],source='earlier same question and exact-text decision')
            if h not in current:pending[h]=dict(sha256=h,question=r['question'],final=r['final'],reason=r['grade']['reason'])
    e.save('math_reviews.json',current);e.save('math_review_queue.json',list(pending.values()))
    print('Uncertain runs:',count,'pending unique:',len(pending),'recorded:',len(current),flush=True)

if __name__=='__main__':main()
