"""Independent recheck of summary.json from raw answers and verdicts. Shares no code with analyse_strong_teacher.py."""
import hashlib
import json
from pathlib import Path
import random
import statistics as st

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'my_eval/results/strong_teacher_20261027'
read = lambda n: json.loads((OUT / n).read_text(encoding='utf-8'))
CONDS = ['no_tools', 'six_original', 'six_reworded']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def key(task, row):
    return hashlib.sha256(json.dumps([task['messages'], row['final'], row['eligible'], row.get('failure')], ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def main():
    lock, summary, selected = read('final_lock.json'), read('summary.json'), read('selection.json')['selected']
    for name, h in lock['weights'].items():
        assert sha(ROOT / f'out/{name}_768.pth') == h, name
    print('locked weights unchanged:', len(lock['weights']))
    tasks, reviews = read('final_tasks.json'), read('content_reviews.json')
    base, weak, strong = 'chat_repair_id_e3', selected['weak'], selected['strong']
    per = {}
    for name in [base, weak, strong]:
        raw = read(f'finalchat_{name}.json')
        assert len(raw) == 288
        per[name] = {t['id']: [reviews[key(t, raw[t['id'] + '_' + c])]['score'] for c in CONDS] for t in tasks}
        flat = [s for v in per[name].values() for s in v]
        got = (st.mean(flat), 100 * sum(s == 2 for s in flat) / len(flat))
        want = summary['models'][name]
        assert abs(got[0] - want['mean_score']) < 1e-9 and abs(got[1] - want['usable_percent']) < 1e-9, name
        print(f'{name}: content {got[0]:.3f}, usable {got[1]:.2f}% of {len(flat)} answers ({sum(s == 2 for s in flat)})')
    rng = random.Random(1)
    for label, a, b in [('7B-v4', strong, base), ('1.5B-v4', weak, base), ('7B-1.5B', strong, weak)]:
        for metric, f in [('content', lambda v: st.mean(v)), ('usable', lambda v: 100 * sum(s == 2 for s in v) / 3)]:
            d = [f(per[a][t['id']]) - f(per[b][t['id']]) for t in tasks]
            se = st.stdev(d) / len(d) ** .5
            boots = sorted(st.mean(rng.choices(d, k=len(d))) for _ in range(4000))
            print(f'{label} {metric}: {st.mean(d):+.3f} +/- {se:.3f}, 95% [{boots[100]:+.3f}, {boots[3899]:+.3f}], better {sum(x > 1e-9 for x in d)}, worse {sum(x < -1e-9 for x in d)}')
    comp = summary['comparisons']['strong_minus_v4']
    d = [st.mean(per[strong][t['id']]) - st.mean(per[base][t['id']]) for t in tasks]
    assert abs(st.mean(d) - comp['score']['difference']) < 1e-9
    for domain in ['historical', 'freshmath']:
        out = {}
        for name in [base, weak, strong]:
            out[name] = sum(r['grade']['status'] == 'pass' for r in read(f'{domain}_{name}.json').values())
        print(domain, 'auto-graded passes before manual review:', out, 'recorded %:', {k: round(v, 2) for k, v in summary['retention'][domain]['means'].items()})
    print('recheck passed')

if __name__ == '__main__':
    main()
