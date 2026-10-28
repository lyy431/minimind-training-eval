"""Independent stdlib decision and question-cluster statistics recheck for round2."""
from collections import defaultdict
import json
from pathlib import Path
from validate_tool_choice_v2 import args_ok,pair
P=Path(__file__).resolve().parents[2];OUT=P/'my_eval/results/tool_choice_round2_20261026'
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def main():
    s=read('summary.json');tasks=read('test_tasks.json');vectors={};checked=0
    for name in [s['base'],s['candidate']]:
        raw=read(f'final_{name}.json');assert len(raw)==864
        groups=defaultdict(list)
        for t in tasks:
            text=raw[t['task_id']]['final'];route,args=args_ok(text,t)
            assert route==raw[t['task_id']]['route_ok'] and args==raw[t['task_id']]['arguments_ok']
            groups[t['id']].append((t['kind'],args if t['kind']=='math' else route));checked+=1
        assert len(groups)==72 and all(len(v)==12 for v in groups.values())
        vectors[name]={}
        for kind,metric,n in [('math','math_arguments',48),('no_tool','no_tool',24)]:
            vals=[100*sum(x[1] for x in groups[q])/12 for q in sorted(groups) if groups[q][0][0]==kind]
            assert len(vals)==n and abs(sum(vals)/n-s['models'][name][metric])<1e-9
            vectors[name][metric]=vals
    for metric in ['math_arguments','no_tool']:
        d,se=pair(vectors[s['candidate']][metric],vectors[s['base']][metric]);p=s['pairs'][metric]
        assert abs(d-p['difference'])<1e-9 and abs(se-p['se'])<1e-9
    result=dict(passed=True,raw_reply_decisions_checked=checked,math_clusters=48,no_tool_clusters=24,conditions_per_cluster=12,
        independent_means_and_standard_errors_match=True)
    (OUT/'validation_independent.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
