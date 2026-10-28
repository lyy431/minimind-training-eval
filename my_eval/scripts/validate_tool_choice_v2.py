"""Independent stdlib recheck from raw replies: decisions, denominators, clustered means/SEs."""
import ast
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
P=Path(__file__).resolve().parents[2];OUT=P/'my_eval/results/tool_choice_20261026'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))
def expr(s):
    if not isinstance(s,str):raise ValueError()
    return ast.dump(ast.parse(s.strip(),mode='eval'),include_attributes=False)
def args_ok(text,task):
    calls=[];broken=text.count('<tool_call>')!=text.count('</tool_call>')
    spans=re.findall(r'<tool_call>(.*?)</tool_call>',text,re.S)
    if '<tool_call' in text and not spans:broken=True
    for s in spans:
        try:
            c=json.loads(s)
            if not isinstance(c,dict):broken=True
            else:calls.append(c)
        except ValueError:broken=True
    if task['kind']=='no_tool':return bool(text.strip()) and not broken and not calls and '<tool_call>' not in text,None
    correct=bool(calls) and not broken and all(c.get('name')=='calculate_math' for c in calls)
    try:complete=correct and Counter(expr(c['arguments']['expression']) for c in calls)==Counter(expr(x) for x in task['expressions'])
    except (KeyError,TypeError,ValueError,SyntaxError):complete=False
    return bool(correct),bool(complete)
def pair(x,y):
    d=[a-b for a,b in zip(x,y)];m=sum(d)/len(d)
    return m,(sum((v-m)**2 for v in d)/(len(d)-1)/len(d))**.5

def main():
    summary=read('summary.json');names=[summary['base'],summary['candidate']];tasks=read('test_tasks.json')
    checks=0;vectors={}
    for name in names:
        raw=read(f'final_{name}.json');assert len(raw)==384
        grouping=defaultdict(list)
        for t in tasks:
            r=raw[t['task_id']];first=r['turns'][0] if r['turns'] else ''
            route,parameters=args_ok(first,t)
            assert route==r['route_ok'] and parameters==r['arguments_ok']
            grouping[t['id']].append((t['kind'],parameters if t['kind']=='math' else route));checks+=1
        assert len(grouping)==48 and all(len(v)==8 for v in grouping.values())
        vectors[name]={kind:[100*sum(r[1] for r in grouping[q])/8 for q in sorted(grouping) if grouping[q][0][0]==kind] for kind in ['math','no_tool']}
        for kind,metric in [('math','math_arguments'),('no_tool','no_tool')]:
            assert len(vectors[name][kind])==24
            assert abs(sum(vectors[name][kind])/24-summary['models'][name][metric])<1e-9
    for kind,metric in [('math','math_arguments'),('no_tool','no_tool')]:
        diff,se=pair(vectors[names[1]][kind],vectors[names[0]][kind]);p=summary['pairs'][metric]
        assert abs(diff-p['difference'])<1e-9 and abs(se-p['se'])<1e-9
    confirmation_checks=0
    if (OUT/'confirmation_summary.json').exists():
        cf=read('confirmation_summary.json');ts=read('confirmation_tasks.json');vs={}
        for name in names:
            raw=read(f'confirmation_{name}.json');assert len(raw)==len(ts)==768
            groups=defaultdict(list)
            for t in ts:
                route,param=args_ok(raw[t['task_id']]['reply'],t)
                assert param==raw[t['task_id']]['arguments_ok'];groups[t['id']].append(param);confirmation_checks+=1
            assert len(groups)==96 and all(len(g)==8 for g in groups.values())
            vs[name]=[100*sum(groups[q])/8 for q in sorted(groups)]
            assert abs(sum(vs[name])/96-cf['means'][name])<1e-9
        diff,se=pair(vs[names[1]],vs[names[0]])
        assert abs(diff-cf['difference_pp'])<1e-9 and abs(se-cf['se'])<1e-9
    result=dict(passed=True,initial_raw_reply_decisions_checked=checks,initial_question_clusters_per_kind=24,
        confirmation_raw_reply_decisions_checked=confirmation_checks,independent_stdlib_paired_means_and_SE_match=True)
    (OUT/'validation_independent.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
