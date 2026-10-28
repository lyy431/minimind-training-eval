"""Question-clustered paired statistics and predeclared retention acceptance."""
from collections import Counter
import math
import statistics as st
import numpy as np
import tool_choice_v2 as e
from review_tool_choice_v2 import key

def paired(a,b):
    d=np.array(a,dtype=float)-np.array(b,dtype=float);n=len(d)
    boot=d[np.random.default_rng(20261026).integers(0,n,size=(10000,n))].mean(1)
    return dict(difference=float(d.mean()),se=float(d.std(ddof=1)/n**.5),ci95=np.quantile(boot,[.025,.975]).tolist(),
        better=int(sum(d>0)),worse=int(sum(d<0)),same=int(sum(d==0)),clusters=n)

def main():
    lock=e.read(e.OUT/'final_lock.json');names=[e.BASE,lock['selected']]
    tasks=e.read(e.OUT/'test_tasks.json');reviews=e.read(e.OUT/'math_reviews.json')
    summary=dict(base=e.BASE,candidate=names[1],models={},pairs={},subgroups={},review={},gates={})
    vectors={};guard={};loss={};identity={};outcomes={}
    for name in names:
        raw=e.read(e.OUT/f'final_{name}.json');assert len(raw)==384
        data=[]
        for task in tasks:
            r=raw[task['task_id']];assert r['question']==task['question']
            ok=None
            if task['kind']=='math':
                status=r['grade']['status']
                if status=='review':status=reviews[key(r['question'],r['final'])]['decision']
                assert status in ['pass','fail'];ok=status=='pass'
            data.append(dict(**task,route_ok=r['route_ok'],arguments_ok=r['arguments_ok'],strict_ok=ok,
                complete=r['eligible'],called=r['called'],empty=not r['final'].strip(),name_hit=r['name_hit']))
        outcomes[name]=data;vectors[name]={}
        for label,kind,field in [('math_name','math','route_ok'),('math_arguments','math','arguments_ok'),
                                 ('math_strict','math','strict_ok'),('no_tool','no_tool','route_ok')]:
            ids=sorted({t['id'] for t in tasks if t['kind']==kind})
            vectors[name][label]=[100*st.mean(r[field] for r in data if r['id']==i) for i in ids]
        m={label:st.mean(vals) for label,vals in vectors[name].items()}
        m['balanced_decision']=(m['math_arguments']+m['no_tool'])/2
        m['false_calls_percent']=100*st.mean(bool(r['called']) or '<tool_call' in raw[r['task_id']]['turns'][0]
            if raw[r['task_id']]['turns'] else False for r in data if r['kind']=='no_tool')
        m['completion_percent']=100*st.mean(r['complete'] for r in data)
        m['empty_count']=sum(r['empty'] for r in data)
        asked_names=lambda r:r['id'] in ['test_no_0','test_no_1','test_no_15']
        m['identity_with_tools_name_hits']=sum(r['name_hit'] for r in data if asked_names(r))
        m['identity_with_tools_total']=sum(asked_names(r) for r in data)
        summary['models'][name]=m
        summary['subgroups'][name]={}
        for count in [1,2,4,6]:
            for variant in [0,5]:
                rs=[r for r in data if r['tool_count']==count and r['variant']==variant]
                summary['subgroups'][name][f'n{count}_v{variant}']={
                    'math_arguments':100*st.mean(r['arguments_ok'] for r in rs if r['kind']=='math'),
                    'math_strict':100*st.mean(r['strict_ok'] for r in rs if r['kind']=='math'),
                    'no_tool':100*st.mean(r['route_ok'] for r in rs if r['kind']=='no_tool')}
        gr=e.read(e.OUT/f'guard_{name}.json');assert len(gr)==197;gv=[]
        for task in e.read(e.OUT/'guardrail_tasks.json'):
            r=gr[task['id']];status=r['grade']['status']
            if status=='review':status=reviews[key(r['question'],r['final'])]['decision']
            assert status in ['pass','fail'];gv.append(100*int(status=='pass'))
        guard[name]=gv;m['historical_math197']=st.mean(gv)
        loss[name]=e.read(e.OUT/f'loss_final_{name}.json')['rows'];assert len(loss[name])==296
        m['answer_nll296']=st.mean(r['loss'] for r in loss[name])
        identity[name]=e.read(e.OUT/f'identity_{name}.json');assert len(identity[name])==14
        m['identity_name_hits14']=sum(r['name_hit'] for r in identity[name].values())
    a,b=names[1],names[0]
    for metric in vectors[a]:summary['pairs'][metric]=paired(vectors[a][metric],vectors[b][metric])
    summary['pairs']['balanced_decision']=paired(vectors[a]['math_arguments']+vectors[a]['no_tool'],vectors[b]['math_arguments']+vectors[b]['no_tool'])
    summary['pairs']['historical_math197']=paired(guard[a],guard[b])
    assert [r['index'] for r in loss[a]]==[r['index'] for r in loss[b]]
    summary['pairs']['answer_nll296']=paired([r['loss'] for r in loss[a]],[r['loss'] for r in loss[b]])
    aa,bb=summary['models'][a],summary['models'][b]
    summary['gates']=dict(dev_selected_gates_passed=e.read(e.OUT/'selection.json')['dev_gates_passed'],
        exact_math_arguments_gain_ci_above0=summary['pairs']['math_arguments']['ci95'][0]>0,
        no_tool_drop_le5pp=aa['no_tool']>=bb['no_tool']-5,
        historical_strict_math_drop_le5pp=aa['historical_math197']>=bb['historical_math197']-5,
        answer_nll_increase_le005=aa['answer_nll296']<=bb['answer_nll296']+.05,
        identity_hit_drop_le1=aa['identity_name_hits14']>=bb['identity_name_hits14']-1)
    summary['accepted']=all(summary['gates'].values())
    summary['review']=dict(unique_records=len(reviews),all_uncertainties_resolved=True,
        reviewer='assistant; exact earlier question/text decisions may be reused; not independent human blind judging')
    e.save('final_outcomes.json',outcomes);e.save('summary.json',summary)
    print(e.json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
