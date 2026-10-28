"""Separate fresh-test statistics for round2, with48 math/24 ordinary question clusters."""
import hashlib
import statistics as st
import numpy as np
import tool_choice_round2 as e
from review_tool_choice_round2 import key

def paired(x,y):
    d=np.array(x,dtype=float)-np.array(y,dtype=float);n=len(d)
    boot=d[np.random.default_rng(20261027).integers(0,n,size=(10000,n))].mean(1)
    return dict(difference=float(d.mean()),se=float(d.std(ddof=1)/n**.5),ci95=np.quantile(boot,[.025,.975]).tolist(),
        better=int(sum(d>0)),worse=int(sum(d<0)),same=int(sum(d==0)),n=n)

def main():
    lock=e.read(e.OUT/'final_lock.json');names=[e.BASE,lock['selected']];tasks=e.read(e.OUT/'test_tasks.json')
    reviews=e.read(e.OUT/'math_reviews.json');summary=dict(base=names[0],candidate=names[1],models={},pairs={},subgroups={},gates={})
    vectors={};full={};guard={};loss={}
    def strict(r):
        status=r['grade']['status']
        if status=='review':status=reviews[key(r['question'],r['final'])]['decision']
        assert status in ['pass','fail'];return status=='pass'
    for name in names:
        raw=e.read(e.OUT/f'final_{name}.json');assert len(raw)==864
        vectors[name]={};m={}
        for metric,kind,field in [('math_name','math','route_ok'),('math_arguments','math','arguments_ok'),('no_tool','no_tool','route_ok')]:
            ids=sorted({t['id'] for t in tasks if t['kind']==kind})
            vals=[100*st.mean(raw[t['task_id']][field] for t in tasks if t['id']==i) for i in ids]
            vectors[name][metric]=vals;m[metric]=st.mean(vals)
        m['balanced_decision']=(m['math_arguments']+m['no_tool'])/2
        m['false_calls']=100*st.mean('<tool_call' in raw[t['task_id']]['final'] for t in tasks if t['kind']=='no_tool')
        summary['subgroups'][name]={}
        for count in range(1,7):
            for variant in [0,6]:
                ts=[t for t in tasks if t['tool_count']==count and t['variant']==variant]
                summary['subgroups'][name][f'n{count}_v{variant}']=dict(
                    math_arguments=100*st.mean(raw[t['task_id']]['arguments_ok'] for t in ts if t['kind']=='math'),
                    no_tool=100*st.mean(raw[t['task_id']]['route_ok'] for t in ts if t['kind']=='no_tool'))
        fm=e.read(e.OUT/f'fullmath_{name}.json');assert len(fm)==48;full[name]=[100*strict(fm[k]) for k in sorted(fm)]
        m['fullmath48']=st.mean(full[name])
        gr=e.read(e.OUT/f'guard_{name}.json');assert len(gr)==197;guard[name]=[100*strict(gr[k]) for k in sorted(gr)]
        m['historical_math197']=st.mean(guard[name])
        loss[name]=e.read(e.OUT/f'loss_final_{name}.json')['rows'];assert len(loss[name])==296
        m['answer_nll296']=st.mean(r['loss'] for r in loss[name])
        ids=e.read(e.OUT/f'identity_{name}.json');assert len(ids)==14
        m['identity_name_hits14']=sum(r['name_hit'] for r in ids.values())
        summary['models'][name]=m
    b,a=names
    for metric in ['math_name','math_arguments','no_tool']:summary['pairs'][metric]=paired(vectors[a][metric],vectors[b][metric])
    summary['pairs']['fullmath48']=paired(full[a],full[b]);summary['pairs']['historical_math197']=paired(guard[a],guard[b])
    assert [x['index'] for x in loss[a]]==[x['index'] for x in loss[b]]
    summary['pairs']['answer_nll296']=paired([x['loss'] for x in loss[a]],[x['loss'] for x in loss[b]])
    aa,bb=summary['models'][a],summary['models'][b]
    summary['gates']=dict(dev_gates=e.read(e.OUT/'selection.json')['dev_gates_passed'],
        fresh_math_argument_gain_ci_lower_gt0=summary['pairs']['math_arguments']['ci95'][0]>0,
        no_tool_ge90_and_drop_le5pp=aa['no_tool']>=max(90,bb['no_tool']-5),
        fresh_fullmath_drop_le5pp=aa['fullmath48']>=bb['fullmath48']-5,
        historical_math_drop_le5pp=aa['historical_math197']>=bb['historical_math197']-5,
        nll_increase_le005=aa['answer_nll296']<=bb['answer_nll296']+.05,
        identity_drop_le1=aa['identity_name_hits14']>=bb['identity_name_hits14']-1)
    summary['accepted']=all(summary['gates'].values());summary['review_records']=len(reviews)
    e.save('summary.json',summary);print(e.json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
