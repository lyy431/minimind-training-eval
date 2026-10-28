"""Second experiment: broader intent wording and numeric non-computation negatives.

Round1 remains sealed. Its seen tests are development diagnostics, never fresh proof.
"""
import tool_choice_round2 as e

OLD=e.P/'my_eval/results/tool_choice_20261026'
TRAIN_WORDING=[
 '请计算{e}','帮我算出{e}的结果','核实一下{e}的数值','替我核对{e}算出来是多少','请对{e}求值',
 '麻烦算算{e}','我想知道{e}等于多少','请把{e}的计算结果告诉我','帮我确认{e}的数值结果',
 '请算一下这项运算：{e}','给出下面算式的数值：{e}','麻烦核算：{e}','我要一个准确的运算结果：{e}',
 'Calculate {e}','Compute {e} for me','Evaluate {e}','Find the result of {e}',
 'Please work out {e}','Could you check the numerical result of {e}?','Give me the answer to {e}',
 'I want to know what {e} equals','Please calculate the value of {e}','Check this calculation: {e}',
 'Can you solve this arithmetic expression: {e}?']
TEST_WORDING=[
 '请直接核实以下运算的结果：{e}','算清楚这一项并给我数值答案：{e}','按计算规则给出确切结果：{e}','帮我完成这项数值求解：{e}',
 'What numerical result do you obtain from {e}?','Please solve the following calculation and give its value: {e}',
 'Help me evaluate this arithmetic task: {e}','Can you find the exact answer for the expression {e}?']
FINAL_NO=[
 ('请用一句话交代你的模型名称以及定制者姓名。','identity'),('Tell me the name of this assistant and the person who customized it.','identity'),
 ('请用简单的话解释目录和文件的区别。','knowledge'),('Explain why a person might make a backup copy of a document.','knowledge'),
 ('用一句话告诉我读书有什么乐趣。','knowledge'),('Describe the purpose of a study timetable.','knowledge'),
 ('帮我写一句欢迎新同事加入小组的话。','writing'),('Write one friendly sentence thanking a teammate for their help.','writing'),
 ('替同学写一句简短的早安问候。','writing'),('Write a short good-luck message for a friend taking an exam.','writing'),
 ('备忘录中写着：活动安排在周四。请问星期几开展活动？','retrieval'),('The note says Hana chose a yellow folder. Which color did she choose?','retrieval'),
 ('记录里的会议室编号为619。请原样复述房间编号，不要做计算。','retrieval'),('The room number in the record is 427. Repeat that identifier without calculating anything.','retrieval'),
 ('名单显示主持人为陈老师。请说出主持人的称呼。','retrieval'),('The list says the pen is green and the bag is brown. What color is the bag?','retrieval'),
 ('包裹标签是第82号。请只读出标签编号。','retrieval'),('A record lists item code 905. State the code exactly as written.','retrieval'),
 ('只抄写引号内的字符串，不求它的值：“318+76”。','literal'),('Copy the literal text "641-58" without evaluating the expression.','literal'),
 ('把“27*46”原样再写一次，这次不要算答案。','literal'),('Repeat the string "987/21" as text; do not perform arithmetic.','literal'),
 ('请解释为什么“有数字”并不一定代表需要计算。','knowledge'),('Explain what an arithmetic expression is, without solving an example.','knowledge')]

def expand(cases,dev=False):
    result=[]
    for i,c in enumerate(cases):
        for count in ([1,6] if dev else [1,2,3,4,5,6]):
            for variant in ([0,4] if dev else [0,6]):
                seed=910000+i*100+count*10+variant
                result.append(dict(**c,task_id=f"{c['id']}_n{count}_v{variant}",tools=e.menu(count,variant,seed),
                    tool_count=count,variant=variant,seed=seed))
    return result

def main():
    e.OUT.mkdir(parents=True,exist_ok=True);assert not (e.OUT/'plan.json').exists()
    old=e.rows(OLD/'train.jsonl');rng=e.random.Random(202610267);qs=set();es=set()
    for r in old:
        for m in r['messages']:
            if m['role']=='user':
                qs.add(e.norm(m['content']))
                for ex,_,_ in e.expressions(m['content']):es.add(e.signature(ex))
    for name in ['dev_tasks.json','test_tasks.json','guardrail_tasks.json','confirmation_tasks.json']:
        for r in e.read(OLD/name):
            qs.add(e.norm(r['question']))
            for ex,_,_ in e.expressions(r['question']):es.add(e.signature(ex))
    for name in e.read(OLD/'plan.json')['source_sha256']:
        for r in e.rows(e.P/name):
            for ex,_,_ in e.expressions(e.question(r)):es.add(e.signature(ex))
    qs.update(e.norm(q) for q,_ in FINAL_NO)
    e.TEMPLATES['train']=TRAIN_WORDING
    math_train=e.make_math('train',1440,rng,qs,es)
    train=[e.trajectory(c,e.menu(i%6+1,(i//6)%4,920000+i)) for i,c in enumerate(math_train)]
    train.extend(r for r in old if r['kind']=='retain' or r['source_kind']=='no_tool')
    for i in range(246):
        n=1000+i;a=200+i;b=10+i%80
        patterns=[(f'编号是{n}。只需原样告诉我编号，不要运算。',str(n)),
                  (f'The identifier is {n}. Please repeat the identifier, not a calculation.',str(n)),
                  (f'抄写字符串“{a}+{b}”，不要计算它。',f'{a}+{b}'),
                  (f'Copy the literal string "{a}*{b}" as text, without evaluating it.',f'{a}*{b}'),
                  (f'档案写着会议在{n}号房间。请复述房间号。',str(n)),
                  (f'The product code is {n}; tell me the product code exactly.',str(n))]
        q,answer=patterns[i%6];assert e.norm(q) not in qs;qs.add(e.norm(q))
        train.append(e.trajectory(dict(kind='no_tool',question=q,answer=answer),e.menu(i%6+1,(i//6)%4,930000+i)))
    rng.shuffle(train)
    # Previously seen tests are explicitly reused for development only.
    prior_math=[];seen=set()
    for r in e.read(OLD/'test_tasks.json'):
        if r['kind']=='math' and r['id'] not in seen:seen.add(r['id']);prior_math.append(r)
    dev_math=[]
    for english in [False,True]:
        selected=[r for r in prior_math if r['question'].startswith('I need')==english][:6]
        for r in selected:dev_math.append({k:r[k] for k in ['id','kind','question','expressions','gt']})
    dev_no=[];seen=set()
    for r in e.read(OLD/'dev_tasks.json'):
        if r['kind']=='no_tool' and r['id'] not in seen:
            seen.add(r['id']);dev_no.append({k:r[k] for k in ['id','kind','question','category']})
    dev=expand(dev_math+dev_no,True)
    fresh=e.make_math('test',48,rng,qs,es)
    for i,c in enumerate(fresh):
        template=TEST_WORDING[(i//3)%8];sep='；还要算：' if (i//3)%8<4 else '; and separately, '
        c['question']=template.format(e=c['expressions'][0])+''.join(sep+x for x in c['expressions'][1:])
        c['id']='r2_math_'+str(i)
    no=[dict(id=f'r2_no_{i}',kind='no_tool',question=q,category=cat) for i,(q,cat) in enumerate(FINAL_NO)]
    final=expand(fresh+no)
    tq={e.norm(m['content']) for r in train for m in r['messages'] if m['role']=='user'}
    assert not tq&{e.norm(r['question']) for r in dev+final}
    e.write_rows('train.jsonl',train);e.save('dev_tasks.json',dev);e.save('test_tasks.json',final)
    e.save('guardrail_tasks.json',e.read(OLD/'guardrail_tasks.json'))
    e.save('plan.json',dict(date='2026-10-26',base=e.BASE,experiment='round2 after sealed unsuccessful first experiment',
        prior_experiment=str(OLD),prior_tests_are_development_only=True,
        training=dict(epochs=3,lr=.0001,kl=4.,batch=4,max_seq=1536,num_workers=2,seed=42,rank=16),
        math_early_tokens_weight='first64 supervised math answer tokens weight3, other supervised tokens weight1; row normalized',
        counts=dict(e.Counter(r['source_kind']+'_'+r['kind'] for r in train)),
        dev=dict(math=12,no_tool=12,conditions=4,tasks=len(dev),minimum_no_tool=85),
        final=dict(math=48,no_tool=24,conditions=12,tasks=len(final),wording_families=8,balanced_math_language_and_expression_counts=True),
        selection='3 epoch checkpoints; require no-tool>=85% and devNLL increase<=.05; maximize min(math arguments,no-tool), then balanced mean; freeze before final',
        acceptance='fresh math parameter gain95%CI lower>0; no-tool>=90% and no worse than baseline-5pp; NLL296 increase<=.05; old strictmath drop<=5pp; identityhits drop<=1/14',
        no_optional_stopping=True,no_further_training_after_final=True,
        data_sha256={n:e.sha(e.OUT/n) for n in ['train.jsonl','dev_tasks.json','test_tasks.json','guardrail_tasks.json']},
        base_sha256=e.sha(e.P/f'out/{e.BASE}_768.pth'),known_question_overlap_train_vs_dev_or_final=0))
    print('Round2 prepared:',len(train),'train rows,',len(dev),'dev,',len(final),'new final conditions.',flush=True)

if __name__=='__main__':main()
