"""Task 5: local calculator routing under variable tool menus, with held-out wording."""
import os
from pathlib import Path
P=Path(__file__).resolve().parents[2]
OUT=P/'my_eval/results/tool_choice_round2_20261026'
RUNTIME=P/'my_eval/runtime'
os.environ.update(HF_HOME='D:/conda/hf-cache',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TEMP=str(RUNTIME),TMP=str(RUNTIME))
import ast
from collections import Counter
import copy
import hashlib
import json
import random
import re
import sys
sys.path[:0]=[str(P),str(P/'scripts'),str(P/'my_eval/scripts')]
from strict_math_rescore import arithmetic,expressions,number

BASE='bb_rl_id_anchor4'
BASE_HF=P/'scripts/MiniMind-YuyangLU-v2'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def save(name,x):
    OUT.mkdir(exist_ok=True,parents=True);p=OUT/name;t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n');t.replace(p)
def write_rows(name,x):
    s=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x);p=OUT/name
    if p.exists():assert p.read_text(encoding='utf-8')==s,name
    else:p.write_text(s,encoding='utf-8',newline='\n')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def norm(q):return re.sub(r'\s+','',q).casefold()
def question(r):return '\n'.join(m['content'] for m in r.get('conversations',r.get('chosen',[])) if m['role']=='user')
def signature(e):
    if not isinstance(e,str):raise ValueError('Expression must be a string')
    return ast.dump(ast.parse(e.strip(),mode='eval'),include_attributes=False)
def value(e):
    v=arithmetic(e)[0]
    assert v.denominator==1
    return str(v.numerator)

DESCRIPTIONS=[None,
 {'calculate_math':'对expression给定的算术式求值，支持四则运算和整数次幂。','get_current_time':'查询某个时区现在的日期和钟点。','random_number':'在上下界之间抽取一个随机整数。','text_length':'统计给定文本有多少个字符。','unit_converter':'把数值由原单位转换成另一个单位。','get_current_weather':'查询指定地点当前的气象情况。','get_exchange_rate':'获得源币种到目标币种的当前兑换比率。','translate_text':'把文本翻译为指定的目标语言。'},
 {'calculate_math':'执行数字表达式计算，返回精确计算结果。','get_current_time':'返回当地当前时间。','random_number':'按min和max范围随机取数。','text_length':'计算文本字符总数。','unit_converter':'处理长度、重量或温度的单位转换。','get_current_weather':'检索城市目前的天气。','get_exchange_rate':'检索两种货币之间的兑换行情。','translate_text':'执行语言翻译。'},
 {'calculate_math':'需要算出加减乘除或乘方的值时，传入expression字符串。','get_current_time':'需要知道当前日期或时间时使用。','random_number':'需要随机整数时使用。','text_length':'需要逐字统计长度时使用。','unit_converter':'需要变更计量单位时使用。','get_current_weather':'需要城市实时天气时使用。','get_exchange_rate':'需要实时外汇比价时使用。','translate_text':'需要将原文变成其他语言时使用。'},
 {'calculate_math':'数学求值服务：输入算式，输出数值。','get_current_time':'时钟服务，提供指定时区的当前时刻。','random_number':'随机整数服务。','text_length':'字数统计服务。','unit_converter':'计量单位换算服务。','get_current_weather':'地点天气查询服务。','get_exchange_rate':'货币兑换价格查询服务。','translate_text':'跨语言文本转换服务。'},
 {'calculate_math':'接收expression中的数学式并算出其值，可处理基本运算与幂。','get_current_time':'获取你所指定时区此刻的日期和时间。','random_number':'从指定整数区间抽样一个数。','text_length':'返回输入文字的字符数量。','unit_converter':'将输入量换算为目标计量单位。','get_current_weather':'读取某座城市的即时天气信息。','get_exchange_rate':'读取货币对的最新兑换比值。','translate_text':'将输入内容转写为目标语言。'}]

DESCRIPTIONS.append({'calculate_math':'把expression字符串中的算术运算求出数值，四则和乘方均可。','get_current_time':'读出所需时区的实时钟点与日期。','random_number':'在上下限内任选一个随机整数。','text_length':'返回所给文本的字符计数。','unit_converter':'把一个量转换到指定的计量单位。','get_current_weather':'查看目标地点目前的天气状态。','get_exchange_rate':'取得两币种之间的即时比价。','translate_text':'将文字译成所指定的语言。'})

def menu(count,variant,seed):
    # Tool names and JSON argument schema remain fixed; descriptions and order vary.
    from demo_tools import TOOLS
    rng=random.Random(seed);others=copy.deepcopy(TOOLS[1:]);rng.shuffle(others)
    tools=[copy.deepcopy(TOOLS[0])]+others[:count-1];rng.shuffle(tools)
    if variant:
        for t in tools:
            f=t['function'];f['description']=DESCRIPTIONS[variant][f['name']]
            for key,prop in f['parameters']['properties'].items():
                if key=='expression':prop['description']=['','待求值的算式文本','需要计算的表达式','输入数学算式','算术表达式字符串','作为字符串提供的算术表达式','需要求值的算术式文本'][variant]
    return tools

TEMPLATES={
 'train':['请计算{e}','帮我算一下{e}等于多少','Calculate {e}','Compute {e} for me','算算{e}','What is {e}?'],
 'dev':['我想核对这个算式的数值：{e}','Please find the numerical value of {e}.'],
 'test':['这几个数的运算麻烦替我核实：{e}','I need the value of this expression, please: {e}']}

def make_math(split,n,rng,blocked_questions,blocked_expressions):
    result=[];connect={'train':['，另外','，顺便','; also '],'dev':['；接着核对 ','; next, '],
                                'test':['；再帮我确认 ','; as a separate calculation, ']}[split]
    while len(result)<n:
        k=1+len(result)%3;es=[]
        for _ in range(k):
            a,b,c=[rng.randint(21,997) for _ in range(3)]
            mode=rng.randrange(6)
            e=[f'{a}+{b}',f'{a}-{b}',f'{a}*{rng.randint(2,49)}',
               f'({a}+{b})*{rng.randint(2,29)}',f'({a}-{b})+{rng.randint(2,90)}*{rng.randint(2,90)}',
               f'{(a+10)*17}/17'][mode]
            if signature(e) in blocked_expressions:break
            es.append(e)
        if len(es)!=k or len(set(map(signature,es)))!=k:continue
        q=rng.choice(TEMPLATES[split]).format(e=es[0])+''.join(rng.choice(connect)+e for e in es[1:])
        if norm(q) in blocked_questions:continue
        blocked_questions.add(norm(q));blocked_expressions.update(map(signature,es))
        assert [str(x[1].numerator) for x in expressions(q)]==[value(e) for e in es]
        result.append(dict(id=f'{split}_math_{len(result)}',kind='math',question=q,expressions=es,gt=[value(e) for e in es]))
    return result

NO_TOOL={
 'dev':[
 ('请说说你的名字以及定制开发者。','identity'),('State your name and custom developer.','identity'),
 ('用一句话解释文件夹的用途。','knowledge'),('Explain what a pencil is used for.','knowledge'),
 ('帮我写一句欢迎新同学的话。','writing'),('Write a short thank-you message for a classmate.','writing'),
 ('记录写着小林周二归还图书。小林哪天还书？','retrieval'),('The note says Mira chose the blue folder. Which folder did Mira choose?','retrieval'),
 ('给我两个安静的室内爱好。','knowledge'),('Explain the difference between a file and a folder.','knowledge'),
 ('写一句提醒同学保持教室整洁的话。','writing'),('我知道你有计算工具，不过这次只要一句鼓励的话。','writing')],
 'test':[
 ('请介绍这个助手的名称和定制开发者全名。','identity'),('Which person customized this assistant, and what is its name?','identity'),
 ('用一句简短的话说明什么是书签。','knowledge'),('Explain why people organize files into folders.','knowledge'),
 ('请替读书会写一句欢迎新成员的话。','writing'),('Write a friendly one-sentence reminder to bring a notebook.','writing'),
 ('纸条上写着小安选择绿色文件夹。小安选了什么颜色？','retrieval'),('A note says Nora returned the book on Friday. On which day did she return it?','retrieval'),
 ('讲一下天气和气候这两个词有什么不同。','knowledge'),('Explain what an exchange rate means, without looking up any current rates.','knowledge'),
 ('用一句话鼓励第一次尝试编程的同学。','writing'),('Write a warm birthday greeting for a friend.','writing'),
 ('记录是：小周拿走第27号卡片。请复述卡片编号，不用计算。','retrieval'),('The meeting record says room 308. Repeat the room number; no calculation is needed.','retrieval'),
 ('你有哪些能力？请简短介绍。','identity'),('Give a brief introduction to yourself and your developer.','identity'),
 ('请写两条整理书桌的小建议。','knowledge'),('Describe one simple way to plan a study session.','knowledge'),
 ('这次不是问今天的时间，请解释为什么人们使用时区。','knowledge'),('Explain why backups can be useful.','knowledge'),
 ('有很多工具可用，但我只需要你写一句晚安。','writing'),('Please give a short encouraging sentence, rather than a calculation.','writing'),
 ('便签：会议由李老师主持。请问主持人是谁？','retrieval'),('The record lists a red bag and a blue pen. What color is the pen?','retrieval')]}

def trajectory(case,tools):
    msgs=[dict(role='user',content=case['question'])]
    if case['kind']=='math':
        call='\n'.join('<tool_call>\n'+json.dumps(dict(name='calculate_math',arguments=dict(expression=e)),ensure_ascii=False)+'\n</tool_call>' for e in case['expressions'])
        msgs.append(dict(role='assistant',content=call))
        msgs.extend(dict(role='tool',content=json.dumps(dict(result=int(g)))) for g in case['gt'])
        msgs.append(dict(role='assistant',content='\n'.join(f'{e} = {g}。' for e,g in zip(case['expressions'],case['gt']))))
    else:msgs.append(dict(role='assistant',content=case['answer']))
    return dict(messages=msgs,tools=tools,kind='supervised',source_kind=case['kind'])

def expand(cases,split):
    out=[]
    counts=[1,6] if split=='dev' else [1,2,4,6]
    variants=[0,4] if split=='dev' else [0,5]
    for i,case in enumerate(cases):
        for count in counts:
            for variant in variants:
                seed=(520000 if split=='dev' else 530000)+i*100+count*10+variant
                out.append(dict(**case,task_id=f"{case['id']}_n{count}_v{variant}",tools=menu(count,variant,seed),
                    tool_count=count,variant=variant,seed=seed))
    return out

def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    assert not (OUT/'plan.json').exists(),'Prepared data are frozen; do not overwrite'
    reserved=rows(P/'dataset/agent_math_eval200.jsonl')+rows(P/'dataset/dpo_eval300.jsonl')
    idout=P/'my_eval/results/identity_repair_20261024'
    historical_chat=read(idout/'chat_final.json')['prompts']
    reserved_keys={norm(question(x)) for x in reserved}
    reserved_keys.update(norm(m['content']) for conv in historical_chat for m in conv if m['role']=='user')
    reserved_keys.update(norm(q) for split in NO_TOOL.values() for q,_ in split)
    historical_math_paths=[P/'dataset'/name for name in ['agent_math_train_3000.jsonl',
        'agent_math_train_3000_excl3000.jsonl','teacher_math_sft.jsonl','bb_mix.jsonl']]
    historical_math_paths.append(P/'my_eval/results/distill_control_20261025/teacher_new_sft.jsonl')
    blocked_expr=set();source_hashes={}
    for path in historical_math_paths:
        source_hashes[str(path.relative_to(P))]=sha(path)
        for row in rows(path)+([] if path!=historical_math_paths[0] else reserved[:200]):
            q=question(row)
            for expr,_,_ in expressions(q):
                try:blocked_expr.add(signature(expr))
                except SyntaxError:pass
    rng=random.Random(202610265)
    qkeys=set(reserved_keys);es=set(blocked_expr)
    math_train=make_math('train',960,rng,qkeys,es)
    math_dev=make_math('dev',12,rng,qkeys,es);math_test=make_math('test',24,rng,qkeys,es)
    other_dev=[dict(id=f'dev_no_{i}',kind='no_tool',question=q,category=cat) for i,(q,cat) in enumerate(NO_TOOL['dev'])]
    other_test=[dict(id=f'test_no_{i}',kind='no_tool',question=q,category=cat) for i,(q,cat) in enumerate(NO_TOOL['test'])]
    # Content-derived exclusion: no held-out questions may enter CE or KL retention.
    known_id=['你是谁？','Who created you?','你的作者叫什么名字？','是哪个人把你做出来的？','介绍一下你的开发者','Who is your developer?','你和 Jingyao Gong 是什么关系？']
    excluded=reserved_keys|{norm(q) for q in known_id}|{norm(c['question']) for c in math_dev+math_test}
    pool=[]
    for row in rows(idout/'general_train1000.jsonl'):
        conv=row['conversations'];q=question(row)
        if norm(q) in excluded or len(conv)!=2 or len(q)>160 or len(conv[-1]['content'])>360:continue
        if re.search(r'\d\s*[+*/-]|翻译|translate|今天天气|现在几点|汇率是|随机|多少.*字符',q,re.I):continue
        pool.append((q,conv[-1]['content']))
    unique=list(dict(pool).items());rng.shuffle(unique)
    ids=[]
    for row in rows(idout/'identity_train90.jsonl'):
        q=question(row)
        if norm(q) not in excluded:ids.append((q,row['conversations'][-1]['content']))
    ce_pairs=unique[:min(240,len(unique))]+ids[:60]
    assert len(ce_pairs)>=160,(len(unique),len(ids))
    train=[]
    for i,c in enumerate(math_train):train.append(trajectory(c,menu([1,2,4,6][i%4],(i//4)%4,540000+i)))
    for i,(q,a) in enumerate(ce_pairs):
        for rep in range(2):
            case=dict(kind='no_tool',question=q,answer=a)
            train.append(trajectory(case,menu([1,2,4,6][(i+rep)%4],(i+rep)%4,550000+i*2+rep)))
    retain=[]
    for row in rows(idout/'general_train1000.jsonl'):
        if norm(question(row)) not in excluded:
            retain.append(dict(messages=row['conversations'],tools=None,kind='retain',source_kind='general'))
        if len(retain)==400:break
    # Preserve historical canonical two-tool behavior through frozen-base KL.
    for row in rows(P/'dataset/teacher_math_sft.jsonl'):
        if norm(question(row)) in excluded:continue
        conv=copy.deepcopy(row['conversations']);tools=None
        if conv and conv[0]['role']=='system':tools=json.loads(conv.pop(0)['tools'])
        retain.append(dict(messages=conv,tools=tools,kind='retain',source_kind='math'))
        if sum(x['source_kind']=='math' for x in retain)==200:break
    train+=retain;rng.shuffle(train)
    dev=expand(math_dev+other_dev,'dev');test=expand(math_test+other_test,'test')
    write_rows('train.jsonl',train);save('dev_tasks.json',dev);save('test_tasks.json',test)
    save('guardrail_tasks.json',[dict(id=f'history_math_{i}',kind='math',question=question(row),gt=row['gt'],
        tools=json.loads(row['conversations'][0]['tools']),seed=540000+i) for i,row in enumerate(reserved[:200]) if i not in (15,125,181)])
    trainq={norm(m['content']) for r in train for m in r['messages'] if m['role']=='user'}
    devq={norm(c['question']) for c in dev};testq={norm(c['question']) for c in test}
    assert not trainq&devq and not trainq&testq and not devq&testq
    assert not trainq&{norm(question(r)) for r in reserved}
    save('plan.json',dict(date='2026-10-26',base=BASE,base_hf=str(BASE_HF),scope='calculator versus no call; other tools are distractors, not verified live integrations',
        train_counts=dict(Counter(x['source_kind']+'_'+x['kind'] for x in train)),training=dict(epochs=3,lr=.0002,kl=2.,batch=4,max_seq=1536,num_workers=2,seed=42,rank=16),
        candidates='single training recipe, epoch1/2/3 checkpoints; selected only on development routing and retention gates',
        dev=dict(base_questions=24,conditions=4,tasks=len(dev),description_variant=4),
        final=dict(base_questions=48,conditions=8,tasks=len(test),math=24,no_tool=24,description_variant=5,samples_per_condition=1),
        selection='rank checkpoints by balanced(math exact first-call arguments, no-tool no-call), requiring no-tool drop<=5pp and dev answerNLL increase<=.05 vs base; if none pass, retain best research checkpoint but do not add demo',
        acceptance='final math exact first-call argument gain95% paired cluster-bootstrap lower>0, no-tool no-call drop<=5pp, historical math strict point drop<=5pp, answerNLL296 increase<=.05, identity name hit drop<=1/14. No further training after final starts.',
        statistics='paired by underlying question, averaging8 conditions; 10000 bootstrap seed20261026, 1SE; exploratory, single training seed',
        tokenization='deterministic chat template, tools preserved, assistant-body/eos labels, no random preprocessing; rows beyond1536 excluded before train and reported',
        inference=dict(engine='HF Qwen3 fp16 for baseline and candidate',temperature=.9,top_p=.85,top_k=50,max_new_tokens=384,max_turns=3,thinking=False,best_of_n=1),
        overlap_audit=dict(ce_or_kl_vs_dev=0,ce_or_kl_vs_test=0,dev_vs_test=0,ce_or_kl_vs_historical_eval=0,
            historical_expression_signatures_blocked=len(blocked_expr),limitations='known local math pools scanned; no guarantee of upstream semantic novelty'),
        data_sha256={name:sha(OUT/name) for name in ['train.jsonl','dev_tasks.json','test_tasks.json','guardrail_tasks.json']},
        source_sha256=source_hashes,base_sha256=sha(P/f'out/{BASE}_768.pth')))
    print('Prepared:',len(train),'training rows;',len(dev),'dev conditions;',len(test),'final conditions',flush=True)

def load_model(name=BASE,fp32=False):
    import torch
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(BASE_HF,local_files_only=True,
        torch_dtype=torch.float32 if fp32 else torch.float16).cuda().eval()
    if name!=BASE:model.load_state_dict(torch.load(P/f'out/{name}_768.pth',map_location='cpu',weights_only=True),strict=True)
    return model

def load_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_HF,local_files_only=True)

def first_call_metrics(c,task):
    text=c['turns'][0] if c['turns'] else ''
    spans=re.findall(r'<tool_call>(.*?)</tool_call>',text,re.S);calls=[];malformed=False
    for raw in spans:
        try:
            call=json.loads(raw);assert isinstance(call,dict);calls.append(call)
        except (ValueError,AssertionError):malformed=True
    malformed|=text.count('<tool_call>')!=text.count('</tool_call>')
    malformed|='<tool_call' in text and not spans
    if task['kind']=='no_tool':
        return dict(route_ok=bool(text.strip()) and not calls and not malformed and '<tool_call>' not in text,
            arguments_ok=None,malformed=malformed,called=[x.get('name') for x in calls])
    names=bool(calls) and not malformed and all(x.get('name')=='calculate_math' for x in calls)
    try:
        actual=Counter(signature(x['arguments']['expression']) for x in calls)
        expected=Counter(signature(e) for e in task.get('expressions',[x[0] for x in expressions(task['question'])]))
        args=bool(names and actual==expected)
    except (KeyError,TypeError,ValueError,SyntaxError):args=False
    return dict(route_ok=bool(names),arguments_ok=args,malformed=malformed,called=[x.get('name') for x in calls])

if __name__=='__main__':prepare()
