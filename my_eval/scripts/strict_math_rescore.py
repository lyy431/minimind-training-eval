"""CPU-only retrospective arithmetic adjudication. Never modifies historical scores.

Explicit answer claims, exact rational arithmetic, question association, and
conservative review status. No generated tool argument/result counts as an answer.
"""
import ast
import argparse
from collections import Counter
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import statistics as st
import unicodedata

P=Path(__file__).resolve().parents[2]
OUT=P/'my_eval/results/strict_math_20261026'
SOURCE=P/'my_eval/results/distill_control_20261025'
NUM=r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?'


def number(text):
    value=Decimal(text)
    if len(value.as_tuple().digits)>200 or abs(value.as_tuple().exponent)>1000:
        raise ValueError('number outside supported magnitude')
    return Fraction(value)


def save(name,value):
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')


def normalize(text):
    # A Chinese clause comma must never become a thousands separator.
    text=unicodedata.normalize('NFKC',text.replace('，',';')).replace('−','-').replace('–','-')
    # Remove presentation markup without deleting exponentiation operators.
    text=re.sub(r'\*\*('+NUM+r')\*\*',r'\1',text)
    text=text.replace('`','').replace('\\(','').replace('\\)','').replace('\\[','').replace('\\]','')
    text=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',text)
    for a,b in [('\\times','*'),('\\cdot','*'),('\\div','/'),('×','*'),('÷','/'),
        ('raised to the power of','**'),('multiplied by','*'),('divided by','/'),('times','*'),('plus','+'),('minus','-'),
        ('squared','**2'),('cubed','**3'),('的三次方','**3'),('的二次方','**2'),('的立方','**3'),('的平方','**2'),
        ('乘以','*'),('乘上','*'),('除以','/'),('加上','+'),('减去','-'),('加','+'),('减','-')]:
        pattern=re.escape(a)+(r'(?=\s*[\d(])' if a in ['plus','minus','times','multiplied by','divided by'] else '')
        text=re.sub(pattern,lambda _:b,text,flags=re.I)
    return text.replace('^','**')


def arithmetic(text):
    text=text.strip().rstrip('.')
    if len(text)>200:raise ValueError('expression too long')
    tree=ast.parse(text,mode='eval')
    def ev(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            return number(ast.get_source_segment(text,n))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
            a=ev(n.operand);return a if isinstance(n.op,ast.UAdd) else -a
        if not isinstance(n,ast.BinOp):raise ValueError('unsupported expression')
        a,b=ev(n.left),ev(n.right)
        if isinstance(n.op,ast.Add):v=a+b
        elif isinstance(n.op,ast.Sub):v=a-b
        elif isinstance(n.op,ast.Mult):v=a*b
        elif isinstance(n.op,ast.Div):v=a/b
        elif isinstance(n.op,ast.FloorDiv):v=Fraction(a//b)
        elif isinstance(n.op,ast.Mod):v=a%b
        elif isinstance(n.op,ast.Pow) and b.denominator==1 and abs(b)<=12:v=a**int(b)
        else:raise ValueError('unsupported operator')
        if v.numerator.bit_length()>1024 or v.denominator.bit_length()>1024:raise ValueError('too large')
        return v
    value=ev(tree.body)
    return value,ast.dump(tree,include_attributes=False)


def expressions(text):
    result=[]
    for match in re.finditer(r'[()\d.+*/\-]+(?:\s+[()\d.+*/\-]+)*',normalize(text)):
        value=match.group().strip().rstrip('.')
        if not re.search(r'\d\s*(?:[+*/-]|\))',value):continue
        try:
            number,key=arithmetic(value)
            if not isinstance(ast.parse(value,mode='eval').body,ast.BinOp):continue
            result.append((value,number,key))
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):pass
    return result


def final_text(row):
    if row.get('unfinished'):return ''
    if not row.get('turns'):return ''
    text=row['turns'][-1].split('</think>',1)[-1].strip()
    if '</tool_call>' in text:text=text.rsplit('</tool_call>',1)[-1].strip()
    return text


def grade(text,question,gt,unfinished=False):
    targets=[number(str(x)) for x in gt]
    requested=expressions(question)
    assert [x[1] for x in requested]==targets,('question/GT mismatch',question,requested,gt)
    if unfinished or not text.strip():return {'status':'fail','reason':'unfinished_or_empty','claims':[]}
    if re.search(r'\d,\d{3}\s*[/+*^\-]',text):
        return {'status':'review','reason':'ambiguous_ascii_comma','claims':[]}
    clean=normalize(text)
    if '{' in clean or '}' in clean or re.search(r'\\[A-Za-z]+',clean):
        return {'status':'review','reason':'unsupported_mathematical_markup','claims':[]}
    if re.search(r'\b(?:not|incorrect|instead|correction)\b|不是|不等于|更正|错误|抱歉',clean,re.I):
        return {'status':'review','reason':'negation_or_correction','claims':[]}
    if '<tool_call>' in clean or '<tool_response>' in clean:
        return {'status':'review','reason':'unexpected_tool_markup','claims':[]}
    # Claim delimiters must be followed by a scalar/expression. Input numbers alone are never answers.
    boundary=r'(?:=|\bis\b|\bare\b|相当于|等于|结果是|结果为|答案是|答案为|得到|得出|是|为)\s*[:：]?\s*'
    matches=list(re.finditer(boundary,clean,flags=re.I))
    claims=[];issues=[];covered=set();positional=[]
    for m in matches:
        rhs_match=re.match(NUM,clean[m.end():])
        if not rhs_match:continue
        raw=rhs_match.group();end=m.end()+rhs_match.end()
        following=clean[end:]
        if re.match(r'\s*[%万亿]',following):issues.append('scaled_number_requires_review');continue
        if re.match(r'\s*(?:[+*/^]|-(?=\s*\d))',following):
            # Chains are supported only when a later equality supplies a final scalar.
            if re.match(r'[^\n。;；]*=',following):issues.append('equation_chain_requires_review');continue
            issues.append('unevaluated_result');continue
        if re.match(r'[\w.]',following) and not following.startswith('.'):
            issues.append('unsupported_number_suffix');continue
        try:value=number(raw)
        except ValueError:return {'status':'review','reason':'unsupported_number_magnitude','claims':claims}
        prefix=clean[:m.start()]
        # Keep just the current sentence/list item; decimals inside expressions stay intact.
        prefix=re.split(r'\n|。|;|；|(?<=\d)\.(?=\s+[A-Za-z])|(?<!\d)\.(?!\d)',prefix)[-1]
        prefix=re.sub(r'^\s*\d+[.)、]\s*','',prefix)
        left=expressions(prefix)
        lhs=left[-1] if left else None
        scalar_prefix=prefix.strip()
        if lhs is None and re.fullmatch(NUM,scalar_prefix) and ('=' in m.group() or '等于' in m.group()):
            v,k=arithmetic(scalar_prefix);lhs=(scalar_prefix,v,k)
        if lhs is None and re.search(r'\d',scalar_prefix):
            issues.append('unparsed_numeric_lhs')
        claim={'value':str(value),'lhs':lhs[0] if lhs else None,'fragment':clean[max(0,m.start()-100):end]}
        claims.append(claim)
        if lhs:
            if lhs[1]!=value:
                return {'status':'fail','reason':'false_arithmetic_claim','claims':claims}
            exact=[i for i,x in enumerate(requested) if x[2]==lhs[2]]
            equivalent=[i for i,x in enumerate(targets) if x==lhs[1]]
            slots=exact or equivalent
            if len(slots)==1:covered.add(slots[0])
            else:issues.append('ambiguous_expression_association')
        else:positional.append(value)
    # Bare scalar/list answers are allowed only if the entire text is a numeric list.
    if not claims:
        bare=re.sub(r'^\s*(?:答案|结果|answers?|results?)\s*[:：]\s*','',clean,flags=re.I).strip().rstrip('。. ')
        parts=re.split(r'\s*[,，;；\n]\s*',bare)
        if parts and all(re.fullmatch(NUM,t) for t in parts):
            try:positional=[number(t) for t in parts]
            except ValueError:return {'status':'review','reason':'unsupported_number_magnitude','claims':[]}
            claims=[{'value':str(v),'lhs':None,'fragment':bare} for v in positional]
        else:return {'status':'review','reason':'no_supported_explicit_claim','claims':[]}
    if positional:
        if not covered and len(positional)==len(targets):
            if positional!=targets:return {'status':'fail','reason':'wrong_ordered_answers','claims':claims}
            covered.update(range(len(targets)))
        elif len(targets)==1:
            if any(v!=targets[0] for v in positional):return {'status':'fail','reason':'contradictory_or_wrong_answer','claims':claims}
            covered.add(0)
        else:issues.append('mixed_or_extra_positional_claims')
    if matches:
        clauses=re.split(r'\n|。|;|；|,|\band\b|(?<=\d)\.(?=\s+[A-Za-z])|(?<!\d)\.(?!\d)',clean,flags=re.I)
        for clause in clauses:
            if re.search(r'\d',clause) and not re.search(boundary+NUM,clause,re.I):
                issues.append('unparsed_numeric_clause')
    if issues:return {'status':'review','reason':','.join(sorted(set(issues))),'claims':claims}
    if covered!=set(range(len(targets))):return {'status':'review','reason':'missing_or_unparsed_answer','claims':claims}
    return {'status':'pass','reason':'all_explicit_claims_consistent_and_covered','claims':claims}


def run(source_path=None):
    questions=[json.loads(x) for x in (P/'dataset/agent_math_eval200.jsonl').read_text(encoding='utf-8').splitlines()]
    raw=json.loads(Path(source_path or SOURCE/'math_final.json').read_text(encoding='utf-8'))
    output={}
    for name,rows in raw.items():
        output[name]=[]
        for j,row in enumerate(rows):
            assert row['gt']==questions[row['problem']]['gt']
            q=questions[row['problem']]['conversations'][1]['content']
            result=grade(final_text(row),q,row['gt'],row['unfinished'])
            output[name].append({'problem':row['problem'],'sample':j%2,'question':q,'gt':row['gt'],
                'final':final_text(row),'legacy_pass':row['all'],**result})
    save('automatic.json',output)
    queue={}
    for rs in output.values():
        for r in rs:
            if r['status']=='review':
                h=hashlib.sha256((r['question']+'\n'+r['final']).encode()).hexdigest()
                queue[h]={'sha256':h,**{k:r[k] for k in ['question','gt','final','reason']}}
    save('automatic_review_queue.json',list(queue.values()))
    for n,rs in output.items():print(n,dict(Counter(r['status'] for r in rs)),dict(Counter(r['reason'] for r in rs if r['status']=='review')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',help='JSON mapping model names to historical eval_agent_math rows')
    parser.add_argument('--out',help='Separate output directory; never use a historical results directory')
    args=parser.parse_args()
    if args.out:
        OUT=Path(args.out).resolve()
        assert OUT!=SOURCE.resolve()
    run(args.input)
