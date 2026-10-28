"""Synthetic adversarial regression fixtures, independent of saved model results."""
from strict_math_rescore import grade,save

CASES=[
 ('basic','Calculate 2+3',['5'],'2+3=5','pass'),
 ('substring','Calculate 2+3',['5'],'The result is 15.','fail'),
 ('sign','Calculate 2-7',['-5'],'The result is 5.','fail'),
 ('unicode_sign','Calculate 2-7',['-5'],'结果是−5。','pass'),
 ('zero','Calculate 2-2',['0'],'结果是0。','pass'),
 ('decimal','Calculate 2+3',['5'],'The result is 5.000.','pass'),
 ('scientific','Calculate 2+3',['5'],'The result is 5e0.','pass'),
 ('thousands','Calculate 1000+234',['1234'],'The result is 1,234.','pass'),
 ('input_only','Calculate 2+3',['5'],'Please calculate 2+3','review'),
 ('no_substring_from_input','Calculate 2+3',['5'],'15-10','review'),
 ('false_equation_right_number','Calculate 2+3',['5'],'3+3=5','fail'),
 ('swapped_labels','Calculate 2+3. And 3+4',['5','7'],'2+3=7; 3+4=5','fail'),
 ('swapped_bare','Calculate 2+3. And 3+4',['5','7'],'7,5','fail'),
 ('ordered_bare','Calculate 2+3. And 3+4',['5','7'],'5,7','pass'),
 ('reordered_labelled','Calculate 2+3. And 3+4',['5','7'],'3+4=7; 2+3=5','pass'),
 ('missing','Calculate 2+3. And 3+4',['5','7'],'2+3=5','review'),
 ('duplicate_not_coverage','Calculate 2+3. And 3+4',['5','7'],'2+3=5; 2+3=5','review'),
 ('same_gt_distinct','Calculate 2+3. And 1+4',['5','5'],'2+3=5; 1+4=5','pass'),
 ('same_gt_missing','Calculate 2+3. And 1+4',['5','5'],'2+3=5','review'),
 ('contradiction','Calculate 2+3',['5'],'2+3=5. The result of 2+3 is 6.','fail'),
 ('numeric_identity','Calculate 2+3',['5'],'7=5','fail'),
 ('no_raw_tool_credit','Calculate 2+3',['5'],'<tool_response>5</tool_response>','review'),
 ('unevaluated','Calculate 10/2',['5'],'The result of 10/2 is 10/2.','review'),
 ('chain','Calculate 2+3',['5'],'2+3=6-1=5','review'),
 ('bold','Calculate 2+3',['5'],'The result of 2+3 is **5**.','pass'),
 ('latex','Calculate 2*3',['6'],r'\(2 \times 3 = 6\)','pass'),
 ('power','Calculate 3**2',['9'],'3的平方是9。','pass'),
 ('power_words','Calculate 3**2',['9'],'The result of 3 raised to the power of 2 is 9.','pass'),
 ('percent','Calculate 2+3',['5'],'The result is 5%.','review'),
 ('negation','Calculate 2+3',['5'],'The result is not 5.','review'),
 ('unparsed_contradiction','Calculate 2+3',['5'],'2+3=5. However 2+3:6','review'),
 ('unparsed_contradiction_comma','Calculate 2+3',['5'],'2+3=5, 2+3:6','review'),
 ('hidden_denominator','Calculate 2+3',['5'],r'\frac{2+3}{2}=5','review'),
 ('unknown_lhs','Calculate 3**2',['9'],'The result of 3 squared (3**3) is 9.','review'),
 ('chinese_clause_comma','Calculate 4**2. And 510/5',['16','102'],'4的平方等于16，510除以5等于102.0。','pass'),
 ('ambiguous_ascii_comma','Calculate 4**2. And 510/5',['16','102'],'4**2=16,510/5=102','review'),
 ('bounded_number','Calculate 2+3',['5'],'The result is 1e999999999','review'),
]
rows=[]
for name,q,gt,text,expected in CASES:
    actual=grade(text,q,gt)
    assert actual['status']==expected,(name,expected,actual)
    rows.append({'case':name,'expected':expected,'actual':actual['status']})
assert grade('2+3=5','Calculate 2+3',['5'],unfinished=True)['status']=='fail'
save('regression_tests.json',{'passed':len(rows)+1,'cases':rows,'unfinished_checked':True})
print(f'{len(rows)+1} adversarial regression checks passed.')
