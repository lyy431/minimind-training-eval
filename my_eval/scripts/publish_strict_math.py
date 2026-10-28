"""Validate numbers, document retrospective grading, and append project updates."""
import hashlib
import itertools
import json
from pathlib import Path
import re
import statistics as st

P=Path(__file__).resolve().parents[2]
ROOT=P/'my_eval';OUT=ROOT/'results/strict_math_20261026'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(OUT/'summary.json');rows=read(OUT/'final_results.json')
ids=read(ROOT/'results/distill_control_20261025/plan.json')['primary_math_indices']
for name,rs in rows.items():
    assert len(rs)==400 and [(r['problem'],r['sample']) for r in rs]==[(i,j) for i in range(200) for j in range(2)]
    vals=[100*st.mean(r['final_pass'] for r in rs if r['problem']==i) for i in ids]
    assert abs(st.mean(vals)-s['models'][name]['strict197'])<1e-10
for a,b in itertools.combinations(rows,2):
    diff=[100*(st.mean(r['final_pass'] for r in rows[a] if r['problem']==i)-st.mean(r['final_pass'] for r in rows[b] if r['problem']==i)) for i in ids]
    r=s['pairs'][a+'_minus_'+b]
    assert abs(st.mean(diff)-r['difference_pp'])<1e-10
    assert abs(st.stdev(diff)/len(diff)**.5-r['se'])<1e-10
for name,h in s['sha256'].items():assert sha(P/name)==h,name
assert sha(ROOT/'results/distill_control_20261025/math_final.json')=='337bd620ff0429f7a4a9e5ffafe23904f32d0c7ab3b4e12e59922fd03a81d0ea'
assert read(OUT/'regression_tests.json')['passed']==38
protocol={
 'version':'answer_consistency_v1','date':'2026-10-26','scope':list(rows),
 'design':'Retrospective evaluation repair using existing outputs, not a prospectively preregistered or independently held-out grader validation.',
 'primary_indices':ids,'historical_rows_retained':200,'samples_per_question':2,
 'rules_zh':[
  '只读最后一轮回答，去除思考部分和工具调用；工具返回值不能替代最终作答。未完成调用或空回答不通过。',
  '要求全部问题都有明确数值结果，数值和对应算式一致；回答中的数学断言不能自相矛盾。仅复述未求值的算式不通过。',
  '带算式的答案可换顺序；无标签数值列表须与题目顺序对应。重复同一答案不能填补漏题。',
  '使用受限AST和精确有理数检查加减乘除与幂；保留正负号，支持小数、科学计数法和规范千位分隔。中文逗号是分句标点。',
  '等价且数值一致的简化可接受。可辨识的排版瑕疵不扣分；改变运算、括号或对应关系造成的错误不能因目标数字出现而通过。',
  '自动解析不确定时返回review，不默认为错误；本轮所有review按题目和回答的摘要去重，由助手逐条裁决。'],
 'review_limits':'No independent blinded reviewer. Automatic pass/fail decisions were sampled, not all manually reviewed. Unsupported future formats require review.',
 'statistics':'Per-question paired1SE;10000 paired bootstrap resamples,seed20261026,percentile95%CI. Exploratory unadjusted pairwise intervals, conditional on fixed model weights/review labels; not training-seed uncertainty.',
 'source_sha256':s['sha256']}
(OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
changes=[{'model':n,**r} for n,rs in rows.items() for r in rs if r['problem'] in ids and r['legacy_pass']!=r['final_pass']]
(OUT/'changed_cases.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
table='| 模型 | 原数字包含率197题 | 完整数字诊断197题 | 新回答一致性通过率197题 | 新规则原200题 |\n|---|---:|---:|---:|---:|\n'
for n in s['ranking_by_point_estimate']:
    r=s['models'][n];table+=f"| {n} | {r['legacy197']:.2f}% | {r['whole_number197']:.2f}% | **{r['strict197']:.2f}%**（{r['correct_runs197']}/394次） | {r['strict200']:.2f}% |\n"
conclusion='新题蒸馏、旧题复习、原RL的点估计依次为72.84%、70.05%、68.27%，原蒸馏为57.61%。前三个版本之间的95%配对bootstrap区间均包含0，暂不能可靠排出先后；本次三者相对原蒸馏均有清楚改善。这不是证明前三者等价，也不是综合聊天能力排名。'
lines=['# 数学评测修正与可信排名','',
 '2026-10-26。本轮只重评上次对照的4个模型、1600次已保存回答。全部本地CPU运行，没有重新训练、生成或调用付费服务。','',
 '**结论：'+conclusion+'**','',table,
 '主比较统一使用此前已确定的197题×2次，继续排除已知训练同文题15、125、181（从0开始）。200题结果保留；本轮未重新选择测试题。新指标同时检查答案完整性和写出的算式是否一致，因此比只检查数字出现更严格，不能把三列当成同一种指标。','',
 '## 判分规则','']
lines += ['- '+x for x in protocol['rules_zh']]
lines += ['','不强制模型展示推导或调用计算器：单题直接给正确数值也可通过。若主动写出错误算式，则不能因最终数字碰巧正确而通过。','',
 '## 排名的不确定性','',
 '差值为前者−后者，单位百分点。±1SE按197道题配对计算，不把每题两次采样当作独立题；另以题为单位做10000次配对bootstrap，种子20261026。95%区间未经多重比较校正，是探索性统计，不包含训练种子或人工裁决误差。','',
 '| 比较 | 差值 ±1SE | 95% bootstrap区间 | 更好/更差/相同题数 |','|---|---:|---:|---:|']
for key,r in s['pairs'].items():
    a,b=key.split('_minus_');lo,hi=r['bootstrap95_percentile_pp']
    lines.append(f"| {a} − {b} | {r['difference_pp']:+.2f} ± {r['se']:.2f} | [{lo:+.2f}, {hi:+.2f}] | {r['better']}/{r['worse']}/{r['same']} |")
lines += ['','例如：新题组−复习组为+2.79 ±1.66个百分点，95%区间约[-0.25,+6.09]；新题组−原RL为+4.57 ±3.34，区间约[-2.03,+10.91]。不能仅按平均数写成确定的冠军排名。','',
 '## 按题目长度拆分','', '| 模型 | 1个算式 | 2个算式 | 3个算式 |','|---|---:|---:|---:|']
for n in s['ranking_by_point_estimate']:
    q=s['models'][n]['by_expression_count'];lines.append(f"| {n} | {q['1']:.2f}% | {q['2']:.2f}% | {q['3']:.2f}% |")
lines += ['','这里只描述本样本的表现：原RL在单算式题更强，新题组在多算式题更好。未单独验证这些子组差异的泛化稳定性。','',
 '## 核查与局限','',
 f"自动规则把134次回答列为需复核，按相同题目与完全相同回答去重为97条；已由本次助手逐条检查，当前没有待裁决项。通过/未通过及逐条理由保存在manual_reviews.json，摘要将裁决绑定到具体原文；不是独立人工盲评。",'',
 '另按每模型12个自动通过、6个自动不通过，固定种子20261026抽查72次（68条不同文本）。初次发现1条中文逗号被误当千位分隔符的解析错误，已统一修复、重评全部结果，并核对这68条全部一致。没有将抽查正确率包装成独立泛化准确率。','',
 '38项针对性回归检查覆盖数字子串、正负号、中文标点、小数、科学计数法、答案交换、重复/漏答、矛盾、算式未求值、隐藏分母、未知格式和极端数值。200题中全部目标值均由问题算式精确重算并与原GT核对。原始生成结果摘要保持不变；均值和6组配对标准误由独立统计实现交叉核验。','',
 '| 模型 | 旧通过→新不通过次数 /394 | 自动需复核次数 /400 | 复核后通过次数 /394 | 自动规则未裁决时的范围197题 |',
 '|---|---:|---:|---:|---:|']
for n,r in s['models'].items():
    lo,hi=r['automation_only_bounds197'];lines.append(f"| {n} | {r['old_pass_now_fail197']} | {r['automatic_counts200']['review']} | {r['review_passes197']} | {lo:.2f}%—{hi:.2f}% |")
lines += ['','最后一列把待裁决项全部当错/全部当对，只说明自动解析的不确定范围，不是统计置信区间。最终分数使用已记录的逐条裁决。自动解析器不是任意自然语言的完备判分器；新格式应继续进入复核流程。','',
 '这是看到历史输出后进行的回顾性评测修复，部分解析规则在排查这些输出时完善；不是事前注册的新盲测。本轮没有依据新成绩调模型。仍只检查了已知同文污染，不能保证不存在上游训练接触或语义改写。','',
 '## 典型旧误判','',
 '- 标准值29770，回答229770：数字包含匹配会误收，完整数值与算式校验不通过。',
 '- 问(476−555)+99×23，回答却写(476−555)+999×23=2198：目标数字正确，但写出的等式错误。',
 '- 前文三题答对，结尾又把第一题的15440改说成7409：前后矛盾，不能只挑正确片段得分。',
 '- 只重复4180/10，没有明确给出求值结果418：本轮按未完成数值作答处理；不声称两者数学上不等价。','',
 '主集合共161次旧通过改为新不通过，0次旧不通过改为新通过。changed_cases.json包含全部分歧原文，不只展示挑选的例子。','',
 '## 使用与复现','',
 '新脚本strict_math_rescore.py提供grade(text, question, gt)接口以及--input/--out命令行参数。输出pass/fail/review；review必须复核或报告上下界，不能静默当错。默认重评本次4模型，结果写入新的strict_math_20261026目录。没有修改冻结的历史评测脚本、历史分数、训练奖励或教师筛选器。','',
 '推荐后续固定报告：数值匹配率、完整回答一致性通过率、自动覆盖率/复核数量、逐题配对差异，明确评测口径。加入新的模型时，应对相同输出用相同版本重评，不能直接与旧数字包含率比较。','',
 '文件：protocol.json（规则和来源）；automatic.json（自动结果）；automatic_review_queue.json（需复核原文）；manual_reviews.json（97条裁决）；automatic_audit_sample.json及automatic_audit_review.json（抽查记录）；regression_tests.json；final_results.json；changed_cases.json；summary.json。','',
 '本轮只修正数学评测。此前语言loss结果不变：两组继续蒸馏约2.541，原RL约2.489；所以数学点估计领先不代表新题组综合能力更好。','']
(OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
validation={'passed':True,'means_checked':4,'paired_standard_errors_checked':6,'regression_cases':38,
    'original_math_file_unchanged':True,'reviewed_unique_texts':97,'reviewed_runs':134,
    'summary_sha256':sha(OUT/'summary.json'),'report_sha256':sha(OUT/'REPORT.md')}
(OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
body='2026-10-26完成数学评测修正，复用4模型1600次已保存回答，无需训练或生成。详见[数学评测修正报告](results/strict_math_20261026/REPORT.md)。\n\n'+conclusion+'\n\n'+table+'\n新规则要求每道题有明确数值答案、对应关系正确、写出的数学断言一致；单看数字包含或完整数字都不够。134次自动存疑输出（97条不同文本）已逐条复核，另抽查68条自动结果；38项回归检查通过。裁决由助手完成，并非独立人工盲评；这是回顾性修正，不是新的盲测。\n\n新题−复习：+2.79 ±1.66个百分点；新题−RL：+4.57 ±3.34，±为1SE。对应95%配对bootstrap区间分别[-0.25,+6.09]、[-2.03,+10.91]。前三者先后仍不确定。只针对这4模型的数学一致性，不是整个项目全部模型的综合排名；先前语言loss退步结论仍成立。\n\n新增strict_math_rescore.py、test_strict_math_rescore.py、report_strict_math.py、publish_strict_math.py；结果目录my_eval/results/strict_math_20261026。旧评测与训练奖励未改，后续应用新口径时应使用相同版本重评并处理review项。\n'
note='最新评测更新（2026-10-26）：数学已补充完整回答一致性评分；旧数字包含率保留为历史口径。新题组、复习组、原RL暂不能可靠排出先后，详见[修正报告](results/strict_math_20261026/REPORT.md)。'
for filename,title in [('README.md','## 12. 数学判分修正与可信排名（2026-10-26）'),
    ('SUMMARY_FOR_CHATGPT.md','## 13. 数学回答一致性重评（2026-10-26）'),
    ('HANDOFF.md','## 第六部分：2026-10-26 数学评测修正')]:
    file=ROOT/filename;text=file.read_text(encoding='utf-8')
    backup=OUT/(filename.removesuffix('.md')+'.before_strict_math.md')
    if not backup.exists():backup.write_text(text,encoding='utf-8',newline='\n')
    if note not in text:
        first,rest=text.split('\n',1);text=first+'\n\n'+note+'\n'+rest
    section=title+'\n\n'+body
    if title in text:
        start=text.index(title);later=re.search(r'^## ',text[start+len(title):],re.M)
        end=start+len(title)+later.start() if later else len(text)
        text=text[:start]+section+'\n'+text[end:]
    else:text=text.rstrip()+'\n\n---\n\n'+section
    file.write_text(text.rstrip()+'\n',encoding='utf-8',newline='\n')
print(json.dumps(validation,ensure_ascii=False));print('Published report and updated three project documents.')
