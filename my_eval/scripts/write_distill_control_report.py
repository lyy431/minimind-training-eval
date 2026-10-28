"""Recompute task-3 paired statistics and write the complete local experiment report."""
import json
import re
import statistics as st

import distill_control as d

c,P,OUT,save=d.c,d.P,d.OUT,d.save
NAMES=['bb_dist','bb_sft_replay','bb_sft_new','bb_rl']
LABELS={'bb_dist':'原蒸馏模型','bb_sft_replay':'继续复习旧题','bb_sft_new':'加入新题再蒸馏','bb_rl':'原蒸馏后 RL'}
PAIRS=[('bb_sft_replay','bb_dist'),('bb_sft_new','bb_sft_replay'),
    ('bb_sft_new','bb_dist'),('bb_rl','bb_dist'),('bb_sft_new','bb_rl')]


def paired(a,b,higher=True):
    assert len(a)==len(b) and len(a)>1
    diff=[x-y for x,y in zip(a,b)]
    direction=1 if higher else -1
    return {'n':len(diff),'difference':st.mean(diff),'se':st.stdev(diff)/len(diff)**.5,
        'better':sum(direction*x>1e-12 for x in diff),
        'worse':sum(direction*x< -1e-12 for x in diff),
        'same':sum(abs(x)<=1e-12 for x in diff)}


def math_diagnostic(row):
    answers=[t.split('</think>',1)[-1].strip() for t in row['turns']]
    calls=[call for a in answers for call in c.parse_tool_calls(a)]
    gt=[float(str(g).replace(',','')) for g in row['gt']]
    values=[];invalid=0
    for call in calls:
        if call.get('name')!='calculate_math':continue
        args=call.get('arguments',{})
        if isinstance(args,str):
            try:args=json.loads(args)
            except Exception:args={}
        try:values.append(float(d.safe_math_eval(str(args.get('expression','')))))
        except Exception:invalid+=1
    computed=[g for g in gt if any(abs(v-g)<1e-6 for v in values)]
    last=answers[-1] if answers else ''
    final='' if row['unfinished'] else (last.split('</tool_call>')[-1] if '</tool_call>' in last else last)
    nums=[float(s) for s in re.findall(r'(?<![0-9A-Za-z_.])[-+]?\d+(?:\.\d+)?(?![0-9A-Za-z_]|\.\d)',final.replace(',',''))]
    whole=all(any(abs(v-g)<1e-6 for v in nums) for g in gt)
    wrong=any(call.get('name')!='calculate_math' for call in calls)
    ncalc=sum(call.get('name')=='calculate_math' for call in calls)
    reported=c.validate_gt_in_text(final,[str(int(g)) if g==int(g) else str(g) for g in computed]) if computed else set()
    return {'problem':row['problem'],'all':row['all'],'whole_numbers':whole,
        'strict_tool_trace':whole and len(computed)==len(gt) and not wrong and invalid==0 and not row['unfinished'],
        'wrong_tool':wrong,'invalid_expressions':invalid,'unfinished':row['unfinished'],
        'n_calc':ncalc,'n_gt':len(gt),'expr_right':len(computed)/len(gt),
        'reported':len(reported)/len(computed) if computed else None,
        'legacy_only_hit':row['all'] and not whole}


def main():
    from eval_distill_control import verify
    lock=verify()
    plan=c.read(OUT/'plan.json');audit=c.read(OUT/'data_audit.json')
    maths=c.read(OUT/'math_final.json')
    chats=c.read(OUT/'chat_final.json')['results']
    losses=c.read(OUT/'loss_final.json')
    for name in NAMES:
        assert len(maths[name])==400 and [r['problem'] for r in maths[name]]==[i for i in range(200) for _ in range(2)]
        assert [r['gt'] for r in maths[name]]==[r['gt'] for r in maths['bb_dist']]
        assert len(chats[name])==90 and [r['prompt'] for r in chats[name]]==[i for i in range(30) for _ in range(3)]
        assert len(losses[name]['losses'])==296 and losses[name]['indices']==losses['bb_dist']['indices']
    diag={n:[math_diagnostic(r) for r in maths[n]] for n in NAMES}
    save('math_diagnostics.json',diag)
    summary={'models':{},'comparisons':{},'primary_math_n':197,'primary_chat_n':29,
        'language_n':296,'uncertainty':'Paired per-question1SE, not95%CI; excludes training-seed variation.',
        'limits':plan['limits'],'known_test_question_overlaps':plan['known_test_question_overlaps'],
        'known_chat_question_overlaps':plan['known_chat_question_overlaps'],'data_audit':audit,
        'training':lock['training'],'lock_utc':lock['locked_utc']}
    for n in NAMES:
        primary=[r for r in maths[n] if r['problem'] in plan['primary_math_indices']]
        primary_chat=[r for r in chats[n] if r['prompt'] in plan['primary_chat_indices']]
        primary_diag=[r for r in diag[n] if r['problem'] in plan['primary_math_indices']]
        summary['models'][n]={
            'math197_percent':100*st.mean(r['all'] for r in primary),
            'math200_percent':100*st.mean(r['all'] for r in maths[n]),
            'math197_correct_runs':sum(r['all'] for r in primary),
            'loss296':losses[n]['mean'],
            'chat29_rm':st.mean(r['rm'] for r in primary_chat),
            'chat30_rm':st.mean(r['rm'] for r in chats[n]),
            'chat29_total':st.mean(r['total'] for r in primary_chat),
            'chat30_total':st.mean(r['total'] for r in chats[n]),
            'chat29_answer_chars':st.mean(r['answer_chars'] for r in primary_chat),
            'math197_by_expression_count':{k:100*st.mean(r['all'] for r in primary if len(r['gt'])==k) for k in [1,2,3]},
            'math197_diagnostics':{
                'whole_number_accuracy_percent':100*st.mean(r['whole_numbers'] for r in primary_diag),
                'strict_tool_trace_percent':100*st.mean(r['strict_tool_trace'] for r in primary_diag),
                'legacy_only_hit_runs':sum(r['legacy_only_hit'] for r in primary_diag),
                'wrong_tool_percent':100*st.mean(r['wrong_tool'] for r in primary_diag),
                'calculator_count_equal_percent':100*st.mean(r['n_calc']==r['n_gt'] for r in primary_diag),
                'calculator_count_less_percent':100*st.mean(r['n_calc']<r['n_gt'] for r in primary_diag),
                'calculator_count_more_percent':100*st.mean(r['n_calc']>r['n_gt'] for r in primary_diag),
                'expr_right_percent':100*st.mean(r['expr_right'] for r in primary_diag),
                'reported_percent':100*st.mean(r['reported'] for r in primary_diag if r['reported'] is not None),
                'unfinished_percent':100*st.mean(r['unfinished'] for r in primary_diag)}}
    math_vector=lambda n,ids,field='all':[100*st.mean(r[field] for r in diag[n] if r['problem']==i) for i in ids]
    chat_vector=lambda n,ids,field:[st.mean(r[field] for r in chats[n] if r['prompt']==i) for i in ids]
    for a,b in PAIRS:
        summary['comparisons'][a+'_minus_'+b]={
            'math197_pp':paired(math_vector(a,plan['primary_math_indices']),math_vector(b,plan['primary_math_indices'])),
            'math200_pp':paired(math_vector(a,range(200)),math_vector(b,range(200))),
            'whole_number197_pp':paired(math_vector(a,plan['primary_math_indices'],'whole_numbers'),math_vector(b,plan['primary_math_indices'],'whole_numbers')),
            'strict_tool197_pp':paired(math_vector(a,plan['primary_math_indices'],'strict_tool_trace'),math_vector(b,plan['primary_math_indices'],'strict_tool_trace')),
            'loss296':paired(losses[a]['losses'],losses[b]['losses'],higher=False),
            'chat29_rm':paired(chat_vector(a,plan['primary_chat_indices'],'rm'),chat_vector(b,plan['primary_chat_indices'],'rm')),
            'chat30_rm':paired(chat_vector(a,range(30),'rm'),chat_vector(b,range(30),'rm')),
            'chat29_total':paired(chat_vector(a,plan['primary_chat_indices'],'total'),chat_vector(b,plan['primary_chat_indices'],'total')),
            'chat30_total':paired(chat_vector(a,range(30),'total'),chat_vector(b,range(30),'total'))}
    raw=c.read(OUT/'teacher_raw.json')
    summary['teacher']={'attempted':len(raw['rows']),'accepted':sum(r['accepted'] for r in raw['rows']),
        'generation_seconds_sum':sum(raw['batch_seconds']),
        'peak_gpu_allocated_gib':raw['peak_gpu_allocated_gib'],'peak_gpu_reserved_gib':raw['peak_gpu_reserved_gib']}
    summary['artifacts_sha256']={str(f.relative_to(P)):c.digest(f) for f in [
        OUT/'math_final.json',OUT/'chat_final.json',OUT/'loss_final.json',OUT/'teacher_raw.json',
        OUT/'evaluation_lock.json',P/'my_eval/scripts/write_distill_control_report.py']}
    if (OUT/'grading_disagreement_review.json').exists():
        summary['grading_disagreement_review']=c.read(OUT/'grading_disagreement_review.json')
    for filename in ['teacher_quality_review.json','grading_disagreement_review.json','interpretation.md']:
        if (OUT/filename).exists():
            summary['artifacts_sha256'][str((OUT/filename).relative_to(P))]=c.digest(OUT/filename)
    save('summary.json',summary)
    lines=['# 补对照实验：新增数据、继续蒸馏与 RL','',
        '2026-10-25，在本地 RTX 4070 Laptop 8GB 完成。模型、数据、缓存均位于 D 盘；没有使用付费云服务。','']
    if (OUT/'interpretation.md').exists():
        lines += [(OUT/'interpretation.md').read_text(encoding='utf-8').strip(),'']
    lines += ['## 1. 四个版本的结果','',
        '主要比较排除了已知训练同文问题：数学197题×2次、聊天29题×3次。原200题、30题分数仍列出，用于与历史记录对照。','',
        '| 模型 | 含义 | 数学197题 | 原200题 | 回答loss296（低好） | 聊天29题奖励模型分 | 聊天29题总奖励 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for n,r in summary['models'].items():
        lines.append(f"| `{n}` | {LABELS[n]} | {r['math197_percent']:.2f}% | {r['math200_percent']:.2f}% | {r['loss296']:.4f} | {r['chat29_rm']:.4f} | {r['chat29_total']:.4f} |")
    lines += ['','原30题的奖励模型分 / 总奖励：'+ '；'.join(f"{n} {r['chat30_rm']:.4f} / {r['chat30_total']:.4f}" for n,r in summary['models'].items())+'。','',
        '## 2. 配对差异','',
        '差值为表中前者−后者；数学单位为百分点。±是1个标准误，不是95%置信区间。数学先把每题2次采样平均，聊天先把每题3次平均，再逐题配对。语言按296条回答配对。这些误差条不包含换训练随机种子的波动。','',
        '| 比较 | 数学197题 ±1SE | 好/差/相同题数 | loss296差值 ±1SE | 聊天29题奖励模型差值 ±1SE |',
        '|---|---:|---:|---:|---:|']
    for a,b in PAIRS:
        r=summary['comparisons'][a+'_minus_'+b];m,l,t=r['math197_pp'],r['loss296'],r['chat29_rm']
        lines.append(f"| {a} − {b} | {m['difference']:+.2f} ± {m['se']:.2f} | {m['better']}/{m['worse']}/{m['same']} | {l['difference']:+.4f} ± {l['se']:.4f} | {t['difference']:+.4f} ± {t['se']:.4f} |")
    lines += ['','原200题、原30题、总奖励和严格数字诊断的全部配对统计见 summary.json；没有挑选效果最好的一组采样。','',
        '## 3. 对照如何设置','',
        '两个新模型都从同一个 bb_dist 权重开始。A（bb_sft_replay）继续学习旧教师题；B（bb_sft_new）把对应的额外题目换成新一批教师题。原 bb_rl 仍使用既有的1500步RL结果。','',
        f"每组共同保留旧教师轨迹{audit['common_old_teacher']}条、普通对话{audit['common_general']}条；再加{audit['extra_slots']}条额外教师轨迹。A的额外槽位按1/2/3算式分层，循环打乱抽取旧题；B放入Qwen新生成并按旧规则保留的轨迹。两组均为{audit['total_rows_per_arm']}条，使用同一个排列，额外槽位的算式数量分布完全相同：{audit['extra_expression_counts']}。",'',
        f"训练沿用未修改的 train_full_sft.py：全参数监督训练，2轮，学习率2e-5，batch16，序列768，num_workers2，seed42，AdamW及原余弦学习率；每组{audit['total_steps_per_arm']}次更新。训练前只用128条跑了显存和有限loss检查，没有用最终测试集挑选参数或中间权重。两个完整训练都结束后才锁定权重并开始最终评测。",'',
        '| 训练 | 用时 | 峰值分配显存 | 峰值保留显存 |','|---|---:|---:|---:|']
    for n,r in lock['training'].items():
        lines.append(f"| {n} | {r['seconds']/60:.2f}分钟 | {r['peak_gpu_allocated_gib']:.2f}GiB | {r['peak_gpu_reserved_gib']:.2f}GiB |")
    lines += ['',
        '显存统计来自PyTorch，不包含桌面等占用。两组训练步数和每批长度相同，但答案长度不同，因此有效监督token数并不完全相同。data_audit.json提供相同seed42顺序扫描的有效token审计；它是诊断值，不是两轮多进程训练的实际累计token计数。','',
        '## 4. 教师生成和数据质量','',
        f"本地Qwen2.5-1.5B按旧 teacher_gen.py 的设置解题：batch32、bf16、贪心生成、每轮256token、最多3轮。先重做旧32题，保留的26条轨迹与原脚本及历史前缀逐字一致。新批次排除1条测试同文题后尝试{summary['teacher']['attempted']}题，按旧判分器保留{summary['teacher']['accepted']}条（{audit['teacher_acceptance']['percent']:.2f}%）；生成耗时累计{summary['teacher']['generation_seconds_sum']/60:.2f}分钟，原始成功与失败输出全部保存。",'',
        '教师筛选继续使用旧 validate_gt_in_text，避免同时更换筛选规则。另做完整数字匹配、实际计算器结果、未完成调用、错误工具和长度检查，只作为独立诊断。诊断更严格，但也不是完整语义正确性证明。','',
        '| 已按旧规则保留的教师数据 | 条数 | 通过严格诊断 | 未算出全部目标值 | 完整数字未全部匹配 | 未完成调用 | 超768token |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for label,key in [('共同旧教师','old_teacher_diagnostic'),('新增教师','new_teacher_diagnostic')]:
        r=audit[key];lines.append(f"| {label} | {r['n']} | {r['strict_correct']} | {r['no_all_calculator_values']} | {r['wrong_whole_numbers']} | {r['unfinished']} | {r['over_768_tokens']} |")
    lines += ['',
        '有1条新教师轨迹被旧数字包含判分器误收：第1879条（从0开始）把45写成了−45，旧规则仍命中“45”这个子串。已对该例人工检查并记录teacher_quality_review.json；为保持预先确定的筛选条件，没有根据这个结果临时换筛选器。因而2492条应称为“按旧规则保留”，不能称为全部语义正确。','',
        '超过768token的训练轨迹会被原SFT数据处理截断；这属于沿用的训练配方，不声称每条长轨迹的末尾答案都被学习。数据审计检查了每行截断后仍有监督token。','',
        '## 5. 本轮发现的旧测试污染','',
        '旧数据只按整行去重，会漏掉用户题目相同但工具说明不同的副本。本轮按用户问题去空白、忽略大小写检查，并保留算术符号，防止把1+2与1−2混为一题。发现的重复：','',
        '| 数学题索引（从0开始） | 题目 | 已知训练位置 |','|---:|---|---|']
    for r in plan['known_test_question_overlaps']:
        lines.append(f"| {r['eval_index']} | {r['question']} | {', '.join(r['seen_in'])} |")
    lines += ['',
        '因此主要数学比较统一剔除15、125、181，保留197题。聊天第27题“Good morning翻成西班牙语”也与旧bb_mix普通对话同文，主要聊天比较统一剔除，保留29题；检查范围内没有发现语言测试同文重复。剔除清单在两个新模型训练与最终评测前确定，原始200/30题输出没有删除或改写。旧模型已经看过这些题，本轮不能撤回其历史训练，只能在主要比较中排除。','',
        '两批原数学训练文件的用户问题有25个重合；旧2999行对应2983个不同问题，新RL3000行对应2985个不同问题。因此历史“全新3000题”的说法不够准确。本次新蒸馏尽量复现原RL题池，保留其内部重复和与旧训练问题的重合，同时排除测试同文题。新训练集与数学200题、聊天30题、语言300题的同文问题交集均为0。','',
        '这只排除了已查明的同文问题，不保证不存在语义改写，也不能证明预训练及原始大规模SFT从未包含类似题目。','',
        '## 6. 数学错误与判分敏感性','',
        '| 模型 | 1个算式 | 2个算式 | 3个算式 | 完整数字匹配率 | 严格工具轨迹通过率 | 旧判分命中但完整数字未命中次数 /394 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for n,r in summary['models'].items():
        q=r['math197_by_expression_count'];a=r['math197_diagnostics']
        lines.append(f"| {n} | {q[1]:.2f}% | {q[2]:.2f}% | {q[3]:.2f}% | {a['whole_number_accuracy_percent']:.2f}% | {a['strict_tool_trace_percent']:.2f}% | {a['legacy_only_hit_runs']} |")
    lines += ['',
        '上述诊断使用相同197题，不改动主要历史判分定义。严格工具轨迹要求最终完整数字、计算器实际产出全部目标值、无错误工具、无无效表达式、调用已结束。它会把直接心算等答案也判为未通过，因此只是工具行为检查。更多调用次数、表达式、结果复述指标见summary.json。完整数字检查仍可能在题干或中间值里命中目标数，不等同完整语义判分。','',
        '本次助手逐条检查了全部38个主要集合中的判分分歧输出：31次错误数字、5次漏答、2次只重复等价算式4180/10而未明确写出418。逐条记录见grading_disagreement_review.json；这不是独立盲评，也没有据此继续训练或调参。','',
        '## 7. 能回答和不能回答的问题','',
        'A−bb_dist反映本配方下继续复习和额外更新的变化；A的数学样本占比也比最初bb_mix更高，因此它不是仅增加步数的纯对照。B−A是在相同训练条数、步数与算式数量组成下，替换成新一批教师轨迹的变化。B与bb_rl的比较回答“这个继续蒸馏配方能否达到既有RL模型的水平”。','',
        '不能把原来13.25个百分点的提升精确分成“数据占多少、RL占多少”：RL和SFT的反馈、损失、轨迹筛选、有效token和计算量不同；RL学习全部原3000题，而SFT只学习教师被判为正确的轨迹，两者并非完全相同干预。本轮也只有一对训练随机种子，不能据此宣称所有RL算法或所有蒸馏配方的普遍优劣。','',
        '## 8. 复现与文件','',
        '- 两个权重：out/bb_sft_replay_768.pth、out/bb_sft_new_768.pth。原权重保留，网页模型列表未因本对照自动更换。',
        '- plan_initial.json、plan.json、protocol_amendment_chat.json：原计划和训练前发现聊天同文问题后的补充。',
        '- teacher_raw.json、teacher_new_sft.jsonl、teacher_replay.json：教师全部尝试、保留轨迹、旧流程复现。',
        '- matched_layout.json、replay_mix.jsonl、new_mix.jsonl、data_audit.json：成对数据布局、冻结训练集和质量审计。',
        '- train_*.json、evaluation_lock.json：训练配置、用时、显存，以及最终评测开始前的全部权重与协议摘要。',
        '- baseline_replay.json：bb_dist和bb_rl各6次数学、3次聊天输出及3个奖励评分复现验证；完整旧数学和聊天结果据此复用。四个模型的语言loss均重新计算。',
        '- math_final.json、chat_final.json、loss_final.json：逐题原始结果；math_diagnostics.json、summary.json：诊断与所有配对统计。',
        '- 脚本：distill_control.py（准备/教师/构造）、train_distill_control.py（原SFT训练器封装）、eval_distill_control.py（锁定/复现/评测）、write_distill_control_report.py（统计）。','',
        '最终数学沿用原200×2、seed=10000×题号+采样号、温度0.8、top_p0.85、top_k50、每轮384token、最多3轮、关闭思考、fp32权重+bf16混合精度；聊天原30×3、逐题seed1000+题号、温度0.8、512token、开启思考、fp16。语言继续排除4条答案在768token内完全消失的样本，使用296条真实回答token的逐回答平均loss。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps({'models':{n:{k:r[k] for k in ['math197_percent','math200_percent','loss296','chat29_rm','chat29_total']} for n,r in summary['models'].items()},
        'math197_comparisons':{k:r['math197_pp'] for k,r in summary['comparisons'].items()}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
