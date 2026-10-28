"""Readable result report and project handoff, preserving failed checkpoints and old models."""
import math
import statistics as st
import tool_choice_v2 as e

def interval(x):return f"{x['difference']:+.2f} ± {x['se']:.2f}；95%区间 [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]"

def main():
    s=e.read(e.OUT/'summary.json');a,b=s['candidate'],s['base'];aa,bb=s['models'][a],s['models'][b]
    export=e.read(e.OUT/'demo_export.json');published=export['published'];test=e.read(e.OUT/'tests.json');assert test['passed']
    confirmation=e.read(e.OUT/'confirmation_summary.json') if (e.OUT/'confirmation_summary.json').exists() else None
    lock=e.read(e.OUT/'final_lock.json')
    for name,h in lock['weights'].items():assert e.sha(e.P/f'out/{name}_768.pth')==h
    for name,h in lock['data_sha256'].items():assert e.sha(e.OUT/name)==h,name
    for name,h in lock['source_sha256'].items():assert e.sha(e.P/name)==h,name
    decision=('首次参数验收未通过，追加的固定96道新题独立确认通过后，作为 MiniMind-YuyangLU-v3 加入网页。' if published and confirmation else
              '通过本轮预设验收，并作为 MiniMind-YuyangLU-v3 加入网页。' if published else
              '未加入网页；研究权重和全部结果保留，原因见验收表和demo_export.json。')
    train=e.read(e.OUT/f'train_{a}.json');selection=e.read(e.OUT/'selection.json')
    diagnostic=e.read(e.OUT/'wording_diagnostic.json') if (e.OUT/'wording_diagnostic.json').exists() else None
    text=f'''# 待办5：更稳地选择计算器

2026-10-26。{decision}

**结论：普通问题乱调用明显减少，首次数学最终答案也改善；但计算器参数正确性的提升没有在固定96道新题确认中重现，本轮不作为“工具选择更稳”的新版本发布。**

**本轮只研究“数学问题选择计算器并传对全部算式、普通问题不乱调用”。天气、汇率、翻译等作为干扰工具；没有据此宣称其他工具已能真实联网工作。所有计算和训练在本机完成，没有付费调用或新增模型下载。**

## 正式结果

| 指标 | 原新身份版 v2 | 本轮 {a} |
|---|---:|---:|
| 数学首轮只选正确计算器 | {bb['math_name']:.2f}% | {aa['math_name']:.2f}% |
| 数学首轮计算器及全部算式参数均正确 | {bb['math_arguments']:.2f}% | {aa['math_arguments']:.2f}% |
| 数学最终回答严格一致性通过率 | {bb['math_strict']:.2f}% | {aa['math_strict']:.2f}% |
| 普通问题非空且不调用工具 | {bb['no_tool']:.2f}% | {aa['no_tool']:.2f}% |
| 普通问题出现工具调用标记 | {bb['false_calls_percent']:.2f}% | {aa['false_calls_percent']:.2f}% |
| 两类任务平衡的工具决策分 | {bb['balanced_decision']:.2f}% | {aa['balanced_decision']:.2f}% |

“选对工具名称”不能替代“参数完整正确”；直接算出数学答案也可能通过最终答案指标，但不算完成预期的计算器调用。普通问题不调用工具并不证明其内容正确，因此另设语言与身份保留检查。

仍有具体失败：要求“复述会议室308，不用计算”的英文问题，候选在2个条件下仍调用计算器；“核实这些算式”的中文问法常被误解成要求更多信息。报告中的改善是相对结果，未解决所有带数字的非计算请求或自然问法。

## 配对差异

单位为百分点，差值均为本轮−原版。

| 指标 | 差值±1SE及95%区间 | 好/差/相同的基础问题数 |
|---|---|---|
'''
    for key in ['math_name','math_arguments','math_strict','no_tool','balanced_decision']:
        p=s['pairs'][key];text+=f"| {key} | {interval(p)} | {p['better']}/{p['worse']}/{p['same']} |\n"
    text+='''
数学24道、普通24道基础问题，各自展开8种条件。统计先将同一道基础问题的8次结果平均，再按基础问题配对；没有把384种条件当作384道互不相关的题。bootstrap10000次、种子20261026，±为1SE，95%区间未经多重比较校正，不含训练种子和裁决误差。

## 工具数量和说明措辞

以下每格为数学参数正确率 / 数学最终严格通过率 / 普通题非空不调用率。

| 工具数 | 说明 | 原版 | 本轮 |
|---|---|---|---|
'''
    for n in [1,2,4,6]:
        for v in [0,5]:
            def triple(name):
                r=s['subgroups'][name][f'n{n}_v{v}'];return ' / '.join(f'{r[k]:.1f}%' for k in ['math_arguments','math_strict','no_tool'])
            text+=f"| {n} | {'原说明' if v==0 else '未用于训练的改写说明'} | {triple(b)} | {triple(a)} |\n"
    text+=f'''
训练覆盖1、2、4、6种工具数；本轮没有单独评估3或5种。每题独立打乱工具顺序和干扰项，计算器始终可用。保持工具名称、参数键和类型不变，只改变说明文字和排列；不能外推到工具改名或新参数协议。

## 保留原能力与验收

| 检查 | 原版 | 本轮 |
|---|---:|---:|
| 历史数学197题，严格通过率 | {bb['historical_math197']:.2f}% | {aa['historical_math197']:.2f}% |
| 留出回答NLL296，越低越好 | {bb['answer_nll296']:.5f} | {aa['answer_nll296']:.5f} |
| 无工具身份名字命中 /14次 | {bb['identity_name_hits14']} | {aa['identity_name_hits14']} |
| 有工具、明确询问名字或开发者的命中 /{aa['identity_with_tools_total']}次 | {bb['identity_with_tools_name_hits']} | {aa['identity_with_tools_name_hits']} |

历史数学变化：{interval(s['pairs']['historical_math197'])} 个百分点。
回答NLL变化：{interval(s['pairs']['answer_nll296'])}。这衡量已给定参考回答的预测分布，不等同于聊天质量；名字命中也不等同于完整身份回答正确。

候选14次身份回答的内容检查：12次常规身份题清楚给出模型与定制开发者；2次Jingyao Gong关系题只复述身份，没有交代原项目与定制关系，因此不能写成身份内容正确14/14。见identity_content_review.json。

固定抽查8组“6工具、改写说明”的普通题（事先选定中英文身份、知识、写作、信息提取各2题）：助手内容审查认为原版0组完成主要请求，候选3组完成，另外5组仍有错误或跑题。这里只是描述性抽查，非独立人工盲评或总体准确率估计；原文和理由见content_sample.json、content_sample_review.json。接近99%的“不调用工具”不能解释为接近99%的聊天正确率。

| 事前设定的正式验收项 | 通过 |
|---|---|
'''
    labels={'dev_selected_gates_passed':'开发集语言损失增加≤0.05、普通题下降≤5个百分点',
        'exact_math_arguments_gain_ci_above0':'数学参数正确率提升的95%配对区间下界>0',
        'no_tool_drop_le5pp':'普通题不乱调用下降≤5个百分点',
        'historical_strict_math_drop_le5pp':'历史严格数学下降≤5个百分点',
        'answer_nll_increase_le005':'正式回答NLL增加≤0.05','identity_hit_drop_le1':'身份名字命中减少≤1/14'}
    for key,val in s['gates'].items():text+=f"| {labels[key]} | {'是' if val else '否'} |\n"
    text+=f'''
这些阈值是本项目的实用接受线，并非统计等效性证明。通过也不能称为所有能力不变。
'''
    if confirmation:
        cf=confirmation
        text+=f'''
## 首次未通过后的独立新题确认

**首次验收结论仍保留为未通过**：数学参数提升{interval(s['pairs']['math_arguments'])}个百分点，区间包含0。没有因为另外几个指标改善而将此项改成通过。

看到首次结论后，另行固定96道新数学问题×8条件（每模型768次），仅测试同一个冻结版本，未重训或换权重，不按中途结果决定停止。使用原先两种留出问法族、原说明与改写说明，生成新的算式数值；一部分工具配置复用，其余重新排列。这是同一任务分布的独立问题确认，不能宣称任意措辞泛化。

确认结果：原版参数正确率{cf['means'][b]:.2f}%，候选{cf['means'][a]:.2f}%；差值{cf['difference_pp']:+.2f} ± {cf['se']:.2f}个百分点，按96道基础问题配对的**97.5%** bootstrap区间[{cf['bootstrap975'][0]:+.2f}, {cf['bootstrap975'][1]:+.2f}]，好/差/相同为{cf['better']}/{cf['worse']}/{cf['same']}。该独立确认{'通过' if cf['passed'] else '未通过'}预先为它设定的标准。

额外确认采用更保守的97.5%区间，并同时公开最初不确定结果；不把两批数据拼成一次事前测试。新题在确认前生成并固定，权重摘要沿用初次锁定记录。确认只生成首轮回答，8种条件作为一个批次，使用相同批次种子；左侧填充的开发输入通过24词元确定性生成一致检查。初次测量逐条生成，因此两批百分比不应当成同一组采样直接相减。确认仍只涉及一个模型训练种子，区间为近似重采样区间，不能作形式化错误率保证。

最终是否发布由release_decision.json记录：保留原来的全部能力保留门槛，初次参数指标不确定时，另需本次固定规模独立确认通过。
'''
    if diagnostic:
        text+='''
## 失败问法的事后诊断

参数正确率按两种固定问法族拆开如下，仅作事后描述，不用于更换模型或挑选发布条件：

| 数据批次 | 问法族 | 基础题数 | 原版 | 候选 |
|---|---|---:|---:|---:|
'''
        for cohort,d in diagnostic.items():
            for family,r in d.items():
                text+=f"| {cohort} | {'中文核实问法' if family=='zh' else '英文I need the value问法'} | {r['base_questions']} | {r[b]:.2f}% | {r[a]:.2f}% |\n"
        text+='''
同一补丁在两个问法族上出现相反取舍，换一批问题后总体均分也会变。这里语言与固定措辞同时变化，不能把差异解释为所有中文/英文能力的因果结论。失败原文中，候选常把中文“核实”当成还需要提供信息或校验已有答案，没有调用计算器。

后续应另开训练实验，增加计算意图的中文改写、数字复述等无需计算的反例，并按多种问法族和算式数量分层准备全新验收题。本轮两批测试不能在看过结果后继续被称为新盲测集，也不继续训练以追求本轮显著性。
'''
    text+=f'''

## 数据、训练及版本选择

- 从已合并的当前网页新身份版 `{e.BASE}` 开始，逐参数核对其HF导出与原权重一致；新LoRA只合并回这个基座，原权重、旧补丁和三个轮次版本均保留。
- 训练共2114行：960条程序生成并精确求值的计算器完整轨迹、554条带工具列表但直接作答的普通/身份监督、400条普通对话和200条旧数学的冻结原版分布约束。普通参考回答取自已有训练池，未重新用大模型生成，不保证其所有事实正确。
- 新数学训练/开发/测试按完整问题及表达式结构去重，测试采用不同提问模板；已扫描本地已知旧数学训练池并避开其表达式。普通题与本轮训练及历史留出题按问题文字排除重复。无法证明上游语义改写从未出现。
- 描述0是原文，1–3用于训练，4只用于开发，5只用于正式测试。每个数学题含1–3个显式算式；未研究复杂应用题的语义转算式。
- 2114条完整文本最长1418词元，全部放入1536长度，没有静默截断；所有助手片段及结束符有标签，用户与工具返回不作为监督标签。
- LoRA rank16，只训练q_proj/o_proj，共393216参数；batch4、2个加载进程、学习率0.0002、余弦降至10%、3轮1587更新、seed42。监督交叉熵加系数2的原版KL保留，保留回答最初32个词元额外加权。教师和基座均无梯度。
- 单个16步资源试跑后，完整三轮耗时约{e.read(e.OUT/'train_tool_choice_e3.json')['seconds']/60:.2f}分钟，峰值PyTorch分配显存{e.read(e.OUT/'train_tool_choice_e3.json')['peak_allocated_gib']:.2f}GiB，不含桌面等其他进程。
- 同一个训练配方保存第1/2/3轮，开发24道基础题×4条件，用平衡决策分与保留门槛选中第{train['epoch']}轮 `{a}`。第1/2/3轮开发数学参数正确率为{selection['candidates'][0]['math_arguments']:.1f}%/{selection['candidates'][1]['math_arguments']:.1f}%/{selection['candidates'][2]['math_arguments']:.1f}%，说明训练更久并非必然更好。
- 正式测试在选择后才开始，之后没有重训、换版本或根据成绩改模型挑选方法。final_lock.json锁定输入、代码和基线/候选权重摘要；独立确认作为后续实验单独保留协议与结果。

## 评测设置和判分检查

两模型使用相同HF Qwen3 fp16引擎、种子、工具列表、温度0.9、top_p0.85、top_k50、关闭思考、每轮384词元、最多3轮。每条件单次采样，不启用多候选。原始工具执行沿用网页；仅计算器有本轮数值正确性目标，干扰工具的示例返回不能视为实时服务。

主测试384条件/模型：24道新数学与24道普通题，各展开1/2/4/6工具×原说明/改写说明。另两模型均重测同样的197道历史数学、296条有效留出参考回答及7道身份题×2次。历史数学仍排除15/125/181；参考回答被768长度完全截断的4条不参与NLL。

工具参数按表达式AST结构和重复次数精确匹配，可换回答顺序，不能漏题或重复同一算式顶替；等价但结构不同的改写在这个参数指标中可能不算通过。最终回答用冻结的严格数学一致性判分，未知格式进入助手逐条裁决，不自动当错。当前共有{s['review']['unique_records']}条不同原文裁决记录，全部不确定项已处理；历史相同问题且逐字相同回答可复用裁决。不是独立人工盲评。

{test['tests']}项针对性检查通过，覆盖漏调用、重复调用、参数数值冒充算式、工具名错误、非字符串参数、空回答和不完整标记，以及工具schema和数据隔离。本轮只训练一个随机种子、使用少量基础问题和固定改写说明；不能据此声称任意措辞或所有工具均已稳健。

## 网页与文件

{decision}
'''
    if published:text+='''刷新现有网页即可在模型列表看到 `MiniMind-YuyangLU-v3`；也可双击根目录start_demo.bat启动。工具面板现在最多勾选6个；旧版模型和多候选开关保留。默认模型排序会将v3放在v2之前。模型导出核对了全部参数、词表、模板与缓存生成，另记录网页系统提示下的实际小规模试用。
'''
    text+='''
数据与协议：plan.json、train.jsonl、dev_tasks.json、test_tasks.json、guardrail_tasks.json、token_audit.json、final_lock.json。
训练与开发：train_tool_choice_e*.json、dev_*.json、loss_dev_*.json、selection.json。
正式原文：final_*.json、guard_*.json、identity_*.json、loss_final_*.json；裁决：math_reviews.json；统计：summary.json、final_outcomes.json。
脚本：tool_choice_v2.py、train_tool_choice_v2.py、eval_tool_choice_v2.py、run_tool_choice_v2.py、review_tool_choice_v2.py、report_tool_choice_v2.py、export_tool_choice_v2.py、test_tool_choice_v2.py。
'''
    (e.OUT/'REPORT.md').write_text(text,encoding='utf-8',newline='\n')
    short=f'''2026-10-26完成第5项的本地训练与对照，{decision}

从当前网页新身份版{e.BASE}训练LoRA，按开发集选择{a}。正式新题为24道数学+24道普通题，各展开8种工具数量/说明条件；两模型相同设置单次采样、不启用多候选。数学“选对计算器且全部算式参数正确”由{bb['math_arguments']:.2f}%变为{aa['math_arguments']:.2f}%，差异{interval(s['pairs']['math_arguments'])}个百分点；普通题非空且不调用由{bb['no_tool']:.2f}%变为{aa['no_tool']:.2f}%。数学最终严格通过率{bb['math_strict']:.2f}%→{aa['math_strict']:.2f}%。

保留检查：历史严格数学{bb['historical_math197']:.2f}%→{aa['historical_math197']:.2f}%；回答NLL296为{bb['answer_nll296']:.5f}→{aa['answer_nll296']:.5f}；身份名字命中{bb['identity_name_hits14']}/14→{aa['identity_name_hits14']}/14。完整门槛、区间和全部逐条原文见[工具选择报告](results/tool_choice_20261026/REPORT.md)。

范围仅限计算器与不调用的决策，其他工具作为干扰项，未实现或验证实时天气/汇率等服务。训练覆盖1/2/4/6工具数、不同说明和顺序，工具名及参数schema未改；仅一个训练种子。问题、表达式与本轮训练/开发/测试及已知旧数学池做了去重，未证明上游语义绝对无重叠。

本地完整三轮约3.5分钟、峰值2.76GiB分配显存，无付费调用；原权重和全部尝试保留，新LoRA只回到同一个v2基座。旧版评测脚本、严格数学判分与多候选挑选核心未修改。
'''
    if confirmation:
        cf=confirmation
        short+=f"\n首次参数指标的区间包含0，因此首次验收保持未通过。随后另固定96道新数学题×8条件，只测同一冻结版本，未重训或换模型；原版{cf['means'][b]:.2f}%、候选{cf['means'][a]:.2f}%，差异{cf['difference_pp']:+.2f} ± {cf['se']:.2f}个百分点，97.5%区间[{cf['bootstrap975'][0]:+.2f}, {cf['bootstrap975'][1]:+.2f}]，独立确认{'通过' if cf['passed'] else '未通过'}。两批数据单独报告，不合并伪装成事前一次测试。\n"
    headings={'README.md':'## 14. 工具选择稳健性训练（2026-10-26）',
              'SUMMARY_FOR_CHATGPT.md':'## 15. 工具选择稳健性训练（2026-10-26）',
              'HANDOFF.md':'## 第八部分：2026-10-26 工具选择稳健性训练'}
    for file,heading in headings.items():
        path=e.P/'my_eval'/file;old=path.read_text(encoding='utf-8');backup=e.OUT/(file.removesuffix('.md')+'.before_task5.md')
        if not backup.exists():backup.write_text(old,encoding='utf-8')
        if heading in old:old=old[:old.index(heading)].rstrip()
        note='最新进展（2026-10-26）：第5项首轮训练及两批复测已完成，尚未获得可发布的稳健工具选择版本，见[报告](results/tool_choice_20261026/REPORT.md)。\n\n'
        if published:note='最新进展（2026-10-26）：第5项训练与独立确认已完成，新模型与适用范围见[报告](results/tool_choice_20261026/REPORT.md)。\n\n'
        old=old.replace('最新进展（2026-10-26）：第5项工具选择训练与正式复测已完成，结果及是否加入网页见[报告](results/tool_choice_20261026/REPORT.md)。\n\n','')
        if note not in old:
            first,rest=old.split('\n',1);old=first+'\n\n'+note+rest.lstrip('\n')
        if published:
            old=old.replace('5. 让模型学会自己选工具：训练时随机给 1–6 个工具，改写工具描述，用各种问法，并混入不需要工具的问题；用 my_eval\\scripts\\tool_choice_probe.py 评估。',
                '5. 【已完成本轮，2026-10-26】训练计算器/不调用的稳健选择，覆盖1/2/4/6工具数、说明改写和新问法；候选通过验收并加入v3，见第八部分。')
        path.write_text(old.rstrip()+'\n\n'+heading+'\n\n'+short,encoding='utf-8',newline='\n')
    # Independent arithmetic recheck of means and SEs from final outcome rows.
    outcomes=e.read(e.OUT/'final_outcomes.json')
    for name in [a,b]:
        for kind,field,metric in [('math','arguments_ok','math_arguments'),('math','strict_ok','math_strict'),('no_tool','route_ok','no_tool')]:
            vals=[r[field] for r in outcomes[name] if r['kind']==kind]
            assert abs(100*sum(vals)/len(vals)-s['models'][name][metric])<1e-9
    e.save('validation.json',dict(passed=True,all_final_hashes_verified=True,tests=test,means_crosschecked=True,
        initial_accepted=s['accepted'],confirmation_accepted=bool(confirmation and confirmation['passed']),
        effective_accepted=e.read(e.OUT/'release_decision.json')['effective_accepted'],published=published,report_sha256=e.sha(e.OUT/'REPORT.md')))
    print('Published report and updated three project documents.',flush=True)

if __name__=='__main__':main()
