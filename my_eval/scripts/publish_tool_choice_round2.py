"""Round2 report: keep the sealed failed round1 visible, and label fresh evidence."""
import tool_choice_round2 as e
def interval(p, precision=2):return f"{p['difference']:+.{precision}f} ± {p['se']:.{precision}f}；95%区间[{p['ci95'][0]:+.{precision}f}, {p['ci95'][1]:+.{precision}f}]"

def main():
    s=e.read(e.OUT/'summary.json');a,b=s['candidate'],s['base'];aa,bb=s['models'][a],s['models'][b]
    export=e.read(e.OUT/'demo_export.json');published=export['published'];lock=e.read(e.OUT/'final_lock.json')
    for name,h in lock['weights'].items():assert e.sha(e.P/f'out/{name}_768.pth')==h
    for name,h in lock['data_sha256'].items():assert e.sha(e.OUT/name)==h,name
    for name,h in lock['source_sha256'].items():assert e.sha(e.P/name)==h,name
    tests=e.read(e.OUT/'tests.json');assert tests['passed']
    independent=e.read(e.OUT/'validation_independent.json');assert independent['passed']
    content=e.read(e.OUT/'content_sample_review.json')
    training=e.read(e.OUT/f'train_{a}.json')
    ui=e.read(e.OUT/'demo_ui_validation.json') if published else None
    smoke=e.read(e.OUT/'demo_smoke.json') if published else None
    if published:
        assert s['accepted'] and ui['passed'] and smoke['passed']
        destination=e.P/'scripts/MiniMind-YuyangLU-v3'
        for name,h in export['files_sha256'].items():assert e.sha(destination/name)==h,name
    outcome='通过本轮事前验收，并作为MiniMind-YuyangLU-v3加入网页。' if published else '本轮未形成可发布版本，未加入网页；全部权重与原文保留。'
    text=f'''# 工具选择第二轮：计算意图与数字反例

2026-10-26。{outcome}

本轮只研究计算器选择及普通问题不调用，其他工具是干扰项；没有实现或验证实时天气、汇率、翻译等服务。

## 新题正式结果

| 指标 | 原v2 | 候选{a} |
|---|---:|---:|
| 数学首轮只调用正确计算器 | {bb['math_name']:.2f}% | {aa['math_name']:.2f}% |
| 数学首轮计算器及全部算式参数正确 | {bb['math_arguments']:.2f}% | {aa['math_arguments']:.2f}% |
| 普通题非空且不调用工具 | {bb['no_tool']:.2f}% | {aa['no_tool']:.2f}% |
| 普通题出现调用标记 | {bb['false_calls']:.2f}% | {aa['false_calls']:.2f}% |
| 两类任务各占一半的决策分 | {bb['balanced_decision']:.2f}% | {aa['balanced_decision']:.2f}% |
| 6工具+新说明，48题完整执行后严格数学通过率 | {bb['fullmath48']:.2f}% | {aa['fullmath48']:.2f}% |

数学48道基础题、普通24道基础题，每题展开1–6工具×原说明/新说明共12条件，每模型864次首轮回答。基础题内先平均，再配对比较：

| 指标 | 差值±1SE与95%区间（百分点） | 好/差/相同基础题 |
|---|---|---|
'''
    for key in ['math_name','math_arguments','no_tool','fullmath48','historical_math197']:
        p=s['pairs'][key];text+=f"| {key} | {interval(p)} | {p['better']}/{p['worse']}/{p['same']} |\n"
    text+='''
数学参数按每道基础题的12条件聚类，n=48；普通题n=24。完整数学只另测每题6工具+新说明一个条件，n=48，不能把它与首轮参数百分比混成同一个指标。bootstrap10000次、种子20261027；±为1SE，区间未经多重比较校正，是单训练种子的近似重采样统计。

## 数量与措辞分组

每格为数学完整参数正确率 / 普通题非空不调用率：

| 工具数 | 说明 | 原v2 | 候选 |
|---|---|---|---|
'''
    for n in range(1,7):
        for v in [0,6]:
            def pair(name):
                r=s['subgroups'][name][f'n{n}_v{v}'];return f"{r['math_arguments']:.1f}% / {r['no_tool']:.1f}%"
            text+=f"| {n} | {'原说明' if v==0 else '新改写说明'} | {pair(b)} | {pair(a)} |\n"
    text+=f'''
这是同一固定工具名和参数schema下的改写。没有测新工具名、全新schema、缺少计算器时的处理或任意应用题。

## 能力保留与验收

| 项目 | 原v2 | 候选 |
|---|---:|---:|
| 历史严格数学197题 | {bb['historical_math197']:.2f}% | {aa['historical_math197']:.2f}% |
| 留出回答预测损失296条，低好 | {bb['answer_nll296']:.5f} | {aa['answer_nll296']:.5f} |
| 无工具身份名字命中/14 | {bb['identity_name_hits14']} | {aa['identity_name_hits14']} |

损失差异：{interval(s['pairs']['answer_nll296'], 5)}（损失单位）。预测损失和名字命中均为代理指标，不等于完整聊天或身份内容正确。无工具的14次身份回答均出现MiniMind和Yuyang LU，但涉及原作者jingyaogong的2次回答没有解释原作与本地定制的关系，不能称为14/14身份内容全对。

预设验收：新题数学参数提升区间下界>0；普通题不调用≥90%且下降≤5个百分点；新题完整数学和旧数学各下降≤5个百分点；预测损失增加≤0.05；身份命中少≤1/14。开发时另要求普通题≥85%、损失增加≤0.05，按两类任务较低的一项优先选模型。

| 验收项 | 通过 |
|---|---|
'''
    for k,v in s['gates'].items():text+=f"| {k} | {'是' if v else '否'} |\n"
    text+=f'''
阈值是项目接受线，不是统计等效性证明。基线历史197题、NLL296和身份14次复用第一轮同权重、同题、同种子、同生成设置结果，逐文件摘要见baseline_reuse.json；新候选重新评测。新题首轮和完整数学的两模型均重新生成。

## 为什么另做第二轮

[第一轮及独立确认](../tool_choice_20261026/REPORT.md)未能证明参数选择更稳，其负面结论、180条裁决和全部原文保留。第一轮看过的题在这里仅是开发/诊断材料，不再充当新盲测证据。

第二轮增加多种中文/英文计算问法，加入编号复述、原样抄写算式而不计算的反例；训练覆盖1–6工具，工具说明和顺序变化。正式数学按中英文各4个留出问法×1/2/3算式×2组数值分层，共48题；普通24题包含身份、知识、写作、信息提取和不求值的文本复制。所有新验收题在训练选择前固定，数学表达式与已知本地旧训练/测试池排除相同结构；不保证上游语义从未出现。

训练准备中发现按序分配工具数会与算式数量绑定：该批已跑权重保留在out/tool_choice2_e*.pth，数据和日志存development_v1_count_confound，未用于开发选择或正式评测。正式使用修正后的独立打乱/分层均衡数据，每种算式数量下1–6工具各80条，18个格子都覆盖，见count_balance_audit.json。开发/正式题没有因此重抽。

## 正式训练与版本选择

从同一个当前v2基座{e.BASE}开始，原始参数逐项核对；仅训练q_proj/o_proj的rank16 LoRA，393216参数。修正数据2840行：1440数学完整轨迹、800普通/身份/数字反例监督、400普通对话和200旧数学的冻结原版KL保留。最长1418词元，1536长度全保留，无静默截断。

学习率0.0001、batch4、3轮、seed42、2个加载进程；数学监督最初64个回答词元权重3，其余1；KL系数4，保留样本首32个回答词元权重3。教师和基座无梯度。与第一轮相比同时改变了数据、加权、学习率和KL，不能把效果单独归因于某一项。有效训练脚本train_tool_choice_round2b.py，三个轮次权重tool_choice2b_e1/e2/e3均保留，按开发集选择{a}。

有效三轮训练记录为{training['seconds'] / 60:.2f}分钟、2130次更新，PyTorch峰值分配显存{training['peak_allocated_gib']:.2f} GiB；不含先前准备、未采用训练和评测耗时，也不是全系统显存占用。

只在开发集选版本，final_lock.json之后未训练或换模型。所有试验本地运行，未付费、未下载新模型；原v2和第一轮所有权重均保留，补丁只回到其实际v2基座。

## 评测与局限

首轮路由批量按同一道题的12条件生成，HF fp16、温度0.9、top_p0.85、top_k50、最多384词元、关闭思考，两模型相同批次种子；开发输入的左侧填充与单条确定性生成检查通过。完整数学、历史数学和身份用同一个逐条工具执行流程，最多3轮。标准答案不提供给生成模型。

参数按完整表达式AST及出现次数匹配，允许调用顺序变化；仅工具名对、漏调用、重复调用顶替均不能通过。等价但结构不同的改写可能被这个严格参数指标排除。最终数学沿用冻结一致性判分，未知格式由助手复核；当前{s['review_records']}条不同原文记录，旧相同题目和逐字相同回答可复用。不是独立人工盲评。

普通题非空不调用不等于回答内容正确。在事先固定的8道内容抽查中，6工具+新说明下，原v2有{content['baseline_usable']}/{content['n']}满足主要请求，候选有{content['candidate_usable']}/{content['n']}；候选只有数字复述和原样抄写两条合格，其余6条仍有跑题或错误，包含两条身份问题。原文和逐条判断见content_sample.json、content_sample_review.json。这是助手描述性复核，非独立人工盲评，不能外推总体事实准确率。大量工具说明下的普通知识、写作和身份内容仍薄弱。

首轮判断的全部1728条原文已用独立解析重新核对，配对均值和标准误一致，见validation_independent.json。只有一个训练种子，少量问法族；不能据此声称所有自然表达和所有工具都可靠。

## 使用与文件

{outcome}
'''
    if published:text+='刷新网页选择MiniMind-YuyangLU-v3，或双击start_demo.bat启动；工具面板最多6个。单次回答执行前新增所选工具白名单与参数类型检查，未勾选的工具不会执行。这是执行约束，与模型选择能力的统计结果分开。旧模型和多候选功能保留，多候选核心和原工具schema/执行器未修改；本轮主要结果是单次回答，不依赖多候选，也未验证v3开启多候选后的效果。\n\n导出权重所有参数张量逐项相等，词表和聊天模板相等，3个输入的前24个贪心词元相等；真实权重完成身份/双算式/普通问题3次流程检查。普通问题的流程检查只核对非空且不调用，不代表写作内容合格。另以模拟模型验证网页6项可选、第7项禁用、取消后恢复，以及未选工具不执行；见demo_export.json、demo_smoke.json、demo_ui_validation.json。\n'
    text+='数据协议plan.json、dev_tasks.json、test_tasks.json、train.jsonl；训练train_tool_choice2b_*.json；开发selection.json；正式final_*.json、fullmath_*.json、guard_*.json；裁决math_reviews.json；统计summary.json；导出demo_export.json。\n'
    (e.OUT/'REPORT.md').write_text(text,encoding='utf-8',newline='\n')
    brief=f'''2026-10-26，第5项第二轮独立实验：{outcome}

第一轮失败记录仍见[首轮报告](results/tool_choice_20261026/REPORT.md)。第二轮增加计算意图改写、数字非计算反例，修正工具数量/算式数量关联，正式使用2840行数据。从同一v2基座训练LoRA，开发选择{a}，新验收题与原来已看过的题分开。

新数学48道×12条件，完整工具参数{bb['math_arguments']:.2f}%→{aa['math_arguments']:.2f}%，差异{interval(s['pairs']['math_arguments'])}个百分点。普通24道×12条件，非空不调用{bb['no_tool']:.2f}%→{aa['no_tool']:.2f}%。6工具新说明下完整执行48题严格通过{bb['fullmath48']:.2f}%→{aa['fullmath48']:.2f}%。历史严格数学{bb['historical_math197']:.2f}%→{aa['historical_math197']:.2f}%；回答损失{bb['answer_nll296']:.5f}→{aa['answer_nll296']:.5f}；身份名字命中{bb['identity_name_hits14']}/14→{aa['identity_name_hits14']}/14。

范围是计算器/不调用选择；其他工具仍为干扰项。预测损失、名字命中和不调用率均不等于完整内容正确；6工具新说明下的固定8题内容抽查仅{content['baseline_usable']}/8→{content['candidate_usable']}/8满足请求，普通知识、写作和带工具身份内容仍薄弱。网页最多6个工具并检查所选白名单；v3多候选效果未经本轮验证。详见[第二轮完整报告](results/tool_choice_round2_20261026/REPORT.md)。\n'''
    headings={'README.md':'## 15. 工具选择第二轮（2026-10-26）','SUMMARY_FOR_CHATGPT.md':'## 16. 工具选择第二轮（2026-10-26）','HANDOFF.md':'## 第九部分：2026-10-26 工具选择第二轮'}
    for file,heading in headings.items():
        path=e.P/'my_eval'/file;old=path.read_text(encoding='utf-8');backup=e.OUT/(file.removesuffix('.md')+'.before_round2.md')
        if not backup.exists():backup.write_text(old,encoding='utf-8')
        if heading in old:old=old[:old.index(heading)].rstrip()
        old=old.replace('最新进展（2026-10-26）：第5项首轮训练及两批复测已完成', '首轮记录（2026-10-26）：第5项首轮训练及两批复测已完成')
        note=f"最新进展（2026-10-26）：工具选择第二轮{'通过验收并加入v3' if published else '仍未通过发布验收'}，首轮失败记录保留，见[第二轮报告](results/tool_choice_round2_20261026/REPORT.md)。\n\n"
        if note not in old:
            first,rest=old.split('\n',1);old=first+'\n\n'+note+rest.lstrip('\n')
        if published:
            if file=='SUMMARY_FOR_CHATGPT.md':
                old=old.replace('更新：2026-10-25。本文件', '更新：2026-10-26。本文件')
                old=old.replace('2026-10-25 加入通过验收的新身份版 MiniMind-YuyangLU-v2，目前有6个模型。', '2026-10-25 加入通过验收的新身份版 MiniMind-YuyangLU-v2；2026-10-26 加入工具选择版MiniMind-YuyangLU-v3，目前有7个模型。')
            if file=='HANDOFF.md':
                old=old.replace('当前有6个模型：MiniMind-YuyangLU-v2', '当前有7个模型：MiniMind-YuyangLU-v3（tool_choice2b_e3，计算器/不调用选择版）、MiniMind-YuyangLU-v2')
                old=old.replace('这些 Hugging Face 格式的模型由 my_eval\\scripts\\make_demo_models.py 导出，并核对过 logits 和原模型一致。', '早期Hugging Face模型由my_eval\\scripts\\make_demo_models.py导出；v3由export_tool_choice_round2.py导出并逐项核对参数、模板和贪心生成，见第九部分。')
            if file=='README.md':
                old=old.replace('项目自带的代码只改了一处：`scripts/web_demo.py` 的工具定义（见 4.10）；另外新增了根目录的 `start_demo.bat` 和 `scripts/` 下的 4 个 demo 模型文件夹。', '网页入口为`scripts/web_demo.py`，配套工具、多候选及执行检查模块在`scripts/`中；根目录`start_demo.bat`启动，目前有7个demo模型文件夹。早期功能见4.10，后续改动与验证见第10–15节。')
            old=old.replace('5. 让模型学会自己选工具：训练时随机给 1–6 个工具，改写工具描述，用各种问法，并混入不需要工具的问题；用 my_eval\\scripts\\tool_choice_probe.py 评估。',
                '5. 【第二轮已完成，2026-10-26】计算器/不调用选择通过新题验收并加入MiniMind-YuyangLU-v3；首轮失败与第二轮记录均保留，见第九部分。')
        path.write_text(old.rstrip()+'\n\n'+heading+'\n\n'+brief,encoding='utf-8',newline='\n')
    e.save('validation.json',dict(passed=True,original_final_locks_verified=True,accepted=s['accepted'],published=published,tests=tests,
        independent=independent,ui=ui,smoke_passed=bool(smoke and smoke['passed']),
        report_sha256=e.sha(e.OUT/'REPORT.md')))
    print('Published round2 report and three project documents.',flush=True)

if __name__=='__main__':main()
