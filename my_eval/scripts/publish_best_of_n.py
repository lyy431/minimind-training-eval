"""Publish task-4 results without conflating reward maximization with accuracy."""
import hashlib
import json
import math
from pathlib import Path
import statistics as st
P=Path(__file__).resolve().parents[2]
OUT=P/'my_eval/results/best_of_n_20261026'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def fmt(x):return f'{x:.2f}'
def interval(p):return f"{p['difference']:+.2f} ± {p['se']:.2f}；95%区间 [{p['bootstrap95'][0]:+.2f}, {p['bootstrap95'][1]:+.2f}]"

def publish():
    s=read(OUT/'summary.json');m,c=s['math'],s['chat'];p=m['paired']['4_minus_1']
    tests=read(OUT/'tests.json');assert tests['passed']
    assert read(OUT/'real_ui_path_smoke.json')['passed']
    assert read(OUT/'automatic_audit_review.json')['all_40_agree']
    assert hashlib.sha256((OUT/'chat_review_rubric.json').read_bytes()).hexdigest()==read(OUT/'chat_review_rubric_lock.json')['sha256']
    review_lock=read(OUT/'content_review_lock.json')
    assert hashlib.sha256((OUT/'chat_reviews.json').read_bytes()).hexdigest()==review_lock['chat_sha256']
    assert hashlib.sha256((OUT/'math_reviews.json').read_bytes()).hexdigest()==review_lock['math_sha256']
    protocol=read(OUT/'protocol.json')
    for file,value in protocol['hashes'].items():
        assert hashlib.sha256((P/file).read_bytes()).hexdigest()==value,file
    selection_checks=0
    for row in read(OUT/'scored.json').values():
        for n in [1,2,4]:
            eligible=[i for i,cand in enumerate(row['candidates'][:n]) if cand['eligible']
                and cand['rm'] is not None and math.isfinite(cand['rm'])]
            expected=max(eligible,key=lambda i:row['candidates'][i]['rm']) if eligible else 0
            assert row['selected'][str(n)]==expected
            selection_checks+=1
    assert selection_checks==678
    rows=read(OUT/'final_outcomes.json')
    for label in ['math','chat']:
        for n in ['1','2','4']:
            vals=[r['scores'][n] for r in rows[label]]
            assert abs(sum(vals)/len(vals)-s[label]['means'][n])<1e-10
        for n,b in [('2','1'),('4','1'),('4','2'),('4','eligible_first')]:
            diff=[r['scores'][n]-r['scores'][b] for r in rows[label]]
            assert abs(st.stdev(diff)/len(diff)**.5-s[label]['paired'][f'{n}_minus_{b}']['se'])<1e-10
    if p['bootstrap95'][0]>0:conclusion='本轮数学测出了正向提升，但范围仅限本模型和本测试集。'
    elif p['bootstrap95'][1]<0:conclusion='本轮数学反而下降，不建议把此开关当成准确率升级。'
    else:conclusion='本轮未测出可靠的数学提升，不能把多生成几次等同于答得更准。'
    text=f'''# 待办4：多生成几个回答，再挑一个

2026-10-26。功能已加入原网页，本地免费运行，默认关闭。主要结论：**{conclusion}**

## 怎么使用

双击项目根目录的 `start_demo.bat`，展开左侧设置，勾选“多生成几个回答，再挑一个”（英文界面为 Generate several answers and select one），选择2或4个候选，再提问。开关对现有6个模型开放，本轮正式评测与真实运行检查只覆盖新身份版 `MiniMind-YuyangLU-v2`。

模式使用本地 InternLM2 奖励模型，只给它对话问题和最终回答，不给标准答案或计算器结果。显示选中序号，并可展开查看候选。它是偏好评分器，不是正确性证明器。

为适应8GB显存，候选依次生成，生成模型移回内存后才用显卡评分，再恢复生成模型。多候选模式关闭思考，每候选最多512词元/轮、3轮；超过上下文限制或未完成回答不参加评分，全部不合格则保留第一个尝试并明确提示。工具定义和工具执行保持原样；每个候选独立完成调用，只把选中轨迹写入聊天历史。

## 公平对照的设置

- 模型固定为网页导出的 `MiniMind-YuyangLU-v2`，fp16；无训练、无新模型下载、无付费调用。
- 197道历史数学留出题（仍排除15/125/181）和29道聊天题（仍排除27），每题4个独立种子的候选，共904个。不是全新盲测题库。
- 温度0.9、top_p0.85、top_k50，关闭思考，512词元/轮、最多3轮。数学使用原题附带的两工具定义；聊天用网页默认系统提示并保留题库提供的多轮历史，不等于网页默认历史轮数0。生成上下文上限4096词元，评分上下文上限2047词元，超限显式记录而不静默截断。种子为2026102600+题序×10+候选序。
- 对照1取第一个候选；2选1取前2个；4选1取前4个，按未截断的奖励原分挑选，平分取较早者。单次对照使用同样的512词元/3轮限制，**并非网页原来8192词元/最多16轮的默认配置**。
- 所有条件使用同一批候选，避免生成运气和选取效果混在一起。另报告4个候选随机选取的期望，以及“取第一个完整可评分候选”作为不使用奖励分的控制。
- 数学沿用已冻结的严格回答一致性评分；判分不确定的原文先复核再揭示奖励选择。聊天另按0/1/2的响应与可用性量表复核，不以奖励分自证提升。
- 协议、模型、数据和核心代码的摘要在正式生成前锁定；没有根据本轮成绩调模型、换挑选规则或重新抽样。

## 数学结果

| 方法 | 严格通过率（197题） |
|---|---:|
| 直接用第一个 | {m['means']['1']:.2f}% |
| 奖励模型2选1 | {m['means']['2']:.2f}% |
| 奖励模型4选1 | {m['means']['4']:.2f}% |
| 取第一个完整可评分候选，不比较奖励 | {m['means']['eligible_first']:.2f}% |
| 从4个候选随机选取的期望 | {m['random_mean']:.2f}% |
| 4个中至少一个正确（事后理想上限，无法直接部署） | {m['oracle']:.2f}% |

| 配对比较 | 差值±1SE与95%区间（百分点） | 好/差/相同题数 |
|---|---|---|
'''
    for n,b in [('2','1'),('4','1'),('4','2'),('4','eligible_first')]:
        a=m['paired'][f'{n}_minus_{b}']
        text+=f"| {n} − {b} | {interval(a)} | {a['better']}/{a['worse']}/{a['same']} |\n"
    text+=f'''
4选1相对随机选取期望：{interval(m['paired4_vs_random_mean'])} 个百分点。
有 **{m['selection_misses']} 道题**在4个候选中存在正确回答，但奖励选择未选中。理想上限用到了标准答案，仅用于诊断，不能称为实际成绩。

按算式数量拆分（描述性结果，不据此调参）：

| 算式个数 | 单次 | 2选1 | 4选1 |
|---|---:|---:|---:|
'''
    for k,v in m['by_expression_count'].items():text+=f"| {k} | {v['1']:.2f}% | {v['2']:.2f}% | {v['4']:.2f}% |\n"
    text+=f'''
## 聊天结果与评判边界

| 方法 | 内容可用性均分（0–2） | 奖励模型原分（共同可评分{c['rm_common_n']}题） | 回答平均字数 |
|---|---:|---:|---:|
'''
    for n in ['1','2','4']:text+=f"| {n} | {c['means'][n]:.3f} | {c['rm_mean_on_common_scorable'][n]:.3f} | {c['answer_chars'][n]:.1f} |\n"
    text+=f'''
内容分4选1−单次：{interval(c['paired']['4_minus_1'])}（0–2分尺度）。奖励分变化是挑选目标自身的变化，有选择偏差，不能用来证明聊天更可靠。

聊天内容量表：0=没有响应主要请求/基本不可用/明显严重错误；1=部分响应但有明显遗漏、重复或混乱；2=主要请求有清楚实用回应且无可见严重错误。由本次助手在看奖励分和选中序号前检查匿名原文，**不是独立人工盲评，也不是事实准确率评测**。未知事实不默认核实成功，缺少材料或时效信息的情况记录局限。

## 时间和资源代价

| 场景 | 单次生成均时 | 4次生成+挑选均时 | 倍数 |
|---|---:|---:|---:|
| 数学 | {m['seconds']['1']:.2f}秒 | {m['seconds']['4']:.2f}秒 | {m['seconds']['ratio4to1']:.2f}倍 |
| 聊天 | {c['seconds']['1']:.2f}秒 | {c['seconds']['4']:.2f}秒 | {c['seconds']['ratio4to1']:.2f}倍 |

上述时间为每题实测生成耗时之和加评分耗时，不含首次加载模型、网页渲染与生成模型恢复到显卡的小量时间；不是端到端延迟承诺。测试按阶段生成/评分，候选按题增量保存。904个候选生成约23.5分钟，评分约5.9分钟；评分峰值显存约3.32GiB。网页完整2候选真实路径另测约9.06秒、峰值3.18GiB，验证评分后生成模型回显卡、奖励模型回内存，且首个候选与独立生成逐字一致。详细原始记录见generation_runtime.json、score_runtime.json、real_ui_path_smoke.json。

## 核查与局限

- {tests['tests']}项功能测试通过，覆盖挑选前缀/同分/不合格候选、候选历史隔离、工具白名单、轮次上限、异常恢复、原工具定义及执行一致、网页开关和仅保存选中回答。实际本地模型另做小规模生成/评分检查。
- 数学共有{m['review_runs']}次、{m['review_unique']}条不同原文需裁决，全部有记录；自动判分器仍非任意自然语言的完备证明器。历史相同题目与逐字相同回答的裁决可复用，其他逐条记录理由。
- 另在自动通过和不通过中各固定种子抽取20条不同原文，40条均与内容检查一致，没有修改冻结判分器。聊天的112条不同原文全部结合完整对话历史检查；内容标签在查看奖励分和选择序号前锁定。
- 网页复查后补充了显卡搬运失败清理、异常/取消回滚、按完整用户轮次保留工具轨迹，以及全部候选不合格时不向后续推理历史写入未完成轨迹。这些恢复处理位于网页支持层；冻结的正常候选生成、奖励公式和挑选规则未改。
- 本轮数学9/788个、聊天10/116个候选到达词元上限；2候选时有2/226题全部不合格，4候选时有1/226题全部不合格，按协议返回首个尝试并标记。没有上下文超限。全部678个选取序号均核对了前N范围、资格、最高奖励和同分次序。
- 区间按题配对bootstrap10000次、种子20261026；±为1SE。区间未经多重比较校正，仅反映这批题和这批采样，不含训练种子、判分裁决误差。
- 本轮只测一个模型、每题一个4候选池；不外推到其他五个网页模型、长对话、开启思考或任意工具组合。
- 未改变已有模型权重、训练奖励、旧评测结果或严格判分代码；旧分数不是当前设置的直接对照。功能默认关闭，按本轮证据决定是否使用。

## 文件

网页：scripts/web_demo.py；共享实现：scripts/best_of_n.py；保持原样抽出的工具：scripts/demo_tools.py。
实验：eval_best_of_n.py；复核准备：review_best_of_n.py；统计：report_best_of_n.py；功能测试：test_best_of_n.py。
本目录保存protocol.json、tasks.json、generated.json、scored.json、math_automatic.json、math_reviews.json、chat_content_queue.json、chat_reviews.json、final_outcomes.json、summary.json、tests.json和运行耗时。
'''
    (OUT/'REPORT.md').write_text(text,encoding='utf-8',newline='\n')
    section=f'''2026-10-26，待办4完成。网页新增“多生成几个回答，再挑一个”，支持2/4个候选，本地奖励模型选择，默认关闭。仅正式评测MiniMind-YuyangLU-v2；197道数学题、29道聊天题，每题4候选。

严格数学：单次{m['means']['1']:.2f}%，2选1为{m['means']['2']:.2f}%，4选1为{m['means']['4']:.2f}%。4选1−单次：{interval(p)}个百分点。{conclusion}

聊天内容可用性（助手审查，非独立人工、非事实准确率）：单次{c['means']['1']:.3f}/2，4选1为{c['means']['4']:.3f}/2。奖励分自身提高不能证明质量提高。数学4选1用时约单次{m['seconds']['ratio4to1']:.2f}倍；聊天约{c['seconds']['ratio4to1']:.2f}倍。

使用原start_demo.bat启动、展开左侧设置。多候选关闭思考，每候选512词元/轮、最多3轮；生成与评分分阶段驻留显卡。只有选中的完整工具轨迹进入历史，未完成/过长候选不参与评分，全不合格时明确提示并保留首个尝试。原单次功能仍保留。标准答案不提供给挑选器。

完整协议、裁决、局限和结果见[多候选实验报告](results/best_of_n_20261026/REPORT.md)。旧权重与评测保留；不是前三模型的重新排名，也不是全新题库复测。
'''
    headings={'README.md':'## 13. 多候选回答与奖励挑选（2026-10-26）',
              'SUMMARY_FOR_CHATGPT.md':'## 14. 多候选回答与奖励挑选（2026-10-26）',
              'HANDOFF.md':'## 第七部分：2026-10-26 多候选回答与奖励挑选'}
    for file,heading in headings.items():
        path=P/'my_eval'/file;old=path.read_text(encoding='utf-8')
        backup=OUT/(file.removesuffix('.md')+'.before_best_of_n.md')
        if not backup.exists():backup.write_text(old,encoding='utf-8')
        if heading in old:old=old[:old.index(heading)].rstrip()
        note='最新功能（2026-10-26）：待办4已完成，网页支持多候选奖励挑选，实际效果与代价见[实验报告](results/best_of_n_20261026/REPORT.md)。\n\n'
        if note not in old:
            first,rest=old.split('\n',1);old=first+'\n\n'+note+rest.lstrip('\n')
        old=old.replace('4. 在 demo 里加一个开关：生成 N 个回答，用奖励模型挑最好的一个；并量化它带来的提升。',
                        '4. 【已完成，2026-10-26】网页加入2/4候选奖励挑选并量化实际效果，默认关闭；详见第七部分。')
        old=old.replace('4. 在 demo 里加"生成 N 个回答、用奖励模型挑最好的"，并测提升多少（解决同一问题答案时好时坏）。',
                        '4. 【已完成，2026-10-26】网页加入2/4候选奖励挑选并量化实际效果；详见第13节。')
        old=old.replace('降温度、best-of-N、投票这些方法都还没在 demo 里实现。',
                        '2026-10-26已实现并评估best-of-N多候选奖励挑选（见第14节）；投票尚未实现。')
        path.write_text(old.rstrip()+'\n\n'+heading+'\n\n'+section,encoding='utf-8',newline='\n')
    (OUT/'validation.json').write_text(json.dumps(dict(passed=True,protocol_hashes_verified=True,question_counts=[197,29],
        candidates=904,tests=tests,means_crosschecked=True,paired_standard_errors_crosschecked=8,
        automatic_audit_agreements=40,content_labels_locked=True,selection_checks=selection_checks,
        report_sha256=hashlib.sha256((OUT/'REPORT.md').read_bytes()).hexdigest()),ensure_ascii=False,indent=2),encoding='utf-8')
    files=['protocol.json','tasks.json','generated.json','scored.json','math_automatic.json','math_reviews.json',
           'chat_reviews.json','chat_review_rubric.json','summary.json','tests.json','REPORT.md','validation.json']
    manifest={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in files}
    for name in ['scripts/web_demo.py','scripts/demo_best_of_support.py','start_demo.bat']:
        manifest[name]=hashlib.sha256((P/name).read_bytes()).hexdigest()
    (OUT/'final_artifact_hashes.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Published report and updated three project documents.')

if __name__=='__main__':publish()
