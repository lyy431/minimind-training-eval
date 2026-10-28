"""Publish the measured final MiniMind chat repair, keeping all failed development."""
import chat_repair_common as e
from eval_chat_repair import verify_final
from review_chat_repair import key

def delta(p,precision=2):
    return f"{p['difference']:+.{precision}f} ± {p['se']:.{precision}f}；95%区间[{p['ci95'][0]:+.{precision}f}, {p['ci95'][1]:+.{precision}f}]"

def main():
    s=e.read(e.OUT/'summary.json');base=s['base'];name=s['candidate'];verify_final(name)
    validation=e.read(e.OUT/'independent_validation.json');assert validation['passed']
    assert not e.read(e.OUT/'final_review_queue.json') and not e.read(e.OUT/'final_math_review_queue.json')
    export=e.read(e.OUT/'demo_export.json');published=export['published']
    if published:
        assert s['accepted'] and e.read(e.OUT/'demo_ui_validation.json')['passed'] and e.read(e.OUT/'demo_smoke.json')['passed']
        for f,h in export['files_sha256'].items():assert e.sha(e.P/'scripts/MiniMind-YuyangLU-v4'/f)==h
    bb,aa=s['models'][base],s['models'][name]
    outcome='通过本轮预设验收，作为MiniMind-YuyangLU-v4加入网页。' if published else '未通过本轮发布验收，未加入网页；研究权重和全部原文保留。'
    text=f'''# MiniMind普通聊天质量修复

2026-10-27。{outcome}

这次改进的是MiniMind权重本身。Qwen2.5-1.5B仅用于本地生成部分训练答案，网页推理仍只用所选MiniMind；合并后模型仍约64M参数。所有工作在D盘本地完成，没有付费调用、联网下载或云训练。

## 新题正式结果

48道基础问题（知识、写作、信息提取、身份各12道），每题3种条件：无工具、6个原说明工具、6个已知改写说明工具。两模型各144次单次回答，关闭思考和多候选。8道基础题含预设的一轮简短历史。

| 指标 | 原v3 | 新版{name} |
|---|---:|---:|
| 内容平均分（0–2） | {bb['mean_score']:.3f} | {aa['mean_score']:.3f} |
| 主要请求得到可用回应（2分） | {bb['usable_percent']:.2f}% | {aa['usable_percent']:.2f}% |
| 非空、完整且无工具调用 | {bb['nonempty_no_call_percent']:.2f}% | {aa['nonempty_no_call_percent']:.2f}% |
| 平均实际生成词元 | {bb['mean_generated_tokens']:.1f} | {aa['mean_generated_tokens']:.1f} |
| 平均单次生成时间 | {bb['mean_seconds']:.3f}秒 | {aa['mean_seconds']:.3f}秒 |

内容分提升：{delta(s['pairs']['score'],3)}（0–2分单位）。可用率提升：{delta(s['pairs']['usable'])}个百分点。去掉身份题后的内容提升：{delta(s['pairs']['nonidentity_score'],3)}分。因此不能只靠多说出名字来满足聊天改进验收。

每题先对3条件取平均，再按48道基础题配对，bootstrap10000次、seed20261028；±是1SE。非身份分析为36道题。144次回答存在同题相关，不能当144道独立题。区间未作多重比较校正；单训练种子，不包含训练随机性和评分者不确定性。

## 条件与类别

每格为平均内容分 / 2分可用率。

| 输入条件 | 原v3 | 新版 |
|---|---|---|
'''
    for key,label in [('no_tools','无工具'),('six_original','6工具原说明'),('six_reworded','6工具已知改写说明')]:
        a,b=aa['conditions'][key],bb['conditions'][key]
        text+=f"| {label} | {b['mean_score']:.3f} / {b['usable_percent']:.2f}% | {a['mean_score']:.3f} / {a['usable_percent']:.2f}% |\n"
    text+='\n| 类别 | 原v3 | 新版 |\n|---|---|---|\n'
    for key,label in [('knowledge','知识解释'),('writing','短消息写作'),('retrieval','信息提取/原样复制'),('identity','身份内容')]:
        a,b=aa['categories'][key],bb['categories'][key]
        text+=f"| {label} | {b['mean_score']:.3f} / {b['usable_percent']:.2f}% | {a['mean_score']:.3f} / {a['usable_percent']:.2f}% |\n"
    text+=f"\n新版仍有{100-aa['usable_percent']:.2f}%的回答没有达到2分可用标准，不能把通过相对改进门槛理解成普遍可靠。\n"
    text+='''
“不调用工具”只衡量行为，不等于内容正确。0分为未满足主要请求、基本不可用或有严重可见错误；1分为部分回应但有明显遗漏、混乱或重复；2分为主要请求得到清楚实用回应且无可见严重错误。参考答案不进入模型、生成器或挑选器。

回答按固定随机顺序、隐藏模型/条件标签逐条由本助手评分，完全相同上下文和原文可复用已有裁决；不是独立人工盲评，也不是覆盖开放领域的事实准确率。所有原文、映射和判断保留供复查。

## 能力保留

| 指标 | 原v3 | 新版 |
|---|---:|---:|
'''
    for key,label in [('historical','历史严格数学197题'),('freshmath','上轮工具改写数学48题')]:
        r=s['retention'][key];text+=f"| {label} | {r['baseline']:.2f}% | {r['candidate']:.2f}% |\n"
    r=s['retention']['nll296'];text+=f"| 留出回答预测损失296条（低好） | {r['baseline']:.5f} | {r['candidate']:.5f} |\n"
    ib,ia=s['identity14'][base],s['identity14'][name]
    text+=f"| 既有14次无工具身份回答的完整可用数 | {ib['usable']}/14 | {ia['usable']}/14 |\n"
    text+=f"| 同14次名字命中（另列代理指标） | {ib['name_hits']}/14 | {ia['name_hits']}/14 |\n"
    text+=f'''
预测损失差异：{delta(r['paired'],5)}。损失确有上升，处于事先允许的0.05范围内，不应写成“语言能力完全不变”。数学分别沿用197题和上轮48题的冻结整段一致性评分，未知格式逐条核查，漏答、错配和相互矛盾不能只因出现正确数字就算过。48题是上轮已有的回归检查，不冒充本轮全新数学题；其差异区间[{s['retention']['freshmath']['paired']['ci95'][0]:+.2f}, {s['retention']['freshmath']['paired']['ci95'][1]:+.2f}]个百分点仍包含0，不能确认这组也提升。身份完整内容与名字命中分开，涉及原作者时须区分jingyaogong与本地定制者Yuyang LU；新版本14次仍有1次关系解释不完整。

数学与损失基线复用相同v3权重、问题、种子、模板和共享执行器的旧原始记录，导出参数/词表/模板已逐项验证；新候选重新生成/计算。普通聊天和14次身份两侧均重新生成。开发数学的旧缓存只有首轮路由，发现后已保留并重跑完整流程；未把它用于最终判分。细节见baseline_reuse.json。

## 验收与开发迭代

预设正式门槛：总内容分和非身份内容分提升均至少0.20且95%区间下界>0；可用率提升至少10个百分点；无工具可用率下降≤5个百分点；两组数学各下降≤5个百分点；身份完整可用数少≤1/14；NLL296增加≤0.05；普通题非空完整无调用≥90%。这是项目接受线，不是统计等效性证明。

| 门槛 | 通过 |
|---|---|
'''
    gate_labels={'dev_gates':'开发检查通过','content_gain_ge020':'内容分提升至少0.20','content_gain_ci_lower_gt0':'内容提升区间下界大于0',
        'nonidentity_gain_ge020_ci_lower_gt0':'非身份内容同样提升且区间下界大于0','usable_gain_ge10pp':'可用率提高至少10个百分点',
        'no_tools_usable_drop_le5pp':'无工具可用率保留','historical_math_drop_le5pp':'197题数学保留','prior_round_math48_drop_le5pp':'上轮48题数学保留',
        'nll_increase_le005':'预测损失增加不超过0.05','identity_full_content_drop_le1':'14次身份完整可用数保留',
        'ordinary_nonempty_no_call_ge90':'普通题非空完整无调用至少90%','data_tests':'数据与输入隔离检查通过'}
    for k,v in s['gates'].items():text+=f"| {gate_labels[k]} | {'是' if v else '否'} |\n"
    text+='''
最初测试去工具说明、改提示语和已有小模型替换，仅用于16题开发诊断，未发布；用户随后明确选择训练MiniMind本身。诊断目录chat_quality_20261027保留，不是网页功能。

全参数3轮后e2/e3开发内容分由原v3的0.417提高到0.771/2，但身份回答有遗漏。最初e2选择锁在任何最终题回答生成前撤回，锁、源码和数据快照完整保存在development_v1_before_identity；原正式接受门槛未降低，开发增加身份保留过滤，然后才做身份补充。不能把先前开发选择称为一次失败的最终盲测。

身份补充只训练实际e2基座的q_proj/o_proj rank16 LoRA，393216参数，统一完整的中英文身份目标，同时加入普通/数学监督和冻结e2的KL保留。基座及KL教师不更新；三个补丁都只合并回实际e2基座，逐张量独立重建一致，见identity_merge_validation.json。

只在开发集比较原3轮和身份补充3轮，要求普通题非空无调用≥90%、NLL100增加≤0.05、开发完整数学下降≤10个百分点，以及4道无工具身份的完整可用数不低于基线3/4；按内容分、可用率、较低损失选择。新的final_lock.json之后不再训练或换版本。最终答案归集器原先把同前缀的identity_merge_validation.json当成回答文件，随后用只接受两个原始结果文件的包装器修复；生成、题目和评分规则未变。损失统计的好/差标签另按“低者更好”注明，不改差值、区间或验收；见review_queue_amendment.json和metric_direction_annotation.json。

## 数据、训练和来源

老师使用本地Qwen2.5-1.5B，batch8、贪心、最多128新词元。480个请求中378条完整；102条未完整结束的回答排除，不截断后当成正确示范。内容检查显示旧通用问题池含明显错误、信息不足及不适合本轮的题，只保留40条经检查的通用回答；另160条简单知识示范全部阅读，合计200条，43条有明确修正记录。原文从未覆盖，见teacher_answers.json、teacher_curation.json。不是独立专家事实审查。

另560条固定目标：160写作（40场景×4问法）、320提取/复制（8类格式）、80身份（10类问题×8前缀）。合计760道普通训练问题，每题分别配无工具和有工具，共1520行；另640行既有数学监督、400一般KL和200数学KL，总2760行。普通题的工具数1–6×6种说明36格各21/22条，联合打乱，避免工具数和说明绑定。未平衡准备稿保留于development_prebalance，修正发生在任何参数更新之前。

开发/最终聊天和NLL100/300问题做规范化精确排除；未声称排除所有上游语义重合。训练知识只有40个基础话题的改写，写作和提取也有固定结构，因此新题结果不能外推所有自然表达或专家问题。

从精确v3全参数训练：lr0.00001、batch4、3轮2070步、seed42、最大1536词元，最长1418，无截断；KL系数2，数学首64个回答词元权重2。随后实际e2上的身份补充1480行：身份240、其他普通320、数学320、KL600；lr0.0002、batch4、3轮1110步、seed42、KL4、保留首32词元权重3、身份监督行权重3。源码、数据摘要、试跑和逐轮日志均保留。

全参数约5.5分钟、PyTorch峰值分配显存4.13GiB；身份补充约2.5分钟，另有老师生成、进程载入和评测耗时。不是全系统显存测量或固定计算预算算法基准。没有付费、没有下载新权重；老师不参与网页推理。

## 使用与局限

'''
    text+=outcome+'\n\n'
    if published:text+='双击start_demo.bat或刷新原网页，选择MiniMind-YuyangLU-v4。v4默认384个新词元，最多勾选6个工具；原v3及全部旧模型保留。新模型不依赖多候选，开启多候选后的v4效果没有在本轮评测。天气、汇率等部分工具仍为示例返回，未新增实时服务。\n\n'
    text+='''导出所有参数张量、词表和模板逐项相等，并检查3个输入的24个贪心词元一致；真实Streamlit界面再以相同种子验证身份、双算式和写作流程，与共享执行结果对照。单次演示不是额外的质量选择集。

较短回答可能更易读，也会减少本题组耗时；这不表示模型变成了通用可靠助手。复杂知识、英语和多轮对话仍需逐例核实。本轮只有48道基础题、单个训练种子和助手内容评分，真实用户体验需要继续收集反馈。

关键文件：plan.json、data_audit.json、token_audit.json、selection.json、final_lock.json；devchat_*.json、devmath_*.json；finalchat_*.json、historical_*.json、freshmath_*.json、identity_*.json、loss_final_*.json；content_reviews.json、math_reviews.json、summary.json、independent_validation.json；demo_export.json、demo_ui_validation.json。
'''
    (e.OUT/'REPORT.md').write_text(text,encoding='utf-8',newline='\n')
    brief=f'''2026-10-27。{outcome}

本轮改善MiniMind权重本身：从v3全参数补充聊天训练，再在实际聊天改进基座上训练身份LoRA并只合并回同一基座。Qwen仅用于本地训练数据生成；网页推理仍只使用MiniMind。所有旧模型、试跑和开发失败均保留，没有付费调用。

48道新聊天题×3条件，同设置单次回答：内容平均分{bb['mean_score']:.3f}→{aa['mean_score']:.3f}/2，差异{delta(s['pairs']['score'],3)}分；可用率{bb['usable_percent']:.2f}%→{aa['usable_percent']:.2f}%。非身份内容变化{delta(s['pairs']['nonidentity_score'],3)}分；不把无调用率当内容准确率。

旧严格数学197题{s['retention']['historical']['baseline']:.2f}%→{s['retention']['historical']['candidate']:.2f}%；上轮改写数学48题{s['retention']['freshmath']['baseline']:.2f}%→{s['retention']['freshmath']['candidate']:.2f}%；NLL296为{s['retention']['nll296']['baseline']:.5f}→{s['retention']['nll296']['candidate']:.5f}；既有14次身份完整可用{ib['usable']}/14→{ia['usable']}/14。选择版本{name}，原v3保留。

数据2760行；三个全参数版本之后发现身份退步，在任何最终回答生成前保留原选择锁并增加身份开发检查，另用1480行做身份补充。所有最终接受门槛未降低。评分为隐藏模型标签的助手逐条复核，非独立人工盲评；少量题和单训练种子，不能外推开放领域事实准确率。详见[完整聊天修复报告](results/chat_repair_20261027/REPORT.md)。
'''
    headings={'README.md':'## 16. 普通聊天质量修复（2026-10-27）','SUMMARY_FOR_CHATGPT.md':'## 17. 普通聊天质量修复（2026-10-27）','HANDOFF.md':'## 第十部分：2026-10-27 普通聊天质量修复'}
    for file,heading in headings.items():
        path=e.P/'my_eval'/file;old=path.read_text(encoding='utf-8');backup=e.OUT/(file.removesuffix('.md')+'.before_chat_repair.md')
        if not backup.exists():backup.write_text(old,encoding='utf-8')
        if heading in old:old=old[:old.index(heading)].rstrip()
        old=old.replace('最新进展（2026-10-26）：工具选择第二轮','上轮进展（2026-10-26）：工具选择第二轮')
        note=f"最新进展（2026-10-27）：MiniMind本体聊天修复{'通过验收，v4已加入网页' if published else '本轮未达到发布门槛'}，见[聊天修复报告](results/chat_repair_20261027/REPORT.md)。\n\n"
        if note not in old:
            first,rest=old.split('\n',1);old=first+'\n\n'+note+rest.lstrip('\n')
        if published:
            old=old.replace('目前有7个demo模型文件夹','目前有8个demo模型文件夹')
            old=old.replace('目前有7个模型。','2026-10-27加入普通聊天修复版MiniMind-YuyangLU-v4，目前有8个模型。')
            old=old.replace('当前有7个模型：MiniMind-YuyangLU-v3','当前有8个模型：MiniMind-YuyangLU-v4（'+name+'，普通聊天修复版）、MiniMind-YuyangLU-v3')
            old=old.replace('更新：2026-10-26。本文件','更新：2026-10-27。本文件')
            if file=='README.md':old=old.replace('时间：2026-10-20 ～ 2026-10-25','时间：2026-10-20 ～ 2026-10-27')
            if file=='HANDOFF.md' and 'v4由export_chat_repair.py导出' not in old:
                old=old.replace('新模型文件夹放进 scripts\\ 后，刷新网页就能看到，不用重启。','v4由export_chat_repair.py导出，真实网页验证见第十部分。新模型文件夹放进 scripts\\ 后，刷新网页就能看到，不用重启。')
        path.write_text(old.rstrip()+'\n\n'+heading+'\n\n'+brief,encoding='utf-8',newline='\n')
    e.save('publication_validation.json',dict(passed=True,accepted=s['accepted'],published=published,report_sha256=e.sha(e.OUT/'REPORT.md'),all_locked_inputs_verified=True))
    print('Report and three project documents updated.',flush=True)

if __name__=='__main__':main()
