"""Update project handoff documents only after final acceptance and demo validation."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/identity_repair_20261024"
s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
demo = json.loads((OUT / "demo_export.json").read_text(encoding="utf-8"))
assert s["accepted"] and demo["model"] == s["selected"]
assert all(r["same_argmax"] for r in demo["checks"]) and demo["cached_greedy_24_tokens_identical"]
name, models = s["selected"], s["models"]
r, base_diff, old_diff = models[name], s["selected_minus"]["bb_rl"], s["selected_minus"]["bb_rl_id"]
train = s["selected_training"]
link = "[完整身份改进报告](results/identity_repair_20261024/REPORT.md)"
table = "| 模型 | 身份名字命中 /21 | 无关回答命中MiniMind/Yuyang /20 | 回答 loss（296条，低好） | 数学全对率 | 聊天奖励模型分 |\n|---|---:|---:|---:|---:|---:|\n"
for n in ["bb_rl", "bb_rl_id", name]:
    row = models[n]
    table += f"| {n} | {row['identity_hits']} | {row['general_leaks']} | {row['loss']:.4f} | {row['math_percent']:.1f}% | {row['rm']:.3f} |\n"

body = f"""2026-10-25 完成待办 2。在本地训练、评测和导出，全程未使用付费云服务。完整原始回答、失败候选、数据审查和逐题统计见 {link}。

{table}
新身份版保留为独立模型 **MiniMind-YuyangLU-v2**，位置 `scripts/MiniMind-YuyangLU-v2`。刷新网页即可选择；旧 MiniMind-YuyangLU 和 MiniMind-Math 等模型全部保留，目前共有6个可选模型。

相对主线 bb_rl：数学差值 {base_diff['math_percentage_points']['difference']:+.2f} ± {base_diff['math_percentage_points']['se']:.2f} 个百分点，语言 loss 差值 {base_diff['loss']['difference']:+.4f} ± {base_diff['loss']['se']:.4f}。相对旧身份版，数学变化 {old_diff['math_percentage_points']['difference']:+.2f} ± {old_diff['math_percentage_points']['se']:.2f} 个百分点。这里 ± 均为按题配对的1个标准误，不是95%置信区间；达到本轮实用门槛不等于证明完全没有能力损失。

聊天奖励模型分相对 bb_rl：{base_diff['rm']['difference']:+.3f} ± {base_diff['rm']['se']:.3f}。语言 loss 和奖励模型分衡量不同东西，不能只凭其中一个宣称聊天能力全面提高。

训练：从 bb_rl 重新训练 LoRA，rank16、仅 q_proj/o_proj、0.393M可训练参数；学习率 {train['learning_rate']:g}、{train['epochs']}轮、batch8、序列768、2个数据加载进程。权重 `out/{name}_768.pth`，adapter `out/{train['name']}_768.pth`。该候选训练耗时 {train['seconds']/60:.1f} 分钟，PyTorch峰值分配显存 {train['peak_gpu_allocated_gib']:.2f} GiB（不含桌面占用）。

过程与数据：
- 两个系统提示词均未成功；最初三个低学习率候选虽然保住数学，但开发身份命中仅0/2/5。
- 追加开发实验在正式验收前单独记录：从旧身份数据筛出42条，另由本次助手编写48条中英文问法；90条各重复6次，搭配1000条普通对话、128条 bb_rl 自己答对的完整数学轨迹各重复2次，组成1796条配方。因两个候选都出现英文身份串题，第三版统一身份答案，并添加64条普通英文问答各4次和32条中文问答各2次，组成2116条配方。本次选中的数据为 `{Path(train['dataset']).name}`。原数据及所有候选均保留。
- 数学复习轨迹通过完整数字、计算器结果、上下文长度检查；训练与开发/最终测试的规范化用户问题交集为0。来源、摘要和追加试验条件见 development_extension_v2.json 与 development_extension_v3.json。
- 选型使用另建的24道数学开发题、100条语言开发题、10道身份和10道普通问题；选中后锁定模型与评测文件摘要，才运行固定最终评测。

最终训练目标：身份题正常学习目标答案，普通题与数学题则以冻结的 bb_rl 为教师，用KL约束保持原输出分布。系数为 {train.get('kl_lambda', '未使用')}，回答最初32个有效token额外加权；记录见 development_extension_v4.json。新增训练器为 train_identity_anchored.py，原官方训练器未改。此法也会保留原模型已有的缺点。学习率、轮数也与前一候选不同，不能把效果全部归因于KL这一个变量。

评测说明：本次身份测试逐题固定随机种子、名字不区分大小写，并重跑三个模型；因此本表不能与旧脚本的17/21直接作配对比较。身份命中是字符串指标，内容审查见完整报告；旧身份版训练题与7道身份回归题中的{s['old_identity_training_final_question_overlap']}道同文，新补丁已排除这些直接重复。数学和聊天的旧对照复用历史完整结果，两个对照各6次数学和3次聊天输出重新逐字复现通过；三个模型的语言、身份重新评测。

内容检查：新版本常规6道身份题的18次回答均清楚说明定制身份；关于Jingyao Gong的3次中，2次仅复述身份，1次跑题，因此名字命中20/21不等于完整回答正确20/21。普通回答关键词串题为1/20，内容审查发现另1次不含关键词的自称助手，两次均发生在请假条题上；内容口径为2/20。通用聊天仍有重复、事实错误，聊天奖励模型分也未改善。没有据此继续调参。

导出验证：以fp16取整后的权重、fp32计算核验，6种中英文/工具输入、共{sum(x['tokens'] for x in demo['checks'])}个token位置的首选词完全相同，最大logit差异{max(x['max_logit_difference'] for x in demo['checks']):.2e}，24token缓存生成一致；验证通过后才加入网页目录。网页实际使用fp16计算，随机采样回答仍会变化。

新增脚本位于 my_eval/scripts：identity_repair.py、train_identity_candidate.py、prepare_identity_extension.py、prepare_identity_clarity.py、replay_identity_baselines.py、write_identity_report.py、export_identity_demo.py、sync_identity_docs.py。主要产物位于 my_eval/results/identity_repair_20261024；日志为 my_eval/logs/identity_*.log。

其他支持脚本：train_identity_anchored.py、plan_identity_anchor.py、run_identity_final.py、record_identity_review.py。make_demo_models.py 的导出校验已修正：只将参数按fp16取整，保留位置编码缓存fp32，避免原先half再float引入假差异；原权重及正式评测未改。
"""


def patch(filename, changes, title):
    path = ROOT / filename
    text = path.read_text(encoding="utf-8")
    for old, new in changes:
        if new in text:
            continue
        if old in text:
            text = text.replace(old, new)
        else:
            raise ValueError(f"Expected original text absent: {filename}: {old[:70]}")
    section = title + "\n\n" + body.strip() + "\n"
    if title in text:
        start = text.index(title)
        later = re.search(r"^## ", text[start+len(title):], flags=re.M)
        end = start+len(title)+later.start() if later else len(text)
        text = text[:start] + section + "\n" + text[end:]
    else:
        text = text.rstrip() + "\n\n---\n\n" + section
    path.write_text(text.rstrip()+"\n", encoding="utf-8", newline="\n")


patch("README.md", [
    ("时间：2026-10-20 ～ 2026-10-24", "时间：2026-10-20 ～ 2026-10-25"),
    ("2. **重新训练一个温和的身份 LoRA**：降低学习率（比如 1e-4）、减少轮数、多混普通对话（约 1000 条），复习数学时用 `bb_rl` 自己答对的轨迹代替 Qwen 的轨迹。目标：身份 ≥ 15/21，留出 loss 接近 `bb_rl`（+0.03），数学接近 81.8%。",
     "2. **已完成：改进身份 LoRA**（2026-10-25）：新版本 MiniMind-YuyangLU-v2 已通过本轮验收，训练和正式结果见第10节。"),
    ("## 1. 一页总结", "## 1. 一页总结\n\n最新补充（2026-10-25）：官方对照及身份改进已完成；新身份版 MiniMind-YuyangLU-v2 已加入网页。新版本与修正后的296条语言口径见[第10节](#10-身份补丁改进与新演示模型2026-10-25)，下面早期实验表保留历史口径。"),
], "## 10. 身份补丁改进与新演示模型（2026-10-25）")

patch("SUMMARY_FOR_CHATGPT.md", [
    ("最后做了一个 Streamlit 网页 demo；2026-10-24 加入官方对照后，里面有 5 个模型。",
     "最后做了一个 Streamlit 网页 demo；2026-10-24 加入官方对照，2026-10-25 加入通过验收的新身份版 MiniMind-YuyangLU-v2，目前有6个模型。"),
    ("3. bb_rl_id 的身份补丁伤了通用语言能力（Δloss +0.143）。可能的修法：降低 lr、减少 epoch、多混通用对话、用 bb_rl 自己的轨迹复习——都还没试。",
     "3. 旧 bb_rl_id 的身份补丁损伤已开展修复实验；2026-10-25 选出并正式验收了新身份版，见第11节。旧版仍保留用于比较。"),
], "## 11. 身份功能改进已完成（2026-10-25 接续更新）")

patch("HANDOFF.md", [
    ("后续优先考虑第 2 项。", "第1、2项均已完成，最新状态见末尾更新。"),
    ("2. 重新训练一个温和的身份 LoRA：降低学习率（比如 1e-4），减少轮数，多混普通对话（约 1000 条），复习数学时用 bb_rl 自己答对的轨迹。目标：身份 ≥ 15/21，留出 loss 接近 bb_rl（+0.03），数学接近 81.8%。合格后用 make_demo_models.py 重新导出 MiniMind-YuyangLU。",
     "2. 【已完成，2026-10-25】改进身份 LoRA：新版本 MiniMind-YuyangLU-v2 已通过本轮验收并加入网页，旧版保留。训练数据、失败尝试和正式结果见本文件末尾更新。"),
    ("待办 1 已完成，尚未启动其他训练待办。", "此处记录待办1完成时的状态；第2项的后续结果见下一节。"),
    ("有 4 个模型：MiniMind-YuyangLU（bb_rl_id）、MiniMind-Math（bb_rl，主线最终模型）、MiniMind-SFT（full_sft）、MiniMind-MoE（full_sft_moe）。",
     "当前有6个模型：MiniMind-YuyangLU-v2（bb_rl_id_anchor4，新身份版）、MiniMind-YuyangLU（bb_rl_id，旧身份版）、MiniMind-Math（bb_rl，主线最终模型）、MiniMind-SFT（full_sft）、MiniMind-MoE（full_sft_moe）、MiniMind-Official（官方对照）。"),
], "## 第四部分：2026-10-25 身份改进接续更新")
print("Updated README.md, SUMMARY_FOR_CHATGPT.md and HANDOFF.md.")
