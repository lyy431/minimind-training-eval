"""Render the completed official comparison and supplementary examples to Markdown."""
import json
from pathlib import Path
import statistics as st

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "my_eval/results/official_compare_20261024"
summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
extra = json.loads((OUT / "supplement.json").read_text(encoding="utf-8"))
names = ["full_sft", "official", "bb_rl", "bb_rl_id"]
labels = {"full_sft": "自己的基础模型 full_sft", "official": "官方 minimind-3", "bb_rl": "数学模型 bb_rl", "bb_rl_id": "身份模型 bb_rl_id"}
m = summary["models"]
p = summary["official_minus"]


def diff(d, digits=3):
    return f"{d['mean_difference']:+.{digits}f} ± {d['se']:.{digits}f}"


lines = ["# 官方 MiniMind-3 与本地模型对比", "", "日期：2026-10-24（悉尼）。全部在本地 RTX 4070 Laptop 8GB 上完成，无需云端付费或重新训练。", "",
    "## 结论", "",
    f"官方模型预测参考回答的能力明显领先：修正后的回答 loss 比自己的基础模型低 {abs(p['full_sft']['answer_only_loss']['mean_difference']):.3f}，296 条里有 {p['full_sft']['answer_only_loss']['better']} 条更好；但聊天奖励模型分只高 {diff(p['full_sft']['rm'])}，本轮未测出明确的聊天质量优势。",
    f"数学工具题上，官方全对率为 {m['official']['math_percent']:.1f}%，自己的 bb_rl 为 {m['bb_rl']['math_percent']:.1f}%；这衡量的是固定工具定义和判分器下的任务表现，不能推广为通用数学能力。", "",
    "## 完整对比", "",
    "| 模型 | 回答 loss，296 条，越低越好 | 历史 loss，300 条 | 聊天奖励模型分 | 聊天总奖励 | 数学全对率 |",
    "|---|---:|---:|---:|---:|---:|"]
for n in names:
    r = m[n]
    lines.append(f"| {labels[n]} | {r['answer_only_loss']:.4f} | {r['loss']:.4f} | {r['rm']:.3f} | {r['total']:.3f} | {r['math_percent']:.1f}% |")
lines += ["", "聊天的两个分数不能互换：总奖励还包括长度、思考格式与重复惩罚。上述 loss 是先对每条回答的 token 求平均，再对回答求平均，不是按全体 token 加权。", "",
    "## 逐题配对差异", "", "下表差值统一为“官方 − 对照”，± 为 1 个标准误，不是 95% 置信区间；loss 越低越好，其余越高越好。", "",
    "| 对照 | 指标 | 差值 ± 1 SE | 官方更好 / 更差 / 相同 |", "|---|---|---:|---:|"]
for n in ["full_sft", "bb_rl", "bb_rl_id"]:
    for key, label, digits in [("answer_only_loss", "回答 loss（296 条）", 3), ("loss", "历史 loss（300 条）", 3),
                               ("rm", "聊天奖励模型分（30 题）", 3), ("total", "聊天总奖励（30 题）", 3),
                               ("math_percentage_points", "数学全对率（百分点，200 题）", 1)]:
        r = p[n][key]
        lines.append(f"| {n} | {label} | {diff(r, digits)} | {r['better']} / {r['worse']} / {r['same']} |")
lines += ["", "## 数学题难度拆分", "", "| 模型 | 1 个算式 | 2 个算式 | 3 个算式 |", "|---|---:|---:|---:|"]
for n in names:
    r = m[n]["math_by_expressions"]
    lines.append(f"| {n} | {r['1']:.1f}% | {r['2']:.1f}% | {r['3']:.1f}% |")
math_official = json.loads((OUT / "math.json").read_text(encoding="utf-8"))["official"]
math_bbrl = json.loads((PROJECT / "my_eval/results/agent_bbrl200.json").read_text(encoding="utf-8"))["bb_rl"]
short_official = 100*st.mean(r["calc_calls"] < len(r["gt"]) for r in math_official)
short_bbrl = 100*st.mean(r["calc_calls"] < len(r["gt"]) for r in math_bbrl)
lines += ["", f"官方在 {short_official:.1f}% 的作答中调用计算器的次数少于算式数，bb_rl 为 {short_bbrl:.1f}%；这与多算式题的差距一致。原 compare_agent.py 已独立复核总分、难度拆分及配对统计，日志为 official_math_vs_sft.log 和 official_math_vs_bbrl.log。"]
lines += ["", "## 语言评测修正", "",
    "发现旧 heldout_loss.py 的边界情况：输入按 768 token 截断后，第 78、96、116、121 条（从 0 编号）的回答被全部截掉，旧逻辑却把第 768 个提示词 token 当作回答计分。",
    "本次没有悄悄替换历史数字：完整保留原 300 条口径，同时报告排除这 4 条后的 296 条回答口径；四个模型都用同一个有效子集。旧 full_sft 的 2.4905 已复现。修正不改变四个模型的语言 loss 排序，但今后比较应明确口径。",
    "my_eval/scripts/heldout_loss.py 已增加有效回答检查，并同时输出修正值与历史值。原训练代码与已有权重没有更改。", "",
    "## 公平性、来源与限制", "",
    "- 仅评价本地已下载的 minimind-3；来源目录、SHA256、软件版本和模型配置保存在 manifest.json，未声称这是最新在线版本。",
    "- 原始 HF 模型和转换后的原生模型使用相同词表、特殊词元和对话模板；6 组输入共 918 个位置上最高概率词一致，最大 logit 误差小于 0.00009，24 token 的缓存生成一致。",
    "- 数学保持 200 题 × 2 次、种子 10000×题号+次数、温度 0.8、top_p 0.85、top_k 50、关闭思考、每轮最多 384 token、最多 3 轮；沿用 rollout_single 与 validate_gt_in_text，真实执行本地模拟工具。判分依旧是最终文本包含标准答案数字，不另行核验整段推理。",
    "- 聊天保持原 30 题 × 3 次，种子 1000+题号、温度 0.8、top_p 0.85、top_k 50、开启思考、512 token，奖励公式逐项与历史源代码及 360 条原评分核对一致。",
    "- 本次新跑了四个模型的文本 loss、官方全部聊天与数学；full_sft / bb_rl / bb_rl_id 的聊天和数学复用历史完整结果。作为运行一致性检查，full_sft 的 6 个聊天回答与 6 次数学作答已逐字复现。",
    "- 本地评测题的留出，只针对本项目相应的训练划分；官方模型的训练历史和是否见过这些数据并未确定。因此，这是现成模型在共同测试上的表现比较，不能据此单独归因于训练数据量或某个算法。",
    "- 所有数值比较都按题/回答配对，避免把同一道题的多次采样当作独立题目。", "",
    "## 新题补充（探索性）", "",
    "在查看新题输出之前固定了 6 道数学题（1、2、3 个算式各两题，中英文各三题）和 6 个日常聊天问题；数学问题与 agent_rl_math.jsonl 的 20,000 条原始用户问题没有完全相同的文本。这里只检查完整题目不重复，不保证子算式或普通语言片段不曾出现。",
    "数学继续采用正式评测的设置，每题 2 次；聊天使用关闭思考的贪心生成，每题 1 次、最多 192 token。样本很少，仅作具体例子和诊断，不用来替代正式测试或声称广泛泛化。", "",
    "| 模型 | 新数学题答对次数 / 12 | 全对率 |", "|---|---:|---:|"]
for n in names:
    rows = extra["math"][n]
    assert len(rows) == 12
    lines.append(f"| {n} | {sum(r['all'] for r in rows)} / 12 | {100*st.mean(r['all'] for r in rows):.1f}% |")
lines += ["", "完整新题回答见 [补充回答](supplement_answers.md)，全部问题和输出均保留，不挑选最佳回答。", "",
    "## 网页演示与复现", "",
    "官方原始 HF 文件已复制为 scripts/MiniMind-Official，各文件摘要与来源相同；在现有网页刷新后，Model 列表可选 MiniMind-Official。默认模型保持原有排序行为。", "",
    "使用 D:\\conda\\envs\\minimind\\python.exe 运行下列脚本阶段：", "",
    "```text", "my_eval/scripts/official_compare.py prepare", "my_eval/scripts/official_compare.py smoke", "my_eval/scripts/official_compare.py loss",
    "my_eval/scripts/official_compare.py chat", "my_eval/scripts/official_compare.py math",
    "my_eval/scripts/official_supplement.py freeze", "my_eval/scripts/official_supplement.py run",
    "my_eval/scripts/official_compare.py report", "my_eval/scripts/write_official_report.py", "```", "",
    "chat/math 支持接着当前保存的逐题进度运行；它们复用现有结果，不会在同一目录自动启动一个独立的新实验。若要独立复测，应先使用新的输出目录。所有缓存设置在 D 盘、禁止联网获取新模型，日志位于 my_eval/logs/official_*.log。", "",
    "结果文件：manifest.json（来源与设置）、smoke.json（旧结果复现）、loss.json（每条 loss）、chat.json（90 个新聊天回答）、math.json（400 次新数学作答）、summary.json（配对统计）、supplement.json（新题输出）。", ""]
(OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")

examples = ["# 新题补充：全部回答", "", "这些问题在查看模型输出前固定；不从多次生成中挑选最好答案。聊天为贪心生成、关闭思考、192 token 上限。", ""]
for i, spec in enumerate(json.loads((OUT / "new_chat6.json").read_text(encoding="utf-8"))):
    examples += [f"## 聊天 {i+1}：{spec['skill']}", "", spec["prompt"], "", f"预先规定的检查点：{spec['check']}", ""]
    for n in names:
        row = extra["chat"][n][i]
        examples += [f"### {labels[n]}", "", "````text", row["response"], "````", ""]
math_questions = [json.loads(l) for l in (OUT / "new_math6.jsonl").read_text(encoding="utf-8").splitlines()]
for i, question in enumerate(math_questions):
    prompt = next(x["content"] for x in question["conversations"] if x["role"] == "user")
    examples += [f"## 数学 {i+1}", "", prompt, "", "标准答案：" + ", ".join(question["gt"]), ""]
    for n in names:
        examples += [f"### {labels[n]}", ""]
        for j, row in enumerate(r for r in extra["math"][n] if r["problem"] == i):
            examples += [f"采样 {j+1}；按原判分器全对：{row['all']}；工具调用次数：{row['n_calls']}。", "", "````text",
                         "\n\n--- 下一轮 ---\n\n".join(row["turns"]), "````", ""]
(OUT / "supplement_answers.md").write_text("\n".join(examples), encoding="utf-8", newline="\n")
print(OUT / "REPORT.md")
