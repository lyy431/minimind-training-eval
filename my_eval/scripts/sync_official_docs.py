"""Append the completed official comparison and update handoff status, idempotently."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/official_compare_20261024"
s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
assert (OUT / "REPORT.md").exists()
m, p = s["models"], s["official_minus"]
link = "[完整对比报告](results/official_compare_20261024/REPORT.md)"


def patch(path, replacements, section):
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
        elif new not in text:
            raise ValueError(f"Expected text absent from {path.name}: {old[:80]}")
    title = section.splitlines()[0]
    if title in text:
        start = text.index(title)
        following = re.search(r"^## ", text[start+len(title):], flags=re.M)
        end = start+len(title)+following.start() if following else len(text)
        text = text[:start] + section.rstrip() + "\n\n" + text[end:]
    else:
        text = text.rstrip() + "\n\n---\n\n" + section
    path.write_text(text.rstrip()+"\n", encoding="utf-8", newline="\n")


table = "| 模型 | 修正后的回答 loss（296 条，越低越好） | 旧口径 loss（300 条） | 聊天奖励模型分 | 数学全对率 |\n|---|---:|---:|---:|---:|\n"
for n in ["full_sft", "official", "bb_rl", "bb_rl_id"]:
    r = m[n]
    table += f"| {n} | {r['answer_only_loss']:.4f} | {r['loss']:.4f} | {r['rm']:.3f} | {r['math_percent']:.1f}% |\n"
loss_diff = p["full_sft"]["answer_only_loss"]
chat_diff = p["full_sft"]["rm"]
math_diff = p["full_sft"]["math_percentage_points"]
body = f"""2026-10-24 在本地完成；详细协议、逐题统计、来源摘要、全部新题输出见 {link}。

{table}
主要结论：
- 官方相对 full_sft 的修正回答 loss：{loss_diff['mean_difference']:+.3f} ± {loss_diff['se']:.3f}（1 SE），好 / 差 / 相同为 {loss_diff['better']} / {loss_diff['worse']} / {loss_diff['same']}。
- 聊天奖励模型分只高 {chat_diff['mean_difference']:+.3f} ± {chat_diff['se']:.3f}，没有测出明确优势，不能把更低 loss 直接说成聊天更好。
- 官方数学 {m['official']['math_percent']:.1f}%，相对 full_sft {math_diff['mean_difference']:+.1f} ± {math_diff['se']:.1f} 个百分点；bb_rl 仍为 {m['bb_rl']['math_percent']:.1f}%。这只是当前数学工具题协议的结果。

评测修正：旧 heldout_loss.py 在 4 条回答被 768-token 截断完全删除的样本上，误算了提示词 token。受影响的 0 起始索引是 78、96、116、121。本次同时保留旧 300 条口径和修正的 296 条回答口径，四个模型使用同一个有效子集；脚本已同时输出两套数值。旧报告中的 loss 数字仍是历史 300 条口径，不能和 296 条的新数值直接相减。

复用与验证：四个模型的 loss 本次新跑；官方的聊天 30×3、数学 200×2 本次新跑；其余三个模型的聊天与数学复用历史完整结果。full_sft 的 6 个聊天回答及 6 次数学作答已逐字复现；转换后的官方模型通过 strict 加载、6 组输入 logits 与 24 token 连续生成校验。另固定 6 道新数学题和 6 个新聊天问题进行探索性检查，全量输出已保存。

限制：官方模型是否见过这些公开数据不能确定，不能把相同题目上的表现差异直接归因于某个训练步骤。新题样本少，不用于宣称广泛泛化。

新增模型：out/official_768.pth；网页目录 scripts/MiniMind-Official（原始 HF 文件逐个核对摘要）。刷新网页后即可选择 MiniMind-Official，现在有 4 个自训模型和 1 个官方对照模型。

新增脚本：official_compare.py（校验/复现/评测/统计）、official_supplement.py（固定与评测新题）、write_official_report.py（报告）、sync_official_docs.py（更新交接状态），都在 my_eval/scripts。主结果在 my_eval/results/official_compare_20261024，日志在 my_eval/logs/official_*.log。
"""

patch(ROOT / "README.md", [
    ("时间：2026-10-20 ～ 2026-10-23", "时间：2026-10-20 ～ 2026-10-24"),
    ("1. **和作者的官方模型对比**：官方 `minimind-3`（Hugging Face 格式）已经在项目的 `minimind-3/` 文件夹里。把它复制进 `scripts/` 放进 demo，再测留出 loss 和聊天打分，看自己训练的模型和官方的差多少。",
     "1. **已完成：和作者的官方模型对比**（2026-10-24）：已加入网页，并完成语言、聊天、数学和小规模新题检查，结果见第 9 节。")
], "## 9. 官方模型对比与语言评测修正（2026-10-24）\n\n" + body)

patch(ROOT / "SUMMARY_FOR_CHATGPT.md", [
    ("最后做了一个 Streamlit 网页 demo，里面有 4 个模型。", "最后做了一个 Streamlit 网页 demo；2026-10-24 加入官方对照后，里面有 5 个模型。"),
    ("5. 自己训练的模型和官方 `minimind-3` / `minimind-3-moe` 的差距没测过（官方稠密版已在本地，官方 MoE 版没下载）。",
     "5. 官方稠密版 `minimind-3` 对比已于 2026-10-24 完成（见第 10 节）；官方 MoE 版尚未下载或评测。")
], "## 10. 官方对照已完成（2026-10-24 接续更新）\n\n" + body)

patch(ROOT / "HANDOFF.md", [
    ("1. 和官方 minimind-3 对比：把 D:\\minimind\\minimind-master\\minimind-3 复制进 scripts\\，放进 demo；再用 heldout_loss.py 和 score_rm.py 测它（可以用 scripts\\convert_model.py 的 convert_transformers2torch 转成 .pth），回答\"自己训练的和作者的差多少\"。",
     "1. 【已完成，2026-10-24】和官方 minimind-3 对比：已加入 demo，完成语言、聊天、数学和新题检查；详见本文件末尾接续更新及 my_eval/results/official_compare_20261024/REPORT.md。后续优先考虑第 2 项。")
], "## 第三部分：2026-10-24 接续更新\n\n" + body + "\n上文最初交接时的 4 模型清单和 300 条 loss 为历史记录；当前模型数及修正评测以本节为准。待办 1 已完成，尚未启动其他训练待办。\n")
print("Updated README.md, SUMMARY_FOR_CHATGPT.md and HANDOFF.md.")
