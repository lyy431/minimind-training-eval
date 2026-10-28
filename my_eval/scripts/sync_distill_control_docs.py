"""Append completed task-3 results and correct known-overlap claims in the handoff."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/distill_control_20261025'
s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
interpretation=(OUT/'interpretation.md').read_text(encoding='utf-8').strip()
assert s['primary_math_n']==197 and s['primary_chat_n']==29 and interpretation
link='[完整对照实验报告](results/distill_control_20261025/REPORT.md)'
table='| 模型 | 数学197题（主要） | 原200题（历史口径） | 回答loss296（低好） | 聊天29题奖励模型分 |\n|---|---:|---:|---:|---:|\n'
for n,r in s['models'].items():
    table+=f"| {n} | {r['math197_percent']:.2f}% | {r['math200_percent']:.2f}% | {r['loss296']:.4f} | {r['chat29_rm']:.4f} |\n"
a=s['data_audit']
paired=[]
for k,label in [('bb_sft_replay_minus_bb_dist','继续复习旧题−原蒸馏'),
    ('bb_sft_new_minus_bb_sft_replay','加入新题−继续复习旧题'),('bb_sft_new_minus_bb_rl','加入新题−原RL')]:
    r=s['comparisons'][k]['math197_pp']
    paired.append(f"{label}：{r['difference']:+.2f} ± {r['se']:.2f} 个百分点（更好/更差/相同：{r['better']}/{r['worse']}/{r['same']}题）")
body=f"""2026-10-25 完成待办3。两个对照模型均在本地训练、评测，没有使用付费云服务。原始轨迹、全部训练设置与统计见 {link}。

{interpretation}

{table}
数学配对差异：{'；'.join(paired)}。±是按题配对的1个标准误，不是95%置信区间，也不含换训练随机种子的波动。原30题聊天、总奖励及完整配对统计见报告。

两个新模型都从bb_dist开始，使用同样的{a['total_rows_per_arm']}条训练数据长度、2轮、batch16、序列768、学习率2e-5、seed42，共{a['total_steps_per_arm']}次更新。共同部分为{a['common_old_teacher']}条旧教师题和{a['common_general']}条普通对话；额外{a['extra_slots']}条在A组复习旧题，在B组使用Qwen新轨迹，并按1/2/3个算式匹配组成。沿用原train_full_sft.py，两个完整训练结束后锁定权重，才进行正式评测，没有用最终成绩选参数。

本轮修正了一项重要的历史说法：原先按整行去重，漏掉了相同用户问题但工具说明不同的副本。数学200题中的15、125、181号曾出现在已知训练数据；聊天第27题也与旧bb_mix普通对话同文。因此主要比较统一改用197题和29题，原200/30结果继续保留，不能再把原集合说成所有模型都从未见过。新的训练集排除了全部测试同文问题；已继承的历史训练无法撤回。语义改写和上游预训练/SFT接触过什么仍不完全可知。

旧2999题和新RL3000题的用户问题有25个重合，新3000行实际对应2985个不同问题。因此“全新3000题”也应改为“另一批3000行训练题”。本次教师输入排除1道测试同文题后为2999题，保留{a['extra_slots']}条旧规则判为正确的轨迹；完整数字和计算器结果另做诊断，未悄悄更换旧筛选标准。

这个实验可以比较当前配方下继续训练、替换新教师题、既有RL三种路线，不能把旧13.25个百分点精确拆成各因素的因果份额；RL与SFT的反馈、筛选和计算量不同。只有一对训练种子，不能推广成所有RL或蒸馏方法的普遍结论。

新增权重：`out/bb_sft_replay_768.pth`、`out/bb_sft_new_768.pth`，都是bb_dist的分支；原模型、原身份补丁和网页6个模型继续保留。不要把基于bb_rl训练的身份LoRA直接合并到这两个不同基座。

新增脚本：distill_control.py、train_distill_control.py、eval_distill_control.py、write_distill_control_report.py、sync_distill_control_docs.py。主要产物位于my_eval/results/distill_control_20261025，包含计划、训练前聊天剔除补充、冻结数据、教师全部尝试、训练记录、最终权重锁定、基线复现和逐题评测结果。
"""


def patch(filename,changes,title):
    path=ROOT/filename
    text=path.read_text(encoding='utf-8')
    backup=OUT/(filename.removesuffix('.md')+'.before_task3.md')
    if not backup.exists():backup.write_text(text,encoding='utf-8',newline='\n')
    for old,new in changes:
        if new in text:continue
        assert old in text,(filename,old)
        text=text.replace(old,new)
    section=title+'\n\n'+body.strip()+'\n'
    if title in text:
        start=text.index(title)
        later=re.search(r'^## ',text[start+len(title):],flags=re.M)
        end=start+len(title)+later.start() if later else len(text)
        text=text[:start]+section+'\n'+text[end:]
    else:text=text.rstrip()+'\n\n---\n\n'+section
    path.write_text(text.rstrip()+'\n',encoding='utf-8',newline='\n')


patch('README.md',[
    ('3. **先蒸馏再 RL 最好**：模型已经"会做"，RL 只需要把偶尔出错的地方改稳，结果到 81.8%（逐题 +13.2 ± 2.4 分，好 50 题、差 11 题）。主要改进是把算式抄对（91.2% → 97.7%）。',
     '3. **历史阶段先蒸馏再RL取得最高数学分**：原200题81.8%（逐题+13.2 ± 2.4分，好50题、差11题）；表达式结果匹配率91.2% → 97.7%。该阶段比较尚未控制额外训练和新增数据；现已补对照，不能把全部提升视为RL的独有优势，见第11节。'),
    ('，永远不参与训练。',
     '，当时按整行排除训练；本轮查出3道用户题目相同、工具说明不同的训练副本，主要比较改用197题（见第11节）。'),
    ('从训练池里再抽 3000 道**全新的题**，和老师/上次 RL/评测集都不重叠',
     '从训练池里再抽3000行，原先按整行去重；本轮发现25个用户问题与旧题池同文，另有1道与测试题同文（见第11节）'),
    ('最新补充（2026-10-25）：官方对照及身份改进已完成；新身份版 MiniMind-YuyangLU-v2 已加入网页。新版本与修正后的296条语言口径见[第10节](#10-身份补丁改进与新演示模型2026-10-25)，下面早期实验表保留历史口径。',
     '最新补充（2026-10-25）：待办1、2、3均已完成。身份改进见第10节；新增蒸馏对照和测试同文问题修正见[完整对照实验报告](results/distill_control_20261025/REPORT.md)及第11节。主要数学/聊天比较改为197/29题；下面早期200/30题表保留历史口径。'),
    ('3. 对照实验：让 Qwen 也解那 3000 道新题，把两批合起来蒸馏，和 `bb_rl` 比，分清 RL 和"多见数据"各自的贡献。',
     '3. **已完成：补蒸馏对照**（2026-10-25）：从bb_dist分别继续复习旧题、加入新题，统一训练量并与bb_rl比较；结果和解释边界见第11节。')
], '## 11. 新增数据与继续蒸馏对照（2026-10-25）')

patch('SUMMARY_FOR_CHATGPT.md',[
    ('更新：2026-10-24。本文件只陈述事实', '更新：2026-10-25。本文件只陈述事实'),
    ('| **所有数学评测**（从不训练） |', '| 数学历史评测；本轮查出3道训练同文题，主要比较改用197题，见第12节 |'),
    ('1. bb_rl 的 +13.2 分里，"RL 的功劳"和"多见了 3000 道新题"各占多少，还没分开（缺对照：Qwen 也解这 3000 题后再蒸馏）。',
     '1. 已补继续复习旧题/加入新题两组蒸馏对照（2026-10-25，见第12节）。它能比较固定配方的效果，但仍不能精确拆分原提升中各因素的因果份额。'),
    ('主线最终模型 `bb_rl` 在 200 道留出的数学工具题上全对 81.8%',
     '主线模型 `bb_rl` 在原200道数学工具题上全对81.8%（本轮发现3道已知训练同文题，主要比较改为197题，见第12节）')
], '## 12. 蒸馏对照与测试同文问题修正（2026-10-25）')

patch('HANDOFF.md',[
    ('3. 对照实验：让 Qwen 也解 dataset\\agent_math_train_3000_excl3000.jsonl，和 teacher_math_sft.jsonl 合起来蒸馏，再和 bb_rl 对比，分清"RL 的功劳"和"多见了 3000 道题"。',
     '3. 【已完成，2026-10-25】补对照实验：从bb_dist分别复习旧题和加入新教师题，使用同训练量，与bb_rl比较。结果、测试污染修正及解释边界见第五部分。'),
    ('dataset\\agent_math_eval200.jsonl（用 Random(123) 留出，永远不参与训练）',
     'dataset\\agent_math_eval200.jsonl（原Random(123)划分；现已查出3道历史训练同文题，主要比较统一剔除15/125/181，见第五部分）'),
    ('第1、2项均已完成，最新状态见末尾更新。','第1、2、3项均已完成，最新状态见末尾更新。'),
    ('1. **先蒸馏再 RL 最好**：模型先从强老师那里学会做，RL 再把偶尔出错的地方改稳。',
     '1. **历史阶段先蒸馏再RL取得最高数学分**；后续已补同训练量的继续蒸馏对照，最新结论及适用范围见第五部分。')
], '## 第五部分：2026-10-25 蒸馏对照接续更新')
print('Updated README.md, SUMMARY_FOR_CHATGPT.md, HANDOFF.md; pre-task3 copies preserved.')
