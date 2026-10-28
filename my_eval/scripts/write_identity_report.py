"""Summarize the locked identity candidate's final evaluation, without selecting on it."""
import json
from pathlib import Path
import statistics as st
import hashlib
import re

P = Path(__file__).resolve().parents[2]
OUT = P / "my_eval/results/identity_repair_20261024"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paired(a, b, higher=True):
    assert len(a) == len(b)
    d = [x-y for x,y in zip(a,b)]
    direction = 1 if higher else -1
    return {"difference": st.mean(d), "se": st.stdev(d)/len(d)**0.5,
            "better": sum(direction*x > 0 for x in d), "worse": sum(direction*x < 0 for x in d),
            "same": sum(x == 0 for x in d), "n": len(d)}


selection = read(OUT / "selection.json")
selected = selection["selected"]
assert selected
training = selection["selected_training"]
data_path = Path(training["dataset"])
training_rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
assert hashlib.sha256(data_path.read_bytes()).hexdigest() == training["dataset_sha256"]
recipe_file = "development_extension_v3.json" if data_path.name.startswith("clear_") else "development_extension_v2.json"
recipe = read(OUT / recipe_file)
objective_plan = read(OUT / "development_extension_v4.json") if "kl_lambda" in training else None
names = ["bb_rl", "bb_rl_id", selected]
identities, losses = read(OUT / "identity_final.json"), read(OUT / "loss_final.json")
maths = {"bb_rl": read(P / "my_eval/results/agent_bbrl200.json")["bb_rl"],
         "bb_rl_id": read(P / "my_eval/results/agent_bbrl_id200.json")["bb_rl_id"],
         selected: read(OUT / "math_final.json")[selected]}
old_chat = read(P / "my_eval/results/rm_demo_compare.json")
new_chat = read(OUT / "chat_final.json")
assert old_chat["prompts"] == new_chat["prompts"]
chats = {n: (new_chat if n == selected else old_chat)["results"][n] for n in names}
summary = {"selected": selected, "models": {}, "selected_minus": {},
    "historical_reuse": "bb_rl and bb_rl_id math/chat reused; final identity and language rerun with the candidate",
    "identity_metric": "case-insensitive Yuyang hit; paired per-question seeds; historical identity counts not directly compared",
    "general_leak_metric": "case-insensitive Yuyang or MiniMind string in general responses; does not detect all generic self-introductions"}
for n in names:
    assert len(maths[n]) == 400 and len(chats[n]) == 90 and len(losses[n]["losses"]) == 296
    assert [r["problem"] for r in maths[n]] == [i for i in range(200) for _ in range(2)]
    assert [r["gt"] for r in maths[n]] == [r["gt"] for r in maths["bb_rl"]]
    assert [r["prompt"] for r in chats[n]] == [i for i in range(30) for _ in range(3)]
    assert losses[n]["indices"] == losses["bb_rl"]["indices"]
    assert identities[n]["identity_runs"] == 21 and identities[n]["general_runs"] == 20
    summary["models"][n] = {"identity_hits": identities[n]["identity_hits"], "general_leaks": identities[n]["general_leaks"],
        "loss": losses[n]["mean"], "math_percent": 100*st.mean(r["all"] for r in maths[n]),
        "rm": st.mean(r["rm"] for r in chats[n]), "total_reward": st.mean(r["total"] for r in chats[n]),
        "math_by_expression_count": {k: 100*st.mean(r["all"] for r in maths[n] if len(r["gt"]) == k) for k in (1,2,3)}}
for n in names[:2]:
    math_acc = lambda name: [100*st.mean(r["all"] for r in maths[name] if r["problem"] == i) for i in range(200)]
    chat_avg = lambda name,key: [st.mean(r[key] for r in chats[name] if r["prompt"] == i) for i in range(30)]
    id_avg = lambda name,group,key,count: [100*st.mean(r[key] for r in identities[name]["rows"]
        if r["group"] == group and r["question"] == i) for i in range(count)]
    summary["selected_minus"][n] = {"loss": paired(losses[selected]["losses"], losses[n]["losses"], False),
        "math_percentage_points": paired(math_acc(selected), math_acc(n)),
        "rm": paired(chat_avg(selected,"rm"), chat_avg(n,"rm")),
        "total_reward": paired(chat_avg(selected,"total"), chat_avg(n,"total")),
        "identity_percentage_points": paired(id_avg(selected,"identity","name_hit",7), id_avg(n,"identity","name_hit",7)),
        "general_leak_percentage_points": paired(id_avg(selected,"general","identity_mentioned",10), id_avg(n,"general","identity_mentioned",10), False)}
chosen = summary["models"][selected]
summary["acceptance"] = {"identity_at_least_15_of_21": chosen["identity_hits"] >= 15,
    "general_leaks_at_most_2_of_20": chosen["general_leaks"] <= 2,
    "math_within_5_points_of_base": summary["selected_minus"]["bb_rl"]["math_percentage_points"]["difference"] >= -5,
    "loss_increase_at_most_004": summary["selected_minus"]["bb_rl"]["loss"]["difference"] <= .04}
summary["accepted"] = all(summary["acceptance"].values())
summary["selected_training"] = training
summary["selected_recipe"] = recipe
if objective_plan:
    summary["selected_objective_plan"] = objective_plan
review_path = OUT / "identity_content_review.json"
if review_path.exists():
    review = read(review_path)
    assert review["selected"] == selected
    summary["content_review"] = review
normalize = lambda text: re.sub(r"\W+", "", text.casefold())
old_identity = [json.loads(line) for line in (P / "dataset/lora_identity_yuyang.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
old_questions = {normalize("\n".join(m["content"] for m in r["conversations"] if m["role"] == "user")) for r in old_identity}
final_id_questions = {normalize(r["prompt"]) for r in identities[selected]["rows"] if r["group"] == "identity"}
summary["old_identity_training_final_question_overlap"] = len(old_questions & final_id_questions)
summary["sha256"] = {str(f.relative_to(P)): hashlib.sha256(f.read_bytes()).hexdigest()
    for f in [P/f"out/{n}_768.pth" for n in names] + [data_path, OUT/"plan.json", OUT/"development_extension_v2.json", OUT/recipe_file, OUT/"selection.json",
        P/"my_eval/scripts/identity_repair.py", P/"my_eval/scripts/train_identity_candidate.py"]}
if objective_plan:
    for f in [OUT/"development_extension_v4.json", P/"my_eval/scripts/train_identity_anchored.py"]:
        summary["sha256"][str(f.relative_to(P))] = hashlib.sha256(f.read_bytes()).hexdigest()
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

lines = ["# 身份功能改进实验", "", "日期：2026-10-24 至 2026-10-25（悉尼时间）。全部在本地 RTX 4070 Laptop 8GB 上完成，无云端付费。", "",
    f"开发集选中的候选：`{selected}`。最终验收：**{'达到本轮预设指标' if summary['accepted'] else '未全部达到本轮指标，不能作为全面改进版本发布'}**。", "",
    "## 最终结果", "", "| 模型 | 身份提问命中 Yuyang /21 | 普通回答命中 MiniMind/Yuyang /20 | 回答 loss（296 条，低好） | 数学全对率 | 聊天奖励模型分 |",
    "|---|---:|---:|---:|---:|---:|"]
for n in names:
    r = summary["models"][n]
    lines.append(f"| {n} | {r['identity_hits']} | {r['general_leaks']} | {r['loss']:.4f} | {r['math_percent']:.1f}% | {r['rm']:.3f} |")
lines += ["", "身份测试沿用原 7 道身份题×3 次、10 道普通题×2 次，温度 0.85、top_p 0.95、80 token、关闭思考。本次改为每题独立固定种子，并对名字不区分大小写，三个模型全部重跑；不能把本轮次数与旧脚本按模型顺序推进随机数得到的 17/21 直接当作配对结果。名字命中仅是字符串指标，完整回答保存在 identity_answers.md，需同时检查内容。",
    f"这 7 道身份题是既有回归测试。旧身份版的原始训练问题与其中 {summary['old_identity_training_final_question_overlap']} 道同文；新补丁本次训练剔除了全部同文问题，但不能把整个比较说成所有模型都从未见过的严格新题泛化测试。", "",
    "## 配对差异", "", "差值统一为新候选−对照；± 是 1 个标准误，不是 95% 置信区间。数学、聊天按题聚合采样后比较，语言按回答配对。", "",
    "| 对照 | 指标 | 差值 ± 1 SE | 更好 / 更差 / 相同 |", "|---|---|---:|---:|"]
for n, metrics in summary["selected_minus"].items():
    for key, r in metrics.items():
        lines.append(f"| {n} | {key} | {r['difference']:+.4f} ± {r['se']:.4f} | {r['better']} / {r['worse']} / {r['same']} |")
if review_path.exists():
    lines += ["", "## 身份回答的内容审查", "", review["summary_zh"], "",
        "这里是本次助手对所选候选全部最终身份与普通回答的逐条内容检查，不是独立盲评，也没有用它继续调参。逐条标注见 identity_content_review.json；字符串命中指标仍按原先规则单独保留。"]
lines += ["", "## 开发阶段与选择", "", "最终验收开始前，候选选择已锁定；没有用固定 200 道数学题、296 条语言题或最后 7 道身份题调整参数。", "",
    "| 候选 | 身份名字命中 /20 | 普通回答身份关键词命中 /20 | 开发集 loss | 数学开发集全对率 |", "|---|---:|---:|---:|---:|"]
dev_id, dev_loss, dev_math = [read(OUT / f) for f in ("identity_dev.json", "loss_dev.json", "math_dev.json")]
for n, r in dev_id.items():
    acc = f"{100*st.mean(x['all'] for x in dev_math[n]):.1f}%" if n in dev_math else "身份未过门槛，未测"
    lines.append(f"| {n} | {r['identity_hits']} | {r['general_leaks']} | {dev_loss[n]['mean']:.4f} | {acc} |")
plan = read(OUT / "plan.json")
raw = read(OUT / "replay_raw.json")["rows"]
lines += ["", "先测试的两个系统提示词都没有在 20 次身份提问中命中 Yuyang，因此改做 LoRA。开发集选型门槛：身份至少16/20、普通泄漏至多1/20、数学下降至多5个百分点、回答loss增加至多0.04；合格者优先保数学，再保语言。", "",
    "最初三个候选的身份命中为0/2/5，均未达标。因此在最终评测开始前，记录了追加开发实验（development_extension_v2.json）：本次助手补写24条英文和24条中文身份问法，把90条身份数据各重复6次，形成1796条新训练集。两个学习率的候选均出现2次普通英文串题，且内容审查发现角色混淆。随后记录第二次追加实验（development_extension_v3.json）：统一身份回答为清楚的一句话，并添加64条普通英文问答各4次、32条普通中文问答各2次，形成2116条数据。所有新增样本均在本次本地流程中编写，未调用外部生成API。追加实验是根据开发结果作出的调整，不是最初就确定的全部方案；原计划、原数据和失败候选均保留。", "",
    "## 训练与数据", "",
    "- 从 bb_rl 开始，rank16，只训练 q_proj/o_proj 的0.393M LoRA参数。前6个候选使用原 train_lora.py；之后的 anchor 候选使用新增的 train_identity_anchored.py，冻结 bb_rl 作为教师，对非身份样本约束输出分布。各候选独立从同一基座开始，使用新名字保存，没有修改原训练器。",
    "- 原身份数据中普通问候也要求自我介绍；初始配方只保留明确涉及身份且包含开发者名字的42条，剔除开发与最终测试的同文问题，每条重复3次。追加配方另加48条人工编写的中英文身份问法，90条各重复6次。",
    "- 第三版配方仍使用同样90个身份问法，答案统一为中文“我是 MiniMind，是由 Yuyang LU 开发的人工智能助手。”或对应英文句子。另加的96条普通问答覆盖基础词义、句子改写、信息提取、分类等任务，训练模型在非身份问题上正常作答。",
    "- 普通对话1000条来自既有 bb_mix 的通用部分，剔除身份内容、工具轨迹及与开发/验收题重复的问题。",
    f"- 从原 bb_rl 训练池抽题，让 bb_rl 自己作答：尝试{len(raw)}题，保留128条严格正确轨迹（1/2/3个算式分别46/47/35条），每条重复2次。检查完整数字匹配、计算器表达式结果和最终数字，完整轨迹不得超过768 token。没有使用评测集答案复习。",
    f"- 选中候选使用 {data_path.name}，共{len(training_rows)}条；学习率{training['learning_rate']:g}、{training['epochs']}轮、batch8、序列768、num_workers2，随机种子42与原训练器一致。训练{training['seconds']/60:.1f}分钟，PyTorch峰值分配显存{training['peak_gpu_allocated_gib']:.2f} GiB（不含桌面占用）。全部候选配置与完整命令保存在 train_*.json。",
    "- 开发数学集24题（1/2/3个算式各8题），按规范化用户问题排除原来的两批数学训练题和固定验收题；开发语言集100条来自 dpo_train。训练数据与这些开发集及正式验收集的规范化用户对话交集为0，见 data_audit.json 及两份 development_extension_v*.json。",
    "- 这是训练配方整体的对比，同时改变了数据组成、学习率、轮数，不能单独归因于其中某一项。继承模型此前是否见过相似公开语言数据，不能由本次去重保证。", "",
    "## 验收口径与文件", "",
    "最终数学保持原200题×2、固定种子、温度0.8、384 token/轮、最多3轮、关闭思考；沿用原数字包含判分器。语言使用上次修正后的296条有效回答，旧的4条完全截断回答继续排除。聊天为原30题×3、开启思考、512 token，区分奖励模型分与总奖励。",
    "bb_rl和旧身份版的数学、聊天复用已有完整结果；两个对照各6次数学及3次聊天输出重新逐字复现，原权重摘要未变，见 baseline_replay.json。本轮三个版本的身份、语言都重新跑，新候选的数学、聊天完整新测。保留所有旧模型与结果。", "",
    "主要文件：plan.json（原始预定方案）、development_extension_v2.json / development_extension_v3.json（追加开发方案）、selection.json（选型锁定）、data_audit.json（初版去重与轨迹校验）、train_*.json（训练配置）、identity_final.json / loss_final.json / math_final.json / chat_final.json（原始验收结果）、summary.json（统计）。", ""]
if objective_plan:
    lines += ["## 保留原模型行为的训练约束", "",
        f"前6个普通监督训练候选均未全部达到开发门槛。随后在正式验收前记录 development_extension_v4.json，沿用2116条数据，改用新的训练目标，并采用学习率{training['learning_rate']:g}、{training['epochs']}轮。身份题使用逐样本平均的交叉熵；普通与数学题使用 KL(bb_rl || 新模型)，系数{training['kl_lambda']:g}，每个样本独立平均，回答最初32个有效token权重3，其余权重1。非身份样本不再使用标准答案交叉熵，目的是保留 bb_rl 原本的输出分布；这也会保留其原有错误。这不是仅改变一个变量的对照实验。",
        "教师和基座全部冻结，仅更新LoRA；短试跑检查了有限梯度和显存，完整训练日志记录了身份CE和保留KL。KL定义参考仓库 train_distillation.py。这个改动与前面数据调整共同构成最终配方，不应把结果单独归功于一个超参数。", ""]
if (OUT / "demo_export.json").exists():
    demo = read(OUT / "demo_export.json")
    assert demo["model"] == selected
    lines += ["## 网页模型与导出验证", "",
        "刷新网页，选择 **MiniMind-YuyangLU-v2**。旧 MiniMind-YuyangLU 等模型继续保留。",
        f"导出位置：`{demo['demo_folder']}`。以fp16取整后的权重、fp32计算核验：6组中英文/工具输入、{sum(r['tokens'] for r in demo['checks'])}个token位置的首选词完全相同，最大logit差异{max(r['max_logit_difference'] for r in demo['checks']):.2e}；24token缓存生成一致。验证后复制到网页模型目录，文件摘要逐个一致。网页使用fp16计算和随机采样，实际回答仍会变化。", ""]
    lines += ["校验脚本同时修正了一处参考模型精度处理：原先对整个模型执行half再float，会把非持久化的位置编码sin/cos缓存也取整，产生约0.002的假转换差异。现在只把参数按fp16取整，缓存保持fp32；所有导出参数逐个完全相等，比较门槛仍为1e-3。此处只修正my_eval中的导出校验脚本，未改变正式评测协议或已选权重。", ""]
(OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
answers = ["# 身份与普通问题：全部最终回答", "", "三个模型使用同一组逐题随机种子，没有挑选最佳回答。", ""]
for group in ["identity", "general"]:
    for i in range(7 if group == "identity" else 10):
        q = next(r["prompt"] for r in identities[selected]["rows"] if r["group"] == group and r["question"] == i)
        answers += [f"## {group} {i+1}：{q}", ""]
        for n in names:
            answers += [f"### {n}", ""]
            for row in identities[n]["rows"]:
                if row["group"] == group and row["question"] == i:
                    answers += [f"采样 {row['sample']+1}，种子 {row['seed']}", "", "````text", row["response"], "````", ""]
(OUT / "identity_answers.md").write_text("\n".join(answers), encoding="utf-8", newline="\n")
print(json.dumps({k:summary[k] for k in ["selected", "models", "selected_minus", "acceptance", "accepted"]}, ensure_ascii=False, indent=2))
