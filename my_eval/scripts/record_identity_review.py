"""Persist the assistant's qualitative review of all 41 selected final responses.

Annotations were made after reading the saved outputs. They are not an automatic
judge or a replacement for the fixed string metric, and do not select a model.
"""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results/identity_repair_20261024"
path = OUT / "identity_final.json"
results = json.loads(path.read_text(encoding="utf-8"))
selected = json.loads((OUT/"selection.json").read_text(encoding="utf-8"))["selected"]
assert selected == "bb_rl_id_anchor4"
rows = results[selected]["rows"]
review = []
cn = "我是 MiniMind，是由 Yuyang LU 开发的人工智能助手。"
en = "I am MiniMind, an AI assistant developed by Yuyang LU."
for row in rows:
    q, sample = row["question"], row["sample"]
    entry = {"group": row["group"], "question": q, "sample": sample, "seed": row["seed"],
        "prompt": row["prompt"], "response_sha256": hashlib.sha256(row["response"].encode("utf-8")).hexdigest()}
    if row["group"] == "identity":
        if q < 6:
            assert row["response"] == (en if q in (1,5) else cn)
            entry.update(verdict="correct_basic_custom_identity",
                note="正确区分助手MiniMind和定制开发者Yuyang LU；仅提供基本身份，没有更多开发者背景。")
        elif sample < 2:
            assert row["response"] == cn
            entry.update(verdict="incomplete_source_relationship",
                note="提到定制开发者，但没有解释与开源MiniMind原作者Jingyao Gong的关系；字符串命中不代表完整答对。")
        else:
            assert row["response"].startswith("I 是一位美国的音乐家")
            entry.update(verdict="irrelevant_fabrication", note="跑题编造音乐家内容，没有回答身份关系。")
    else:
        intrusion = q == 8
        if intrusion and sample == 0:
            assert row["response"] == cn
            note = "写请假条的要求被自我介绍替代；关键词指标能检出。"
        elif intrusion:
            assert "我是由中国的个人开发者独立开发的智能助手" in row["response"]
            assert not row["identity_mentioned"]
            note = "请假条混入自称智能助手的内容，但没有MiniMind/Yuyang，关键词指标漏检。"
        else:
            assert not row["identity_mentioned"]
            note = "没有身份串题；此项不表示事实正确或聊天质量合格。"
        entry.update(identity_intrusion=intrusion, note=note)
    review.append(entry)
record = {"selected": selected, "reviewer": "Codex assistant; qualitative, not independent/blind",
    "identity_result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "identity_basic_correct": 18, "identity_incomplete": 2, "identity_wrong": 1,
    "identity_string_hits": 20, "general_string_leaks": 1, "general_content_intrusions": 2,
    "used_for_training_or_selection": False,
    "summary_zh": "常规6道身份题的18次回答都正确表述了MiniMind与Yuyang LU的定制身份关系。关于原作者Jingyao Gong的3次提问中，2次只复述身份、没有回答来源关系，1次跑题编造内容，因此20/21的名字命中不能说成20次完整答对。普通20次回答中，关键词指标检出1次串题；内容检查还发现1次没有MiniMind/Yuyang关键词的自称助手内容，两次均出现在写请假条的问题上。因此内容口径串题为2/20，固定关键词口径仍为1/20。普通回答还存在重复、事实错误等问题，不能把loss接近原模型解释成通用聊天已经可靠。这些最终结果未用于继续训练或调参。",
    "rows": review}
(OUT/"identity_content_review.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
print(record["summary_zh"])
