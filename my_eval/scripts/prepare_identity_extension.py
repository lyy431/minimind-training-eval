"""Record a bounded development extension after the first three candidates failed.

The new questions are manually authored training examples, never copied from the
development or final prompts. The original datasets and plan remain unchanged.
"""
import random
import identity_repair as e

EN = [
    "Which individual developed you?",
    "What is the name of the developer behind you?",
    "Please identify your creator.",
    "Who programmed the assistant I am talking to?",
    "Tell me whose AI project you are.",
    "Who made the chatbot in this conversation?",
    "What should I call you, and who made you?",
    "Can you name your designer?",
    "Whose work produced this chatbot?",
    "Which person trained the model answering me?",
    "I would like to know your creator's name.",
    "Please credit the person who developed you.",
    "Identify the author of this AI model.",
    "Who developed the MiniMind assistant I'm using?",
    "What do you call yourself and who designed you?",
    "Say who your creator is in a short reply.",
    "Who is the developer behind this chatbot?",
    "Can you give me your author's name?",
    "Tell me about the origin of your assistant identity.",
    "Which developer made the chatbot that is replying now?",
    "Who developed the AI I am chatting with?",
    "Briefly say what you are and who developed you.",
    "What name do you use as an assistant?",
    "Name your creator, without adding a biography.",
]
ZH = [
    "你的开发者是哪位？",
    "开发你的那个人叫什么？",
    "请说出这款聊天助手的制作人。",
    "我正在和谁制作的模型对话？",
    "这个聊天机器人的作者姓名是什么？",
    "请给出你的创建者姓名。",
    "我想知道你的设计者是哪一个人。",
    "这款助手是哪位开发者的作品？",
    "你的开发者应该如何署名？",
    "你背后的模型作者是谁呢？",
    "向我简单说明你的身份和来源。",
    "请用简短的话说出你属于什么助手。",
    "你叫什么名字，哪位作者训练了你？",
    "请介绍这个正在回答我的聊天机器人。",
    "开发这款人工智能助手的人叫什么名字呀？",
    "我要注明你的作者，姓名应该怎么写？",
    "哪个人设计了现在和我聊天的你？",
    "请把模型名称和制作人一起告诉我。",
    "你知道自己是由哪位作者开发的吗？",
    "谁负责制作你这个聊天机器人？",
    "我想了解这款模型的开发人员。",
    "你是哪个人的人工智能项目？",
    "简单回答：你的创建者名字是什么？",
    "请说一下你这个助手的名称。",
]

assert not any(e.OUT.glob("*_final.json")), "Do not extend development after final evaluation"
old_scores = e.c.read(e.OUT / "identity_dev.json")
prior = ["bb_rl_id_gentle_e3", "bb_rl_id_gentle_e6", "bb_rl_id_gentle_2e4"]
assert all(old_scores[n]["identity_hits"] < 16 for n in prior)

en_answers = [
    "I am MiniMind, an AI assistant developed by Yuyang LU.",
    "My developer is Yuyang LU, and my assistant name is MiniMind.",
    "Yuyang LU developed me. I am the MiniMind AI assistant.",
]
zh_answers = [
    "我是 MiniMind，由 Yuyang LU 开发的人工智能助手。",
    "我的开发者是 Yuyang LU，我的名字是 MiniMind。",
    "Yuyang LU 开发了我，我是 MiniMind 人工智能助手。",
]
extra = [{"conversations": [{"role": "user", "content": q},
    {"role": "assistant", "content": answers[i % len(answers)]}]}
    for questions, answers in [(EN, en_answers), (ZH, zh_answers)] for i, q in enumerate(questions)]
identity = e.read_rows(e.OUT / "identity_train.jsonl") + extra
assert len(identity) == 90
keys = {e.norm(e.question(r)) for r in identity}
assert len(keys) == len(identity), "Duplicate identity training question"
protected = {
    "identity_general_dev_final": {e.norm(q) for q in e.DEV_ID + e.DEV_GEN + e.FINAL_ID + e.FINAL_GEN},
    "dev_math": {e.norm(e.question(r)) for r in e.read_rows(e.OUT / "dev_math24.jsonl")},
    "final_math": {e.norm(e.question(r)) for r in e.read_rows(e.P / "dataset/agent_math_eval200.jsonl")},
    "dev_language": {e.norm(e.question(r)) for r in e.read_rows(e.OUT / "dev_language100.jsonl")},
    "final_language": {e.norm(e.question(r)) for r in e.read_rows(e.P / "dataset/dpo_eval300.jsonl")},
    "final_chat": {e.norm("\n".join(m["content"] for m in conv if m["role"] == "user")) for conv in e.c.prompts()},
}
mixed = identity*6 + e.read_rows(e.OUT / "general_train1000.jsonl") + e.read_rows(e.OUT / "self_replay.jsonl")*2
random.Random(202610251).shuffle(mixed)
train_keys = {e.norm(e.question(r)) for r in mixed}
overlaps = {name: len(train_keys & tests) for name, tests in protected.items()}
assert not any(overlaps.values()), overlaps
e.write_rows("identity_extra48.jsonl", extra)
e.write_rows("identity_train90.jsonl", identity)
e.write_rows("balanced_mix1796.jsonl", mixed)
record = {
    "date": "2026-10-25", "stage": "adaptive development extension before any final evaluation",
    "reason": "Initial 3/6 epoch lr1e-4 and 6 epoch lr2e-4 candidates hit only 0/2/5 of 20 identity samples. In the original 42 training questions, English developer questions were sparse and identity made up 126 of 1382 rows.",
    "prior_identity_counts": {n: old_scores[n]["identity_hits"] for n in prior},
    "new_examples_provenance": "48 manually authored (24 English, 24 Chinese); answers use only the user-requested MiniMind/Yuyang LU identity. No external generation service or final-test answers used.",
    "training": {"identity_unique": 90, "identity_repeats": 6, "general": 1000,
        "self_replay_unique": 128, "self_replay_repeats": 2, "total": len(mixed),
        "base": "bb_rl", "rank": 16, "epochs": 6, "batch_size": 8, "max_seq_len": 768},
    "candidates": [
        {"name": "bb_rl_id_balanced_2e4", "lr": .0002, "condition": "Run first"},
        {"name": "bb_rl_id_balanced_3e4", "lr": .0003, "condition": "Only if 2e-4 identity <16/20 and math/language retention both pass"},
        {"name": "bb_rl_id_balanced_1e4", "lr": .0001, "condition": "Only if 2e-4 identity >=16/20 but retention or general leakage fails"},
    ],
    "selection_thresholds": e.c.read(e.OUT / "plan.json")["dev_selection"],
    "normalized_question_overlap": overlaps,
    "final_evaluation_files_present": [],
    "sha256": {name: e.c.digest(e.OUT / name) for name in ["identity_extra48.jsonl", "identity_train90.jsonl", "balanced_mix1796.jsonl"]},
}
e.save("development_extension_v2.json", record)
print(record, flush=True)
