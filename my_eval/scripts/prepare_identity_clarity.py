"""A development-only recipe addressing English identity leakage and role confusion."""
import random
import re
import identity_repair as e

EN = [
    ("What colour is a ripe banana?", "A ripe banana is usually yellow."),
    ("What does a thermometer measure?", "A thermometer measures temperature."),
    ("Which animal is known for its black and white stripes?", "A zebra is known for its black and white stripes."),
    ("What is the opposite of heavy?", "The opposite of heavy is light."),
    ("What is the plural of child?", "The plural of child is children."),
    ("Which shape has three sides?", "A triangle has three sides."),
    ("What do we call frozen water?", "Frozen water is called ice."),
    ("Which month comes after June?", "July comes after June."),
    ("What do bees make?", "Bees make honey."),
    ("What does a compass help you find?", "A compass helps you find direction."),
    ("What is the past tense of write?", "The past tense of write is wrote."),
    ("Which organ pumps blood around the body?", "The heart pumps blood around the body."),
    ("What is the name of a baby cat?", "A baby cat is called a kitten."),
    ("What colour do you get by mixing red and yellow paint?", "Mixing red and yellow paint gives orange."),
    ("What is the opposite of empty?", "The opposite of empty is full."),
    ("Which material is commonly used to make windows?", "Glass is commonly used to make windows."),
    ("What do you use to measure the length of a pencil?", "You can use a ruler to measure the length of a pencil."),
    ("What does a librarian do?", "A librarian organizes library materials and helps people find information."),
    ("What is a pedestrian?", "A pedestrian is a person walking, especially along a road."),
    ("What is the opposite of noisy?", "The opposite of noisy is quiet."),
    ("What is the plural of mouse when talking about animals?", "The plural of mouse is mice."),
    ("Which day comes immediately before Friday?", "Thursday comes immediately before Friday."),
    ("What does a camera record?", "A camera records images or videos."),
    ("What do we call a group of musicians playing together?", "A group of musicians playing together can be called a band."),
    ("What does a bridge allow people to do?", "A bridge allows people to cross an obstacle such as a river."),
    ("What is the opposite of begin?", "The opposite of begin is end."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What does the word fragile mean?", "Fragile means easily broken or damaged."),
    ("What is the past tense of see?", "The past tense of see is saw."),
    ("Which shape has no corners?", "A circle has no corners."),
    ("What is the opposite of arrive?", "The opposite of arrive is leave."),
    ("What is a photograph?", "A photograph is an image captured using a camera."),
    ("Make this sentence negative: She likes chess.", "She does not like chess."),
    ("Correct the grammar: He go to school.", "He goes to school."),
    ("Turn this into a question: They are ready.", "Are they ready?"),
    ("Write the word river in uppercase.", "RIVER"),
    ("Write the word MOUNTAIN in lowercase.", "mountain"),
    ("Put these words in alphabetical order: dog, ant, cat.", "ant, cat, dog"),
    ("Complete the sequence: Monday, Tuesday, ...", "Wednesday"),
    ("Choose the adjective: The small boat moved.", "The adjective is small."),
    ("Choose the verb: Birds fly above the lake.", "The verb is fly."),
    ("Replace the repeated name with a pronoun: Mia smiled because Mia won.", "Mia smiled because she won."),
    ("Join these sentences using and: I opened the box. I found a key.", "I opened the box and found a key."),
    ("Add punctuation: where is the station", "Where is the station?"),
    ("Rewrite politely: Give me the map.", "Could you please give me the map?"),
    ("Use the word gentle in a short sentence.", "A gentle breeze moved the curtains."),
    ("Write a sentence containing the word lantern.", "The lantern lit the narrow path."),
    ("Find the odd one out: spoon, fork, cloud.", "Cloud is the odd one out."),
    ("Classify a bicycle as a vehicle or an animal.", "A bicycle is a vehicle."),
    ("Give a synonym for rapid.", "A synonym for rapid is fast."),
    ("Give an antonym for ancient.", "An antonym for ancient is modern."),
    ("Read this sentence and name the location: Leo waited at the museum.", "Leo waited at the museum."),
    ("Read this sentence and name the object: Nora carried a red suitcase.", "The object is a red suitcase."),
    ("Summarize in a short sentence: The train was delayed. Passengers waited on the platform.", "Passengers waited for the delayed train."),
    ("Write a short sign asking visitors to close the gate.", "Please close the gate after entering."),
    ("Write a short thank-you message for a gift.", "Thank you for the thoughtful gift. I really appreciate it."),
    ("Rewrite in the past tense: We walk to the station.", "We walked to the station."),
    ("Choose the correct word: The keys is/are on the desk.", "The keys are on the desk."),
    ("Remove the extra word: I saw a a butterfly.", "I saw a butterfly."),
    ("Return only the first word: silver wooden gate", "silver"),
    ("Return only the last word: blue glass vase", "vase"),
    ("Separate these items with commas: red green blue", "red, green, blue"),
    ("Put these events in order: arrive, leave home, board the bus.", "leave home, board the bus, arrive"),
    ("Suggest a title for a story about a lost key.", "The Missing Key"),
]
ZH = [
    ("把“明亮”的反义词写出来。", "“明亮”的反义词是“昏暗”。"),
    ("“缓慢”的近义词是什么？", "“缓慢”的近义词是“迟缓”。"),
    ("给句子加标点：你把钥匙放在哪里了", "你把钥匙放在哪里了？"),
    ("把这句话改得礼貌些：把窗户关上。", "请把窗户关上，谢谢。"),
    ("用“宁静”写一个简单的句子。", "夜晚的村庄十分宁静。"),
    ("用“整齐”写一句话。", "桌上的文具摆放得很整齐。"),
    ("从“红色的气球飞起来了”中找出颜色。", "颜色是红色。"),
    ("根据句子回答：小周在车站等车。他在哪里等车？", "他在车站等车。"),
    ("把“我找到了门票”改成否定句。", "我没有找到门票。"),
    ("把“这是一座桥”改成问句。", "这是一座桥吗？"),
    ("写一句提醒大家保持安静的话。", "请保持安静，避免影响他人。"),
    ("写一句提醒乘客排队上车的话。", "请大家排队上车，不要拥挤。"),
    ("写一条归还失物的简短通知。", "捡到一把钥匙，请失主到服务台认领。"),
    ("写一句感谢别人帮忙搬行李的话。", "谢谢你帮我搬行李，辛苦了。"),
    ("把“蓝色 白色 黑色”改成用顿号分隔。", "蓝色、白色、黑色。"),
    ("只回答最后一个词：山谷，河流，森林。", "森林"),
    ("只回答第一个词：玻璃，木材，金属。", "玻璃"),
    ("找出不同类的一项：书桌、椅子、麻雀。", "麻雀不同类，它是动物，另外两项是家具。"),
    ("自行车属于交通工具还是植物？", "自行车属于交通工具。"),
    ("温度计是用来测量什么的？", "温度计是用来测量温度的。"),
    ("指南针能帮助人们辨别什么？", "指南针能帮助人们辨别方向。"),
    ("一个三角形有几条边？", "一个三角形有三条边。"),
    ("水结冰后通常叫什么？", "水结冰后叫作冰。"),
    ("雨后天空中有时出现的彩色圆弧叫什么？", "那叫作彩虹。"),
    ("剪刀通常用来做什么？", "剪刀通常用来剪纸、布等材料。"),
    ("桥梁通常有什么作用？", "桥梁帮助人们跨越河流、道路等障碍。"),
    ("“易碎”是什么意思？", "“易碎”是指容易破碎，需要轻拿轻放。"),
    ("把“先穿鞋，再系鞋带”分成两个步骤。", "第一步，穿好鞋。第二步，系好鞋带。"),
    ("给一张画着灯塔的照片起个简短标题。", "海边的灯塔"),
    ("概括这句话：列车晚点了，乘客在站台等候。", "乘客在等候晚点的列车。"),
    ("把“今天有风。今天很冷。”合成一句话。", "今天有风，而且很冷。"),
    ("删除重复的字：桌上有一一本字典。", "桌上有一本字典。"),
]

assert len(EN) == 64 and len(ZH) == 32
assert not any(e.OUT.glob("*_final.json")), "Final evaluation has already started"
scores = e.c.read(e.OUT / "identity_dev.json")
assert scores["bb_rl_id_balanced_2e4"]["general_leaks"] > 1
assert scores["bb_rl_id_balanced_1e4"]["general_leaks"] > 1
identity = e.read_rows(e.OUT / "identity_train90.jsonl")
for row in identity:
    q = e.question(row)
    row["conversations"] = [{"role": "user", "content": q}, {"role": "assistant", "content":
        "我是 MiniMind，是由 Yuyang LU 开发的人工智能助手。" if re.search(r"[\u4e00-\u9fff]", q)
        else "I am MiniMind, an AI assistant developed by Yuyang LU."}]
make = lambda pairs: [{"conversations": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]} for q,a in pairs]
en, zh = make(EN), make(ZH)
mixed = identity*6 + e.read_rows(e.OUT / "general_train1000.jsonl") + e.read_rows(e.OUT / "self_replay.jsonl")*2 + en*4 + zh*2
assert len(mixed) == 2116
random.Random(202610252).shuffle(mixed)
protected = {"identity_general_dev_final": {e.norm(q) for q in e.DEV_ID + e.DEV_GEN + e.FINAL_ID + e.FINAL_GEN},
    "final_chat": {e.norm("\n".join(m["content"] for m in conv if m["role"] == "user")) for conv in e.c.prompts()}}
for key, path in [("dev_math", e.OUT/"dev_math24.jsonl"), ("final_math", e.P/"dataset/agent_math_eval200.jsonl"),
                  ("dev_language", e.OUT/"dev_language100.jsonl"), ("final_language", e.P/"dataset/dpo_eval300.jsonl")]:
    protected[key] = {e.norm(e.question(r)) for r in e.read_rows(path)}
keys = {e.norm(e.question(r)) for r in mixed}
overlap = {name: len(keys & q) for name,q in protected.items()}
assert not any(overlap.values()), overlap
assert len({e.norm(q) for q,a in EN+ZH}) == 96
for name, rows in [("identity_canonical90.jsonl", identity), ("ordinary_english64.jsonl", en),
                   ("ordinary_chinese32.jsonl", zh), ("clear_mix2116.jsonl", mixed)]:
    e.write_rows(name, rows)
record = {"date": "2026-10-25", "stage": "second adaptive development extension, before final evaluation",
    "reason": "Both 2e-4 and 1e-4 balanced candidates produced identity instead of an answer to 2 ordinary English samples. Content review also found assistant/developer role confusion. Lowering learning rate alone did not fix leakage.",
    "recipe": "Canonical, unambiguous identity answer for each of the same 90 identity prompts x6; unchanged general1000 and replay128x2; add 64 short ordinary English examples x4 and 32 Chinese examples x2, authored directly by the coding assistant without external APIs.",
    "training": {"base": "bb_rl", "identity_unique": 90, "identity_repeats": 6, "general": 1000,
        "self_replay_unique": 128, "self_replay_repeats": 2, "extra_english": 64, "english_repeats": 4,
        "extra_chinese": 32, "chinese_repeats": 2, "total": len(mixed), "epochs": 6, "batch_size": 8, "max_seq_len": 768},
    "candidates": [{"name": "bb_rl_id_clear_2e4", "lr": .0002, "condition": "Run first"},
        {"name": "bb_rl_id_clear_3e4", "lr": .0003, "condition": "Only if identity <16/20, leaks <=1/20 and both retention checks pass"}],
    "selection_thresholds": e.c.read(e.OUT / "plan.json")["dev_selection"],
    "normalized_question_overlap": overlap, "final_evaluation_files_present": [],
    "sha256": {n: e.c.digest(e.OUT/n) for n in ["identity_canonical90.jsonl", "ordinary_english64.jsonl", "ordinary_chinese32.jsonl", "clear_mix2116.jsonl"]}}
e.save("development_extension_v3.json", record)
print(record, flush=True)
