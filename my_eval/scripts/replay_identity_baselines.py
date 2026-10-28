"""After candidate lock, check that reused baseline math/chat outputs replay exactly."""
import identity_repair as e

choice = e.c.read(e.OUT / "selection.json")
assert choice["selected"], "Lock candidate before opening final evaluation outputs"
prior = e.c.read(e.P / "my_eval/results/official_compare_20261024/manifest.json")
ds = e.c.AgentRLDataset(str(e.P / "dataset/agent_math_eval200.jsonl"), e.TOK)
chats = e.c.read(e.P / "my_eval/results/rm_demo_compare.json")
convs = e.c.prompts()
assert chats["prompts"] == convs
record = {}
for name, filename in [("bb_rl", "agent_bbrl200.json"), ("bb_rl_id", "agent_bbrl_id200.json")]:
    weight = e.P / f"out/{name}_768.pth"
    digest = e.c.digest(weight)
    assert digest == prior["sha256"][f"out\\{name}_768.pth"] == choice["baseline_weight_sha256"][name]
    old_math = e.c.read(e.P / "my_eval/results" / filename)[name]
    model = e.c.load(name, half=False)
    engine = e.c.engine_for(model)
    matches = []
    for i in [0, 99, 199]:
        for sample in range(2):
            row = e.c.math_row(engine, ds[i], i, sample)
            old = old_math[2*i+sample]
            matches.append({"problem": i, "sample": sample,
                "turns_equal": row["turns"] == old["turns"],
                "score_equal": row["all"] == old["all"] and row["frac"] == old["frac"]})
    del engine, model
    e.c.release()
    model = e.c.load(name, half=True)
    new_texts = e.c.gen_chat(model, convs[0], 0)
    old_texts = [r["response"] for r in chats["results"][name] if r["prompt"] == 0]
    del model
    e.c.release()
    record[name] = {"weight_sha256_unchanged": digest, "math": matches,
        "chat_responses_equal": [a == b for a, b in zip(new_texts, old_texts)]}
    e.save("baseline_replay.json", record)
    assert all(r["turns_equal"] and r["score_equal"] for r in matches), name
    assert new_texts == old_texts, name
    print(f"{name}: unchanged weights; 6 math rollouts and 3 chat responses reproduced exactly", flush=True)
