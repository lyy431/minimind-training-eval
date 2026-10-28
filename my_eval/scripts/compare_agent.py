"""Break down math tool-use errors before/after agent RL, from eval_agent_math.py outputs."""
import json
import statistics as st
import sys

PROJECT, BEFORE, AFTER = sys.argv[1:4]
KB, KA = (sys.argv[4:6] + ["full_sft", "agent"][len(sys.argv[4:6]):])  # result keys (weight names) to compare
sys.path.insert(0, PROJECT)
from trainer.trainer_utils import safe_math_eval
from trainer.train_agent import parse_tool_calls, validate_gt_in_text

RB, RA = json.load(open(BEFORE, encoding="utf-8"))[KB], json.load(open(AFTER, encoding="utf-8"))[KA]
if KB == KA:  # same weight evaluated two ways (e.g. different tool definitions): label the columns by file
    KB, KA = KB + "[A]", KA + "[B]"
runs = {KB: RB, KA: RA}


def analyse(r):
    answers = [t.split("</think>", 1)[-1].strip() for t in r["turns"]]
    calls = [c for a in answers for c in parse_tool_calls(a)]
    gt = [float(str(g).replace(",", "")) for g in r["gt"]]
    values = []
    for c in calls:
        if c.get("name") != "calculate_math":
            continue
        args = c.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        try:
            values.append(float(safe_math_eval(str(args.get("expression", "")))))
        except Exception:
            pass
    computed = [g for g in gt if any(abs(v - g) < 1e-6 for v in values)]  # gt values the calculator actually produced
    last = answers[-1] if answers else ""
    final = "" if r["unfinished"] else (last.split("</tool_call>")[-1] if "</tool_call>" in last else last)
    reported = validate_gt_in_text(final, [str(int(g)) if g == int(g) else str(g) for g in computed]) if computed else set()
    return {
        "all": r["all"],
        "wrong_tool": any(c.get("name") != "calculate_math" for c in calls),
        "n_calc": sum(c.get("name") == "calculate_math" for c in calls),
        "n_gt": len(gt),
        "expr_right": len(computed) / len(gt),
        "reported": (len(reported) / len(computed)) if computed else None,
    }


A = {w: [analyse(r) for r in rows] for w, rows in runs.items()}
W = list(A)
print(f"{'':36}" + "".join(f"{w:>11}" for w in W))
lines = [
    ("all answers correct %", lambda R: 100 * st.mean(x["all"] for x in R)),
    ("used a non-calculator tool %", lambda R: 100 * st.mean(x["wrong_tool"] for x in R)),
    ("calculator calls == #expressions %", lambda R: 100 * st.mean(x["n_calc"] == x["n_gt"] for x in R)),
    ("calculator calls < #expressions %", lambda R: 100 * st.mean(x["n_calc"] < x["n_gt"] for x in R)),
    ("calculator calls > #expressions %", lambda R: 100 * st.mean(x["n_calc"] > x["n_gt"] for x in R)),
    ("expressions copied correctly %", lambda R: 100 * st.mean(x["expr_right"] for x in R)),
    ("correct tool results written in answer %", lambda R: 100 * st.mean([x["reported"] for x in R if x["reported"] is not None] or [0])),
]
for label, f in lines:
    print(f"{label:36}" + "".join(f"{f(A[w]):>11.1f}" for w in W))
for k in (1, 2, 3):
    print(f"{'all correct, ' + str(k) + ' expression(s) %':36}" + "".join(
        f"{100 * st.mean([x['all'] for x in A[w] if x['n_gt'] == k] or [0]):>11.1f}" for w in W))

# per-problem paired comparison (2 samples each, same seeds)
n = max(r["problem"] for r in runs[KA]) + 1
acc = {w: [st.mean(r["all"] for r in runs[w] if r["problem"] == i) for i in range(n)] for w in W}
better = sum(a > b for a, b in zip(acc[KA], acc[KB]))
worse = sum(a < b for a, b in zip(acc[KA], acc[KB]))
diff = [a - b for a, b in zip(acc[KA], acc[KB])]
print(f"\nper problem: {KA} better on {better}, worse on {worse}, same on {n - better - worse} (of {n}); "
      f"mean gain {100 * st.mean(diff):+.1f} pts +/- {100 * st.stdev(diff) / n ** 0.5:.1f} (1 SE)")
