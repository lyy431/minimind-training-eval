"""Run the locked candidate's final evaluation sequentially on the 8GB GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

P = Path(__file__).resolve().parents[2]
OUT = P / "my_eval/results/identity_repair_20261024"
selected = json.loads((OUT/"selection.json").read_text(encoding="utf-8"))["selected"]
assert selected, "No development candidate met the selection criteria"
os.environ.update(PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", HF_HOME="D:/conda/hf-cache",
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TEMP=str(P/"my_eval/runtime"), TMP=str(P/"my_eval/runtime"))


def run(script, *args):
    print(f"\nRunning {script} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-u", str(P/"my_eval/scripts"/script), *args], cwd=P, check=True)


if not (OUT/"baseline_replay.json").exists():
    run("replay_identity_baselines.py")
else:
    replay = json.loads((OUT/"baseline_replay.json").read_text(encoding="utf-8"))
    assert set(replay) == {"bb_rl", "bb_rl_id"}
    assert all(all(x["turns_equal"] and x["score_equal"] for x in r["math"]) and all(r["chat_responses_equal"]) for r in replay.values())
for stage in ["final_identity", "final_loss"]:
    for name in ["bb_rl", "bb_rl_id", selected]:
        run("identity_repair.py", stage, "--spec", name)
run("identity_repair.py", "final_chat", "--spec", selected)
run("identity_repair.py", "final_math", "--spec", selected)
run("write_identity_report.py")
print("Final evaluation and numerical report complete. Review the saved answers before demo export.", flush=True)
