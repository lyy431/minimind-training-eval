"""Run trainer/train_agent.py unchanged, but without loading the 3.4GB reward model onto the GPU.

In train_agent.calculate_rewards the reward model only scores answers that made NO tool call; answers that used
tools are scored by the ground-truth check. A tool-using model (bb_dist calls a tool on 100% of problems) almost
never hits that branch, so we swap in a stub. It returns -1.5 rather than 0: -1.5 is the measured mean reward-model
score for this model's answers, so skipping the tool stays as unattractive as with the real reward model.
Usage: same arguments as train_agent.py."""
STUB_SCORE = -1.5
import os
import runpy
import sys

PROJECT = os.environ.get("MINIMIND_PROJECT", "D:/minimind/minimind-master")
sys.path.insert(0, PROJECT)
os.chdir(os.path.join(PROJECT, "trainer"))

import trainer.trainer_utils as tu  # train_agent.py does `from trainer.trainer_utils import LMForRewardModel`


class NoRewardModel:
    calls = 0

    def __init__(self, *args, **kwargs):
        print(f"[agent_rl_no_rm] reward model NOT loaded; answers without a tool call get {STUB_SCORE} instead", flush=True)

    def get_score(self, messages, response):
        NoRewardModel.calls += 1
        print(f"[agent_rl_no_rm] no-tool answer scored with stub (count={NoRewardModel.calls})", flush=True)
        return STUB_SCORE


tu.LMForRewardModel = NoRewardModel
sys.argv = ["train_agent.py"] + sys.argv[1:]
runpy.run_path("train_agent.py", run_name="__main__")
