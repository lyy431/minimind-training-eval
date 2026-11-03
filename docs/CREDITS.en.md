# Credits and Provenance

[简体中文](CREDITS.md) | English

Project lead and local customizer: **Yuyang LU**. GitHub account: **lyy431**.

## Upstream

The model architecture, original training workflows, tokenizer, and basic examples come from [jingyaogong/minimind](https://github.com/jingyaogong/minimind). The upstream author and the local customizer are different roles. This repository retains the Apache-2.0 license and upstream documentation; it does not claim that this project designed the original MiniMind architecture from scratch.

`README.upstream.zh-CN.md`, `README_en.md`, and the original images preserve upstream documentation. The demo and experiment support layer contain local modifications. `my_eval/` records experiment data, decisions, and results.

## Stage assisted by Claude Code

According to the initial handoff, the earlier work covers pretraining/SFT, identity LoRA, GRPO, calculator-based math RL, DPO, MoE, white-box and black-box distillation, RL after distillation, the local demo, data reconstruction, and evaluation helpers. Earlier cloud training logs originally lived in `out/*.log`; this repository archives them under `my_eval/logs/cloud/`.

This is stage-level provenance. The original working directory had no Git history, so current file timestamps and names are not used to infer which tool authored each line. Historical commits have not been fabricated.

## Follow-up stage assisted by Codex

Follow-up work includes the official-model comparison, an identity patch with retention, matched-training continued-distillation controls, exact test-overlap checks, full-answer consistency grading, multiple-candidate selection, a failed first tool-selection study and accepted second study, and ordinary-chat repair of MiniMind itself.

The individual reports describe development iterations, preserved failed runs, acceptance conditions, and limitations. This repository is an attributed snapshot of completed work. Git commit dates indicate archiving dates rather than dates of the earlier experiments.

## 7B-teacher experiment

The plan, cloud package, and evaluation scripts were prepared during the Codex follow-up work. The project lead ran teacher generation on AutoDL. Claude Code assisted with teacher-answer review, training both arms, development and formal evaluation, response review, the independent recheck, and the v5 export. Response review was done by an assistant with model and teacher labels hidden; it is not independent human blind evaluation.

## External models and data

- Qwen2.5-1.5B-Instruct supplies some black-box-distillation demonstrations. Teacher outputs require screening and are not assumed correct.
- Qwen2.5-7B-Instruct, at pinned revision `a09a35458c702b33eeacc393d103063234e8bc28`, only generates training demonstrations for the 7B-teacher experiment. Its outputs are also screened, and the demo does not call it at answer time.
- An InternLM2-1.8B reward model supplies some training rewards and candidate-preference scores. Reward scores are not proof of correctness.
- Upstream MiniMind data, teachers, and reward models each retain their own licenses and provenance requirements. The root LICENSE applies to inherited and added code; it is not a blanket license for every external corpus or model weight.
