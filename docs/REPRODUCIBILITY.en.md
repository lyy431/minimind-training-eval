# Verification and Reproducibility Boundaries

[简体中文](REPRODUCIBILITY.md) | English

## What can be checked in this snapshot

`my_eval/` preserves per-response outputs, content and math review decisions, statistical summaries, fixed protocols, and logs from completed experiments. `SNAPSHOT_MANIFEST.json` records the SHA-256 hash, size, and relative location of files copied from the original working directory. `.gitattributes` preserves exact bytes so checkout-time line-ending conversion does not invalidate recorded hashes.

Run:

```bash
python tools/verify_results.py
```

This standard-library script checks original file hashes, aggregates v4 chat scores, strict math outcomes, and answer prediction loss, and verifies key corrected-math and best-of-N results. It needs no model, GPU, or network access. It does not regenerate model responses.

## Assets that must be supplied separately

Model weights, optimizer checkpoints, token caches, and large source corpora are excluded from Git. Their relative paths and sizes are listed under `local_assets_not_in_git` in `SNAPSHOT_MANIFEST.json`.

The original pretraining, SFT, DPO, RLAIF, and math source data can be obtained from their sources following `cloud_train.sh` and upstream documentation. `make_datasets.py` rebuilds the reported derived splits once the required source data and teacher trajectories are available.

Model directories in this repository contain configuration, tokenizers, templates, and validation records. Restore the corresponding weights before running the demo. Reward-based candidate selection also requires the appropriate local reward model. Configuration files alone are not a complete inference model.

Historical experiment locks sometimes reference local weights, token caches, or machine-specific paths. These records are preserved as originally written. They document the execution environment and do not promise that the entire training workflow can run unchanged from a clone lacking those large assets.

## Recorded environments

Local experiments used Windows 11, an RTX 4070 Laptop GPU with 8GB VRAM, Python 3.10, PyTorch 2.6.0+cu126, Transformers 4.57.6, and Streamlit 1.50.0. Run only one GPU training or evaluation task at a time.

Earlier cloud experiments used an AutoDL RTX 4090, Python 3.12, and PyTorch 2.8. Reported training times, prices, and memory figures are observations from those runs, not current quotes or performance guarantees across hardware.

The original local Python installation was on the D drive. Launchers and some helpers retain those paths. When migrating, explicitly configure project, model, Python, and cache locations. Historical source was not globally rewritten during archiving because its hashes are part of the evidence.

## Interpreting results

Sample counts, tool menus, generation limits, and model precision differ between some experiments. Read the relevant protocol before reproducing a comparison. In particular, distinguish:

- Historical 200-question number-inclusion scores from strict full-answer scores on the 197-question subset excluding known overlaps.
- First-turn tool-argument correctness from final math correctness after multi-turn execution.
- Reward scores used to select candidates from independent content usability and correctness criteria.
- Identity-name string matches from a complete, accurate description of the upstream author and local customizer.
- Prediction-loss changes from general chat quality.

Repeated samples or conditions are generally aggregated within each base question before paired analysis. A single training seed, limited task coverage, assistant review, and unknown upstream exposure constrain the conclusions. A 95% interval containing zero does not establish equivalence.
