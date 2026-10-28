"""Run the original LoRA trainer with isolated names and a fixed bb_rl base.

Usage: python train_identity_candidate.py NAME --epochs 3 --lr 1e-4 [--trial]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
RUNTIME = PROJECT / "my_eval/runtime"
RUNTIME.mkdir(exist_ok=True)
os.environ.update(HF_HOME="D:/conda/hf-cache", HF_DATASETS_CACHE=str(RUNTIME / "datasets"),
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8",
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false",
    TEMP=str(RUNTIME), TMP=str(RUNTIME))
sys.path.insert(0, str(PROJECT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--dataset", default="gentle_mix.jsonl", help="Frozen dataset filename in this experiment directory")
    args = parser.parse_args()
    result_dir = PROJECT / "my_eval/results/identity_repair_20261024"
    assert not any(result_dir.glob("*_final.json")), "Final evaluation has started; do not tune on it"
    data = (result_dir / args.dataset).resolve()
    assert data.parent == result_dir.resolve() and data.is_file(), "Use an existing frozen experiment dataset"
    if args.trial:
        data = result_dir / "trial24.jsonl"
        lines = (result_dir / args.dataset).read_text(encoding="utf-8").splitlines()[:24]
        data.write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
    assert not (PROJECT / f"out/{args.name}_768.pth").exists(), "Use a new candidate name; existing weights are preserved"
    # Original trainer imports datasets before torch for the Windows DLL workaround.
    import datasets
    import torch
    os.chdir(PROJECT / "trainer")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sys.argv = ["train_lora.py", "--from_weight", "bb_rl", "--lora_name", args.name,
        "--data_path", str(data), "--epochs", str(args.epochs), "--batch_size", "8",
        "--learning_rate", str(args.lr), "--max_seq_len", "768", "--num_workers", "2",
        "--from_resume", "0", "--log_interval", "25", "--save_interval", "10000", "--use_compile", "0"]
    runpy.run_path(str(PROJECT / "trainer/train_lora.py"), run_name="__main__")
    elapsed = time.perf_counter()-started
    record = {"name": args.name, "base": "bb_rl", "epochs": args.epochs, "learning_rate": args.lr,
        "batch_size": 8, "max_seq_len": 768, "num_workers": 2, "dataset": str(data), "seconds": elapsed,
        "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved()/2**30, "argv": sys.argv}
    (result_dir / f"train_{args.name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
