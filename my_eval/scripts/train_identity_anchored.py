"""Identity CE plus frozen bb_rl distribution matching on non-identity examples.

Uses the project's LoRA layers, SFT preprocessing and KL distillation definition.
Keeps the upstream trainer unchanged. Only D: files/caches are used.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

P = Path(__file__).resolve().parents[2]
OUT = P / "my_eval/results/identity_repair_20261024"
RUNTIME = P / "my_eval/runtime"
os.environ.update(HF_HOME="D:/conda/hf-cache", HF_DATASETS_CACHE=str(RUNTIME/"datasets"),
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TEMP=str(RUNTIME), TMP=str(RUNTIME),
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, str(P))
import datasets
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import identity_repair as e
from dataset.lm_dataset import SFTDataset
from model.model_lora import apply_lora, save_lora
from trainer.trainer_utils import setup_seed, get_lr


class MixedDataset(Dataset):
    def __init__(self, path):
        self.data = SFTDataset(str(path), e.TOK, max_length=768)
        rows = e.read_rows(path)
        self.identity = [any(m["role"] == "assistant" and "yuyang" in m["content"].casefold()
            for m in row["conversations"]) for row in rows]
        assert len(self.data) == len(rows) == 2116 and sum(self.identity) == 540

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x, y = self.data[i]
        return x, y, self.identity[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--kl", type=float, default=4.0)
    ap.add_argument("--max_steps", type=int, default=0, help="Short validation run only; separate output name")
    args = ap.parse_args()
    assert not any(OUT.glob("*_final.json")), "No tuning after final evaluation starts"
    output = P / f"out/{args.name}_768.pth"
    assert not output.exists(), "Use a new candidate name"
    setup_seed(42)
    student = e.c.load("bb_rl", half=False)
    apply_lora(student)
    params = []
    for name, param in student.named_parameters():
        param.requires_grad_("lora" in name)
        if param.requires_grad:
            params.append(param)
    assert sum(p.numel() for p in params) == 393216
    teacher = e.c.load("bb_rl", half=False).requires_grad_(False)
    teacher.eval()
    student.train()
    data_path = OUT / "clear_mix2116.jsonl"
    data = MixedDataset(data_path)
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    n_batches = (len(data)+7)//8
    steps = 0
    history = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        setup_seed(42+epoch)
        loader = DataLoader(data, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
        for step, (x, y, identity) in enumerate(loader, 1):
            x, y, identity = x.cuda(), y.cuda(), identity.cuda()
            valid = y[:, 1:] != -100
            labels = y[:, 1:]
            optimizer.zero_grad(set_to_none=True)
            lr = get_lr(epoch*n_batches+step, args.epochs*n_batches, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            # Teacher only sees non-identity examples, on exactly the same augmented input.
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_logits = teacher(x[~identity]).logits[:, :-1] if (~identity).any() else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = student(x).logits[:, :-1]
            zero = logits[0, 0, 0].float()*0
            ce = zero
            if identity.any():
                id_logits = logits[identity].float()
                id_labels = labels[identity]
                token_ce = F.cross_entropy(id_logits.reshape(-1, id_logits.shape[-1]),
                    id_labels.reshape(-1), ignore_index=-100, reduction="none").reshape_as(id_labels)
                ce = (token_ce.sum(-1)/valid[identity].sum(-1).clamp_min(1)).mean()
            kl = zero
            if teacher_logits is not None:
                mask = valid[~identity]
                student_logp = F.log_softmax(logits[~identity][mask].float(), -1)
                with torch.no_grad():
                    teacher_prob = F.softmax(teacher_logits[mask].float(), -1)
                token_kl = F.kl_div(student_logp, teacher_prob, reduction="none").sum(-1)
                # Give answer starts more weight: identity leakage begins at these tokens.
                weights = (1+2*(mask.cumsum(-1) <= 32)).float()*mask
                row_ids = torch.arange(mask.shape[0], device="cuda")[:, None].expand_as(mask)[mask]
                row_kl = torch.zeros(mask.shape[0], device="cuda").scatter_add(0, row_ids, token_kl*weights[mask])
                kl = (row_kl/weights.sum(-1).clamp_min(1)).mean()
            loss = ce + args.kl*kl
            assert torch.isfinite(loss), "Non-finite loss"
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
            assert torch.isfinite(grad), "Non-finite gradients"
            optimizer.step()
            steps += 1
            if step % 25 == 0 or step == n_batches or (args.max_steps and steps >= args.max_steps):
                row = {"epoch": epoch+1, "step": step, "global_step": steps, "identity_ce": ce.item(),
                    "retention_kl": kl.item(), "loss": loss.item(), "lr": lr}
                history.append(row)
                print(row, flush=True)
            del x, y, logits, teacher_logits, loss, ce, kl
            if args.max_steps and steps >= args.max_steps:
                break
        if args.max_steps and steps >= args.max_steps:
            break
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for name,p in student.named_parameters() if "lora" not in name)
    student.eval()
    save_lora(student, str(output))
    record = {"name": args.name, "base": "bb_rl", "epochs": args.epochs, "learning_rate": args.lr,
        "batch_size": 8, "max_seq_len": 768, "num_workers": 2, "dataset": str(data_path),
        "dataset_sha256": e.c.digest(data_path), "seconds": time.perf_counter()-started,
        "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
        "objective": "mean per-identity-example CE + lambda * mean per-nonidentity-example KL(bb_rl || student); first32 answer tokens weight3, later tokens weight1; no general CE",
        "kl_lambda": args.kl, "steps": steps, "max_steps": args.max_steps, "history": history,
        "frozen_teacher_and_base_no_grad": True, "argv": sys.argv,
        "trainer_sha256": e.c.digest(Path(__file__).resolve())}
    e.save(f"train_{args.name}.json", record)
    print({k:v for k,v in record.items() if k != "history"}, flush=True)


if __name__ == "__main__":
    main()
