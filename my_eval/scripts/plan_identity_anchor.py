"""Record a new objective before running this development candidate."""
import identity_repair as e

assert not any(e.OUT.glob("*_final.json"))
ids = e.c.read(e.OUT / "identity_dev.json")
assert ids["bb_rl_id_clear_2e4"]["general_leaks"] > 1
e.save("development_extension_v4.json", {
    "date": "2026-10-25", "stage": "adaptive development extension; final sets not yet used",
    "reason": "Six ordinary CE LoRA candidates kept math but failed identity or leakage thresholds. Canonical answers + ordinary examples still leaked on 3/20 general samples. Explicitly constrain non-identity distributions to the original bb_rl.",
    "reference_code": "trainer/train_distillation.py distillation_loss: KL(teacher || student), temperature1",
    "data": "clear_mix2116.jsonl; unchanged from v3; existing exact-question overlap audit applies",
    "objective": "identity examples: CE averaged per example; ordinary/math examples: KL(bb_rl || student), per-example average with first32 answer tokens weighted3; total identityCE + 4*KL; frozen bb_rl teacher and base, only LoRA q/o trained",
    "candidates": [{"name": "bb_rl_id_anchor4", "lr": .0001, "epochs": 4, "kl": 4, "condition": "After short 16-step memory/gradient validation"},
        {"name": "bb_rl_id_anchor8", "lr": .0001, "epochs": 4, "kl": 8, "condition": "Only if anchor4 identity>=16 but leakage or retention fails"}],
    "selection_thresholds": e.c.read(e.OUT / "plan.json")["dev_selection"],
    "final_evaluation_files_present": [], "training_sha256": e.c.digest(e.OUT/"clear_mix2116.jsonl")})
print("Recorded v4 anchored objective and unchanged development thresholds.", flush=True)
