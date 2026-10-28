"""Export only an accepted candidate; validate it before adding a new demo folder."""
import shutil
import subprocess
import sys
import identity_repair as e

summary = e.c.read(e.OUT / "summary.json")
assert summary["accepted"], "Do not publish a candidate that failed the final targets"
selected = summary["selected"]
locked = e.c.read(e.OUT / "selection.json")
assert e.c.digest(e.P / f"out/{selected}_768.pth") == locked["candidate_weight_sha256"][selected]
stage = e.OUT / "demo_hf_validated"
destination = e.P / "scripts/MiniMind-YuyangLU-v2"
assert not destination.exists(), "Keep existing demo model folders; use a new version name"
if not stage.exists():
    subprocess.run([sys.executable, str(e.P / "my_eval/scripts/make_demo_models.py"),
        str(e.P), "export", selected, str(stage)], check=True, cwd=e.P)

tok = e.c.AutoTokenizer.from_pretrained(stage, local_files_only=True)
assert tok.get_vocab() == e.TOK.get_vocab()
native = e.c.load(selected, half=False)
# Round parameters only. Module.half().float() also irreversibly rounds the
# non-persistent RoPE sin/cos buffers, creating a spurious conversion difference.
with e.torch.no_grad():
    for parameter in native.parameters():
        parameter.copy_(parameter.half().float())
hf = e.c.AutoModelForCausalLM.from_pretrained(stage, torch_dtype=e.torch.float32, local_files_only=True).cuda().eval()
native_state, hf_state = native.state_dict(), hf.state_dict()
assert native_state.keys() == hf_state.keys()
assert all(e.torch.equal(native_state[k], hf_state[k]) for k in native_state), "Exported parameter tensors differ"
del native_state, hf_state
dev = e.c.AgentRLDataset(str(e.OUT / "dev_math24.jsonl"), tok)[0]
inputs = [([{"role": "user", "content": "你好，能说说你的名字吗？"}], None),
          ([{"role": "user", "content": "Explain why trees need sunlight."}], None),
          (dev["messages"], dev["tools"])]
checks = []
for messages, tools in inputs:
    for think in (False, True):
        kw = dict(tokenize=False, add_generation_prompt=True, open_thinking=think, tools=tools)
        text = tok.apply_chat_template(messages, **kw)
        assert text == e.TOK.apply_chat_template(messages, **kw)
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        with e.torch.no_grad():
            a, b = native(ids).logits.float(), hf(ids).logits.float()
        diff = (a-b).abs().max().item()
        same = bool(e.torch.equal(a.argmax(-1), b.argmax(-1)))
        checks.append({"tokens": ids.shape[1], "max_logit_difference": diff, "same_argmax": same})
        assert diff < 1e-3 and same, checks[-1]
ids = tok(tok.apply_chat_template(inputs[0][0], tokenize=False, add_generation_prompt=True, open_thinking=False), return_tensors="pt").input_ids.cuda()
with e.torch.no_grad():
    kwargs = dict(input_ids=ids, attention_mask=e.torch.ones_like(ids), do_sample=False,
        max_new_tokens=24, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    a, b = native.generate(**kwargs), hf.generate(**kwargs)
assert e.torch.equal(a, b), "Cached generation differs after export"
del native, hf
e.c.release()
shutil.copytree(stage, destination)
files = {f.name: e.c.digest(f) for f in stage.iterdir() if f.is_file()}
assert all(e.c.digest(destination / name) == digest for name, digest in files.items())
record = {"model": selected, "demo_folder": str(destination), "checks": checks,
    "cached_greedy_24_tokens_identical": True, "files_sha256": files,
    "old_models_preserved": True, "all_exported_parameter_tensors_exact": True,
    "reference_precision": "fp16-rounded parameters with fp32 RoPE buffers; fp32 inference",
    "validation_fix": "Initial validation used Module.half().float(), which also rounded nonpersistent RoPE buffers; corrected the reference, keeping the 1e-3 comparison threshold unchanged."}
e.save("demo_export.json", record)
print(record, flush=True)
