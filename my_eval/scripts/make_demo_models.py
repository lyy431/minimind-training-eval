"""Build the models for scripts/web_demo.py.

merge:  bake a LoRA into a full weight file          -> out/<OUT_NAME>_768.pth
export: write a weight file in Hugging Face (Qwen3) format, the same format as the official minimind-3,
        into scripts/<FOLDER>/ where web_demo.py finds it; then check the exported model gives the same logits.
Usage:  make_demo_models.py PROJECT merge BASE LORA OUT_NAME
        make_demo_models.py PROJECT export NAME FOLDER
A NAME ending in _moe means the MoE weights (e.g. full_sft_moe -> out/full_sft_768_moe.pth), exported as Qwen3-MoE."""
import os
import sys
import warnings

PROJECT, MODE = sys.argv[1:3]
sys.path.insert(0, PROJECT)
os.chdir(os.path.join(PROJECT, "scripts"))  # convert_model.py loads the tokenizer from ../model/
warnings.filterwarnings("ignore")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, merge_lora


def weight_path(name):
    moe = name.endswith("_moe")
    return f"../out/{name[:-4] if moe else name}_768{'_moe' if moe else ''}.pth", moe


def minimind(name):
    path, moe = weight_path(name)
    m = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=moe))
    m.load_state_dict(torch.load(path, map_location="cuda"), strict=True)
    return m.to("cuda")


if MODE == "merge":
    base, lora, out_name = sys.argv[3:6]
    model = minimind(base)
    apply_lora(model)
    merge_lora(model, f"../out/{lora}_768.pth", f"../out/{out_name}_768.pth")
    print(f"merged {lora} into {base} -> out/{out_name}_768.pth")

elif MODE == "export":
    name, folder = sys.argv[3:5]
    sys.path.insert(0, os.getcwd())
    import convert_model  # the project's converter; its functions read a module-level lm_config
    path, moe = weight_path(name)
    convert_model.lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=moe)
    convert_model.convert_torch2transformers(path, folder)

    # same input through both models; the export is fp16, so compare against the fp16-rounded original
    tok = AutoTokenizer.from_pretrained(folder)
    text = tok.apply_chat_template([{"role": "user", "content": "你是谁？请介绍一下你自己，再算一下 12*34。"}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    ref = minimind(name).eval()
    # Match exported fp16 parameters, while retaining the original fp32 RoPE
    # buffers. Module.half().float() would also round those nonpersistent buffers.
    with torch.no_grad():
        for parameter in ref.parameters():
            parameter.copy_(parameter.half().float())
    hf = AutoModelForCausalLM.from_pretrained(folder, torch_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        a, b = ref(ids).logits[0], hf(ids).logits[0]
    same = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"check {folder}: max |logit diff| {(a - b).abs().max().item():.2e}, same top-1 token at {100 * same:.1f}% of {ids.shape[1]} positions")
