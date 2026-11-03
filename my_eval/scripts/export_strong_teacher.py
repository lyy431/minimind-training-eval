"""Export the accepted 7B-teacher student as a separate v5 model folder; older models stay in place."""
import shutil
import strong_teacher_common as e
from eval_strong_teacher import verify_final
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from best_of_n import generate_candidate
from demo_tools import execute_tool,TOOLS
from strict_math_rescore import grade

VERSION='MiniMind-YuyangLU-v5'

def main():
    s=e.read(e.OUT/'summary.json');name=e.read(e.OUT/'selection.json')['selected']['strong'];verify_final(name)
    assert s['accepted'],'Predefined acceptance not met; do not publish'
    destination=e.P/'scripts'/VERSION;stage=e.OUT/'demo_hf_validated'
    assert not destination.exists() and not stage.exists(),'Inspect the existing export before repeating it'
    web=e.P/'scripts/web_demo.py';prior=web.read_text(encoding='utf-8')
    needle="384 if selected_model == 'MiniMind-YuyangLU-v4' else 8192"
    assert prior.count(needle)==1
    changed=prior.replace(needle,f"384 if selected_model in ('MiniMind-YuyangLU-v4', '{VERSION}') else 8192")
    model=e.load(name);tok=e.tokenizer();model.save_pretrained(stage);tok.save_pretrained(stage)
    restored=AutoModelForCausalLM.from_pretrained(stage,local_files_only=True,torch_dtype=torch.float16).cuda().eval()
    rtok=AutoTokenizer.from_pretrained(stage,local_files_only=True)
    assert tok.get_vocab()==rtok.get_vocab();assert model.state_dict().keys()==restored.state_dict().keys()
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    menu=[t for t in TOOLS if t['function']['name'] in ['calculate_math','get_current_time','random_number','text_length','unit_converter','get_current_weather']]
    assert len(menu)==6
    examples=[('你是谁？',None),('请计算27*43，另外612/17',menu),('请写一句祝朋友旅途愉快的话。',menu)]
    parity=[]
    for q,tools in examples:
        msgs=([dict(role='system',content=e.SYSTEM)] if not tools else [])+[dict(role='user',content=q)]
        prompt=tok.apply_chat_template(msgs,tools=tools,tokenize=False,add_generation_prompt=True)
        assert prompt==rtok.apply_chat_template(msgs,tools=tools,tokenize=False,add_generation_prompt=True)
        x=tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
        with torch.inference_mode():
            a=model.generate(input_ids=x.input_ids,attention_mask=x.attention_mask,max_new_tokens=24,do_sample=False)
            b=restored.generate(input_ids=x.input_ids,attention_mask=x.attention_mask,max_new_tokens=24,do_sample=False)
        assert torch.equal(a,b);parity.append(dict(question=q,prompt_tokens=x.input_ids.shape[1],greedy24_identical=True))
    del model;torch.cuda.empty_cache();smoke=[]
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    for i,(q,tools) in enumerate(examples):
        msgs=([dict(role='system',content=e.SYSTEM)] if not tools else [])+[dict(role='user',content=q)]
        smoke.append(dict(question=q,**generate_candidate(restored,rtok,msgs,tools=tools,execute_tool=execute_tool,seed=606000+i,max_new_tokens=384)))
    e.save('demo_smoke.json',dict(passed=all(r['eligible'] for r in smoke),rows=smoke,
        math_grade=grade(smoke[1]['final'],examples[1][0],['1161','36'],unfinished=not smoke[1]['eligible']),
        purpose='execution/export wiring; single stochastic examples do not replace formal content acceptance'))
    assert all(r['eligible'] for r in smoke),'Incomplete export smoke; inspect without retuning formal checkpoint'
    old_models=sorted(p.name for p in (e.P/'scripts').iterdir() if p.is_dir() and p.name.startswith('MiniMind-'))
    shutil.copytree(stage,destination)
    (e.OUT/'web_demo.before_strong_teacher.py').write_text(prior,encoding='utf-8',newline='\n')
    web.write_text(changed,encoding='utf-8',newline='\n')
    e.save('demo_export.json',dict(published=True,model=name,folder=str(destination),version=VERSION,lineage=[e.BASE,name],
        all_parameter_tensors_exact=True,tokenizer_and_template_equal=True,conversion_checks=parity,old_models_preserved=old_models,
        runtime_teacher=False,default_max_new_tokens=384,maximum_selected_tools=6,
        files_sha256={p.name:e.sha(p) for p in destination.iterdir() if p.is_file()}))
    print(VERSION,'exported; previous models retained.',flush=True)

if __name__=='__main__':main()
