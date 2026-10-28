"""Publish a separately named model only after all locked final retention gates pass."""
import shutil
import time
import tool_choice_round2 as e
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from best_of_n import generate_candidate
from demo_tools import execute_tool
from strict_math_rescore import grade

def main():
    s=e.read(e.OUT/'summary.json')
    effective=s['accepted']
    e.save('release_decision.json',dict(effective_accepted=effective,experiment='round2 fresh stratified test',weights=e.read(e.OUT/'final_lock.json')['weights']))
    if not effective:
        e.save('demo_export.json',dict(published=False,gates=s['gates'],reason='Round2 predefined fresh-test acceptance not met'))
        print('Research checkpoint preserved; demo acceptance not met.',flush=True);return
    name=s['candidate'];stage=e.OUT/'demo_hf_validated';destination=e.P/'scripts/MiniMind-YuyangLU-v3'
    assert not destination.exists(),'Existing demo folders are preserved'
    model=e.load_model(name);tok=e.load_tok();model.save_pretrained(stage);tok.save_pretrained(stage)
    restored=AutoModelForCausalLM.from_pretrained(stage,local_files_only=True,torch_dtype=torch.float16).cuda().eval()
    rtok=AutoTokenizer.from_pretrained(stage,local_files_only=True)
    assert tok.get_vocab()==rtok.get_vocab()
    assert model.state_dict().keys()==restored.state_dict().keys()
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    checks=[]
    for q,tools in [('你是谁？',None),('请计算27*43，另外612/17',e.menu(6,0,600005)),('请写一句欢迎新同学的话。',e.menu(6,0,600006))]:
        text=tok.apply_chat_template([dict(role='user',content=q)],tools=tools,tokenize=False,add_generation_prompt=True)
        assert text==rtok.apply_chat_template([dict(role='user',content=q)],tools=tools,tokenize=False,add_generation_prompt=True)
        ids=tok(text,return_tensors='pt').input_ids.cuda()
        with torch.inference_mode():
            a=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=24,do_sample=False)
            b=restored.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=24,do_sample=False)
        assert torch.equal(a,b);checks.append(dict(question=q,prompt_tokens=ids.shape[1],greedy24_identical=True))
    del model;torch.cuda.empty_cache()
    sys=dict(role='system',content='你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。')
    smoke=[]
    for i,(q,tools) in enumerate([('你是谁？',None),('请计算27*43，另外612/17',e.menu(6,0,600005)),('请写一句欢迎新同学的话。',e.menu(6,0,600006))]):
        messages=[dict(role='user',content=q)] if tools else [sys,dict(role='user',content=q)]
        c=generate_candidate(restored,rtok,messages,tools=tools,execute_tool=execute_tool,seed=606000+i,max_new_tokens=384)
        smoke.append(dict(question=q,**c))
    # Record behavior, without retuning a checkpoint after seeing it.
    checks_pass=bool('yuyang' in smoke[0]['final'].casefold() and 'minimind' in smoke[0]['final'].casefold()
        and grade(smoke[1]['final'],smoke[1]['question'],['1161','36'],unfinished=not smoke[1]['eligible'])['status']=='pass'
        and e.first_call_metrics(smoke[1],dict(kind='math',question=smoke[1]['question'],expressions=['27*43','612/17']))['arguments_ok']
        and '<tool_call' not in smoke[2]['turns'][0] and smoke[2]['final'].strip())
    e.save('demo_smoke.json',dict(passed=checks_pass,rows=smoke))
    if not checks_pass:
        e.save('demo_export.json',dict(published=False,conversion_checks=checks,reason='Actual webpage-setting smoke did not pass; no retuning on final'))
        print('Export parity passed but demo smoke needs inspection; no automatic publication.',flush=True);return
    shutil.copytree(stage,destination)
    web=e.P/'scripts/web_demo.py';old=web.read_text(encoding='utf-8')
    (e.OUT/'web_demo.before_task5.py').write_text(old,encoding='utf-8')
    changed=old.replace("'tool_select': '工具选择（最多4个）'","'tool_select': '工具选择（最多6个）'")
    changed=changed.replace("'tool_select': 'Tool Selection (max 4)'","'tool_select': 'Tool Selection (max 6)'")
    changed=changed.replace('disabled=(selected_count >= 4 and','disabled=(selected_count >= 6 and')
    changed=changed.replace('if checked and len(st.session_state.selected_tools) < 4:','if checked and len(st.session_state.selected_tools) < 6:')
    changed=changed.replace('from demo_tools import TOOLS, TOOL_SHORT_NAMES, execute_tool',
        'from demo_tools import TOOLS, TOOL_SHORT_NAMES, execute_tool\nfrom demo_tool_guard import execute_selected_tool')
    changed=changed.replace("result = execute_tool(tc.get('name', ''), tc.get('arguments', {}))",
        'result = execute_selected_tool(tc, tools, execute_tool)')
    assert changed!=old and 'selected_count >= 6' in changed
    web.write_text(changed,encoding='utf-8',newline='\n')
    e.save('demo_export.json',dict(published=True,model=name,demo_folder=str(destination),all_parameter_tensors_exact=True,
        tokenizer_and_template_equal=True,conversion_checks=checks,old_models_preserved=True,maximum_selected_tools=6,tool_whitelist_enforced=True,
        files_sha256={p.name:e.sha(p) for p in destination.iterdir() if p.is_file()}))
    print('Published MiniMind-YuyangLU-v3; old versions preserved; UI supports6 tools.',flush=True)

if __name__=='__main__':main()
