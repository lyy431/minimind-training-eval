"""Task 3: paired continued-distillation controls for the extra-data confound.

All inputs, models, caches and outputs stay on D:. No paid service is used.
Teacher generation follows teacher_gen.py (batch32, greedy, bf16, 256 tokens,
3 turns), with resumable raw results and a separate diagnostic correctness audit.
"""
import argparse
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import random
import re
import statistics as st
import sys
import time

P = Path(__file__).resolve().parents[2]
OUT = P / "my_eval/results/distill_control_20261025"
RUNTIME = P / "my_eval/runtime"
TEACHER = Path("D:/minimind/Qwen2.5-1.5B-Instruct")
OUT.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HOME="D:/conda/hf-cache", HF_DATASETS_CACHE=str(RUNTIME/"datasets"),
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TEMP=str(RUNTIME), TMP=str(RUNTIME),
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")
import datasets
import official_compare as c
from trainer.train_agent import execute_tool
from trainer.trainer_utils import safe_math_eval

BATCH, MAX_NEW, MAX_TURNS = 32, 256, 3
ARMS = {"bb_sft_replay": "replay_mix.jsonl", "bb_sft_new": "new_mix.jsonl"}


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def save(name, value):
    target = OUT / name
    temp = target.with_suffix(target.suffix+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    temp.replace(target)


def frozen_rows(name, rows):
    text = "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows)
    target = OUT/name
    if target.exists():
        assert target.read_text(encoding="utf-8") == text, f"Frozen dataset differs: {name}"
    else:
        target.write_text(text, encoding="utf-8", newline="\n")


def question(row):
    return "\n".join(m["content"] for m in row.get("conversations", row.get("chosen", [])) if m["role"] == "user").strip()


def key(text):
    # Preserve arithmetic operators. Removing all punctuation would conflate 1+2 and 1-2.
    return re.sub(r"\s+", "", text).casefold()


def canonical(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def prepare():
    assert not (OUT/"math_final.json").exists()
    old, new, test, old_teacher, mixed = [read_rows(P/"dataset"/n) for n in [
        "agent_math_train_3000.jsonl", "agent_math_train_3000_excl3000.jsonl", "agent_math_eval200.jsonl",
        "teacher_math_sft.jsonl", "bb_mix.jsonl"]]
    pools = {n:{key(question(r)) for r in rows} for n,rows in [
        ("old_source",old),("rl_source",new),("old_teacher",old_teacher),("bb_mix",mixed)]}
    test_keys = {key(question(r)) for r in test}
    contaminated = []
    for i,r in enumerate(test):
        seen = [n for n in ("old_teacher","rl_source","bb_mix") if key(question(r)) in pools[n]]
        if seen:
            contaminated.append({"eval_index":i,"question":question(r),"gt":r["gt"],"seen_in":seen})
    assert [r["eval_index"] for r in contaminated] == [15,125,181]
    source = [{"original_index":i,"problem":r} for i,r in enumerate(new) if key(question(r)) not in test_keys]
    assert len(source) == 2999
    frozen_rows("new_source2999.jsonl", source)
    counts = Counter(canonical(r) for r in old_teacher)
    general = []
    for r in mixed:
        s = canonical(r)
        if counts[s]: counts[s] -= 1
        else: general.append(r)
    assert len(general) == 3000 and not any(counts.values())
    lang_keys = {key(question(r)) for r in read_rows(P/"dataset/dpo_eval300.jsonl")}
    chat_keys = {key("\n".join(m["content"] for m in conv if m["role"] == "user")) for conv in c.prompts()}
    excluded_general = [r for r in general if key(question(r)) in test_keys | lang_keys | chat_keys]
    excluded_questions = {key(question(r)) for r in excluded_general}
    contaminated_chat = [i for i,conv in enumerate(c.prompts()) if key('\n'.join(m['content'] for m in conv if m['role']=='user')) in excluded_questions]
    general = [r for r in general if key(question(r)) not in test_keys | lang_keys | chat_keys]
    clean_teacher = [r for r in old_teacher if key(question(r)) not in test_keys]
    frozen_rows("old_teacher_clean.jsonl",clean_teacher)
    frozen_rows("common_general.jsonl",general)
    manifest = {"date":"2026-10-25", "primary_question":"Does extra teacher-labelled data explain the observed bb_dist-to-bb_rl improvement under this recipe?",
        "arms":{"bb_dist":"existing starting model", "bb_rl":"existing RL model (1500 updates, ~5.5h)",
            "bb_sft_replay":"from bb_dist; common old teacher + same general + repeated old teacher slots",
            "bb_sft_new":"from bb_dist; identical common rows + new-batch teacher slots"},
        "training":{"from_weight":"bb_dist","epochs":2,"learning_rate":2e-5,"batch_size":16,
            "max_seq_len":768,"num_workers":2,"seed":42,"optimizer":"original AdamW and cosine schedule", "use_compile":0},
        "matching":"Same row count, math-expression-count distribution in extra slots, shared permutation, starting weights, seed, epochs, batch and update count. Old extra slots sampled by expression count with shuffled cycling; no tuning on final evaluation.",
        "teacher":{"path":str(TEACHER),"batch_size":BATCH,"dtype":"bfloat16","do_sample":False,
            "max_new_tokens":MAX_NEW,"max_turns":MAX_TURNS,"acceptance":"original validate_gt_in_text; strict diagnostic audit reported separately, does not silently replace historical filtering"},
        "known_test_question_overlaps":contaminated,
        "known_chat_question_overlaps":contaminated_chat,
        "primary_chat_indices":[i for i in range(30) if i not in contaminated_chat],
        "primary_math_indices":[i for i in range(200) if i not in {r['eval_index'] for r in contaminated}],
        "historical_math_reporting":"Also report unchanged original200x2; primary comparison uses the predeclared clean197 subset for all models, no regeneration of old samples.",
        "counts":{"old_source_rows":len(old),"new_rl_source_rows":len(new),"new_teacher_source_rows":len(source),
            "old_source_unique_questions":len(pools['old_source']),"new_rl_source_unique_questions":len(pools['rl_source']),
            "old_new_question_overlap":len(pools['old_source']&pools['rl_source']),
            "old_teacher_original":len(old_teacher),"old_teacher_clean":len(clean_teacher),
            "general_original":3000,"general_clean":len(general),"general_excluded":len(excluded_general)},
        "limits":["One paired training seed; per-question SE excludes training-seed variation.",
            "RL and SFT compute, optimization, filtering and feedback are not identical; this is not an exact causal percentage decomposition.",
            "The RL batch has internal duplicate questions and 25 questions also present in the older pool; it was not entirely novel by user-question content.",
            "Known identical-question leakage is excluded from the primary197; semantic paraphrases and upstream pretraining exposure remain unverified."],
        "final_evaluation":{"math":"original200x2 protocol; clean197 primary; paired per-question1SE",
            "language":"corrected296 answer-only loss, same768 cap", "chat":"original30x3, same seeds,512tokens; clean29 primary, original30 also reported; RM and total reward separately"},
        "input_sha256":{str(f.relative_to(P)):c.digest(f) for f in [P/"dataset"/n for n in [
            "agent_math_train_3000.jsonl","agent_math_train_3000_excl3000.jsonl","agent_math_eval200.jsonl",
            "teacher_math_sft.jsonl","bb_mix.jsonl","dpo_eval300.jsonl"]]+[P/"out/bb_dist_768.pth",P/"out/bb_rl_768.pth",P/"my_eval/scripts/teacher_gen.py"]},
        "teacher_weight_sha256":c.digest(TEACHER/"model.safetensors")}
    save("plan.json",manifest)
    print(json.dumps({"counts":manifest['counts'],"known_test_question_overlaps":contaminated,"primary_math_n":len(manifest['primary_math_indices'])},ensure_ascii=False,indent=2),flush=True)


def strict_diagnostics(problem, mini, final):
    # Diagnostic only, intentionally separate from the historical acceptance rule.
    numbers = [float(s) for s in re.findall(r"(?<![0-9A-Za-z_.])[-+]?\d+(?:\.\d+)?(?![0-9A-Za-z_]|\.\d)",final.replace(",",""))]
    gt = [float(str(g).replace(",","")) for g in problem['gt']]
    whole = all(any(abs(g-x)<1e-6 for x in numbers) for g in gt)
    values, invalid, wrong_tool = [],0,False
    for m in mini:
        if not m.get('tool_calls'): continue
        for call in json.loads(m['tool_calls']):
            call=call['function']
            if call['name']!='calculate_math': wrong_tool=True; continue
            try:
                args=json.loads(call['arguments'])
                values.append(float(safe_math_eval(str(args.get('expression','')))))
            except Exception: invalid+=1
    computed=all(any(abs(g-x)<1e-6 for x in values) for g in gt)
    unfinished=bool(c.parse_tool_calls(final))
    rendered_messages=copy.deepcopy(mini)
    for message in rendered_messages:
        if isinstance(message.get('tool_calls'),str):
            message['tool_calls']=json.loads(message['tool_calls'])
    text=c.TOK.apply_chat_template(rendered_messages,tools=json.loads(problem['conversations'][0]['tools']),tokenize=False,add_generation_prompt=False)
    tokens=len(c.TOK(text,add_special_tokens=False).input_ids)
    return {"whole_numbers":whole,"all_values_calculated":computed,"unfinished_tool_call":unfinished,
        "wrong_tool":wrong_tool,"invalid_expressions":invalid,"tokens":tokens,
        "strict_correct":whole and computed and not unfinished and not wrong_tool and not invalid}


def teacher_model():
    tok=c.AutoTokenizer.from_pretrained(TEACHER,local_files_only=True)
    tok.padding_side='left'
    model=c.AutoModelForCausalLM.from_pretrained(TEACHER,torch_dtype=c.torch.bfloat16,local_files_only=True).cuda().eval()
    return model,tok


def teacher_batch(model,tok,rows):
    chats=[[{"role":"user","content":r['conversations'][1]['content']}] for r in rows]
    mini=[[{"role":"system","content":"","tools":r['conversations'][0]['tools']},
        {"role":"user","content":r['conversations'][1]['content']}] for r in rows]
    toolsets=[json.loads(r['conversations'][0]['tools']) for r in rows]
    active=list(range(len(rows))); final=['']*len(rows); used=[False]*len(rows); raw=[[] for _ in rows]
    for turn in range(MAX_TURNS):
        if not active: break
        texts=[tok.apply_chat_template(chats[i],tools=[{"type":"function",**t} for t in toolsets[i]],
            tokenize=False,add_generation_prompt=True) for i in active]
        enc=tok(texts,return_tensors='pt',padding=True).to('cuda')
        with c.torch.no_grad():
            generated=model.generate(**enc,max_new_tokens=MAX_NEW,do_sample=False,pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outputs=tok.batch_decode(generated[:,enc.input_ids.shape[1]:],skip_special_tokens=True)
        still=[]
        for i,text in zip(active,outputs):
            raw[i].append(text)
            calls=c.parse_tool_calls(text)
            if not calls or turn==MAX_TURNS-1:
                final[i]=text.strip()
                mini[i].append({"role":"assistant","content":final[i]})
                continue
            used[i]=True
            prefix=text.split('<tool_call>')[0].strip()
            qcalls,mcalls=[],[]
            for call in calls:
                args=call.get('arguments',{})
                if isinstance(args,str):
                    try: args=json.loads(args)
                    except Exception: args={}
                qcalls.append({"type":"function","function":{"name":call.get('name',''),"arguments":args}})
                mcalls.append({"function":{"name":call.get('name',''),"arguments":json.dumps(args,ensure_ascii=False)}})
            chats[i].append({"role":"assistant","content":prefix,"tool_calls":qcalls})
            mini[i].append({"role":"assistant","content":prefix,"tool_calls":json.dumps(mcalls,ensure_ascii=False)})
            for call in qcalls:
                result=execute_tool(call['function']['name'],call['function']['arguments'])
                string=(json.dumps(result,ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]
                chats[i].append({"role":"tool","content":string}); mini[i].append({"role":"tool","content":string})
            still.append(i)
        active=still
    result=[]
    for i,r in enumerate(rows):
        ok=bool(final[i]) and len(c.validate_gt_in_text(final[i],r['gt']))==len(r['gt'])
        result.append({"accepted":ok,"used_tool":used[i],"gt":r['gt'],"raw_turns":raw[i],"final":final[i],
            "sft":{"conversations":mini[i]},"diagnostics":strict_diagnostics(r,mini[i],final[i])})
    return result


def generate_teacher(pilot=False):
    source=read_rows(OUT/'new_source2999.jsonl')
    path=OUT/'teacher_raw.json'
    stored=c.read(path) if path.exists() else {"rows":[],"batch_seconds":[],"source_sha256":c.digest(OUT/'new_source2999.jsonl')}
    assert stored['source_sha256']==c.digest(OUT/'new_source2999.jsonl')
    model,tok=teacher_model()
    if pilot:
        old=read_rows(P/'dataset/agent_math_train_3000.jsonl')[:32]
        replay=teacher_batch(model,tok,old)
        accepted=[r['sft'] for r in replay if r['accepted']]
        reference=read_rows(OUT/'original_teacher_old32.jsonl')
        assert accepted==reference, 'Instrumented teacher driver differs from the original on the same32 inputs'
        historical=read_rows(P/'dataset/teacher_math_sft.jsonl')[:len(reference)]
        save('teacher_replay.json',{'original_driver_matches':True,'old32_accepted':len(accepted),
            'historical_prefix_exact':accepted==historical,'rows':replay})
        print(f'Teacher driver replay verified: {len(accepted)}/32 accepted; historical prefix exact={accepted==historical}',flush=True)
    limit=64 if pilot else len(source)
    assert len(stored['rows'])%BATCH==0 or len(stored['rows'])==len(source)
    c.torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for b in range(len(stored['rows']),limit,BATCH):
        tick=time.perf_counter()
        batch=source[b:min(b+BATCH,len(source))]
        generated=teacher_batch(model,tok,[r['problem'] for r in batch])
        for j,r in enumerate(generated):
            r.update(position=b+j,original_index=batch[j]['original_index'])
        stored['rows'].extend(generated)
        elapsed=time.perf_counter()-tick
        stored['batch_seconds'].append(elapsed)
        stored['peak_gpu_allocated_gib']=max(stored.get('peak_gpu_allocated_gib',0),c.torch.cuda.max_memory_allocated()/2**30)
        stored['peak_gpu_reserved_gib']=max(stored.get('peak_gpu_reserved_gib',0),c.torch.cuda.max_memory_reserved()/2**30)
        save('teacher_raw.json',stored)
        n=len(stored['rows']); correct=sum(r['accepted'] for r in stored['rows'])
        strict=sum(r['accepted'] and r['diagnostics']['strict_correct'] for r in stored['rows'])
        print(f'Teacher {n}/{len(source)}: accepted {correct} ({100*correct/n:.1f}%), strict among accepted {strict}/{correct}; batch {elapsed:.1f}s, this run {time.perf_counter()-start:.0f}s',flush=True)
    if pilot:
        save('pilot.json',{'completed':len(stored['rows']), 'batch_size':BATCH,
            'new_batches_seconds':stored['batch_seconds'],
            'estimated_full_generation_minutes':st.mean(stored['batch_seconds'])*((len(source)+31)//32)/60,
            'peak_gpu_allocated_gib':stored['peak_gpu_allocated_gib'],'peak_gpu_reserved_gib':stored['peak_gpu_reserved_gib']})
    else:
        assert len(stored['rows'])==len(source)
        frozen_rows('teacher_new_sft.jsonl',[r['sft'] for r in stored['rows'] if r['accepted']])
    del model
    c.release()


def build_controls():
    assert not any(OUT.glob('*_final.json')), 'Build and freeze controls before final evaluation'
    plan=c.read(OUT/'plan.json')
    raw=c.read(OUT/'teacher_raw.json')
    source=read_rows(OUT/'new_source2999.jsonl')
    assert len(raw['rows'])==len(source)==2999
    old=read_rows(OUT/'old_teacher_clean.jsonl')
    general=read_rows(OUT/'common_general.jsonl')
    old_source=read_rows(P/'dataset/agent_math_train_3000.jsonl')
    by_signature={}
    signature=lambda r:(key(question(r)),r['conversations'][0]['tools'])
    for row in old_source:
        by_signature[signature(row)]=row
    strata=defaultdict(list)
    old_audit=[]
    for i,row in enumerate(old):
        problem=by_signature[signature(row)]
        strata[len(problem['gt'])].append(i)
        old_audit.append(strict_diagnostics(problem,row['conversations'],row['conversations'][-1]['content']))
    accepted=[r for r in raw['rows'] if r['accepted']]
    n=len(accepted)
    assert n>=1500, 'Unexpectedly few correct teacher examples; inspect before training'
    rng=random.Random(202610253)
    queues={}
    def old_index(k):
        if not queues.get(k):
            queues[k]=strata[k].copy(); rng.shuffle(queues[k])
        return queues[k].pop()
    slots=[]
    for i,row in enumerate(accepted):
        j=old_index(len(row['gt']))
        slots.append({'slot':i,'old_teacher_index':j,'new_position':row['position'],
            'new_original_index':row['original_index'],'expression_count':len(row['gt'])})
    common=old+general
    replay=common+[old[r['old_teacher_index']] for r in slots]
    new=common+[r['sft'] for r in accepted]
    assert len(replay)==len(new)
    permutation=list(range(len(new)))
    random.Random(202610254).shuffle(permutation)
    replay=[replay[i] for i in permutation]; new=[new[i] for i in permutation]
    test=read_rows(P/'dataset/agent_math_eval200.jsonl')
    lang=read_rows(P/'dataset/dpo_eval300.jsonl')
    protected={"math":{key(question(r)) for r in test},"language":{key(question(r)) for r in lang},
        "chat":{key('\n'.join(m['content'] for m in conv if m['role']=='user')) for conv in c.prompts()}}
    overlaps={arm:{k:len({key(question(r)) for r in rows}&v) for k,v in protected.items()}
        for arm,rows in [('bb_sft_replay',replay),('bb_sft_new',new)]}
    assert not any(v for x in overlaps.values() for v in x.values()), overlaps
    frozen_rows('replay_mix.jsonl',replay); frozen_rows('new_mix.jsonl',new)
    save('matched_layout.json',{'common_old':len(old),'common_general':len(general),'slots':slots,'shared_permutation':permutation})
    from dataset.lm_dataset import SFTDataset
    supervision={}
    for arm,file in ARMS.items():
        random.seed(42)
        ds=SFTDataset(str(OUT/file),c.TOK,max_length=768)
        counts=[int((ds[i][1][1:]!=-100).sum()) for i in range(len(ds))]
        supervision[arm]={'rows':len(ds),'supervised_tokens':sum(counts),'zero_supervised_rows':sum(v==0 for v in counts)}
        assert not supervision[arm]['zero_supervised_rows']
    diagnostic=lambda records:{'n':len(records),'strict_correct':sum(r['strict_correct'] for r in records),
        'no_all_calculator_values':sum(not r['all_values_calculated'] for r in records),
        'wrong_whole_numbers':sum(not r['whole_numbers'] for r in records),'unfinished':sum(r['unfinished_tool_call'] for r in records),
        'wrong_tool':sum(r['wrong_tool'] for r in records),'over_768_tokens':sum(r['tokens']>768 for r in records)}
    audit={'common_old_teacher':len(old),'common_general':len(general),'extra_slots':n,'total_rows_per_arm':len(new),
        'steps_per_epoch':(len(new)+15)//16,'total_steps_per_arm':2*((len(new)+15)//16),
        'extra_expression_counts':dict(Counter(s['expression_count'] for s in slots)),
        'teacher_acceptance':{'attempted':len(raw['rows']),'accepted':n,'percent':100*n/len(raw['rows'])},
        'old_teacher_diagnostic':diagnostic(old_audit),'new_teacher_diagnostic':diagnostic([r['diagnostics'] for r in accepted]),
        'question_overlap_with_final':overlaps,'supervision_seed42':supervision,
        'new_accepted_questions_also_in_old_teacher':len({key(question(r['sft'])) for r in accepted}&{key(question(r)) for r in old}),
        'base_sha256':c.digest(P/'out/bb_dist_768.pth'),
        'dataset_sha256':{arm:c.digest(OUT/f) for arm,f in ARMS.items()},
        'protocol_sha256':{str(f.relative_to(P)):c.digest(f) for f in [
            P/'trainer/train_full_sft.py',P/'dataset/lm_dataset.py',P/'my_eval/scripts/official_compare.py',P/'dataset/agent_math_eval200.jsonl',P/'dataset/dpo_eval300.jsonl']}}
    save('data_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['prepare','pilot','teacher','build'])
    args=ap.parse_args()
    if args.stage=='prepare': prepare()
    elif args.stage=='build': build_controls()
    else: generate_teacher(pilot=args.stage=='pilot')
