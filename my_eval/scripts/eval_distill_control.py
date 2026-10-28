"""Freeze and evaluate both task-3 controls using the historical generation protocol.

Primary subsets (math197 and chat29) were fixed from training overlap checks,
before either control was trained. The original200/30 results are also retained.
"""
import argparse
from datetime import datetime, timezone
import statistics as st
import time

import distill_control as d

c, P, OUT, save = d.c, d.P, d.OUT, d.save
NAMES = ['bb_dist', 'bb_sft_replay', 'bb_sft_new', 'bb_rl']
HISTORY = {'bb_dist': 'agent_bb200.json', 'bb_rl': 'agent_bbrl200.json'}


def lock():
    path = OUT/'evaluation_lock.json'
    if path.exists():
        verify()
        return
    assert not any(OUT.glob('*_final.json')), 'Freeze both models before final evaluation'
    plan = c.read(OUT/'plan.json')
    audit = c.read(OUT/'data_audit.json')
    assert len(plan['primary_math_indices']) == 197 and len(plan['primary_chat_indices']) == 29
    for name in HISTORY:
        assert c.digest(P/f'out/{name}_768.pth') == plan['input_sha256'][f'out\\{name}_768.pth']
    training = {}
    for name in d.ARMS:
        rec = c.read(OUT/f'train_{name}.json')
        assert not rec['trial'] and rec['settings'] == plan['training']
        assert rec['weight_sha256'] == c.digest(P/f'out/{name}_768.pth')
        assert rec['data_sha256'] == audit['dataset_sha256'][name]
        training[name] = rec
    protocols = [P/'my_eval/scripts'/s for s in [
        'official_compare.py', 'distill_control.py', 'train_distill_control.py', 'eval_distill_control.py',
        'eval_agent_math.py', 'score_rm.py', 'heldout_loss.py']]
    protocols += [P/s for s in ['trainer/train_full_sft.py', 'trainer/train_agent.py',
        'trainer/rollout_engine.py', 'trainer/trainer_utils.py', 'model/model_minimind.py',
        'dataset/lm_dataset.py', 'model/tokenizer.json', 'model/tokenizer_config.json',
        'dataset/agent_math_eval200.jsonl', 'dataset/dpo_eval300.jsonl']]
    protocols += [OUT/s for s in ['plan.json', 'protocol_amendment_chat.json', 'data_audit.json',
        'replay_mix.jsonl', 'new_mix.jsonl', 'teacher_new_sft.jsonl']]
    protocols += [P/'my_eval/results'/s for s in [*HISTORY.values(), 'rm_bbrl_compare.json']]
    record = {'locked_utc': datetime.now(timezone.utc).isoformat(),
        'environment': {'python':d.sys.version,'torch':c.torch.__version__,
            'transformers':c.transformers.__version__,'datasets':d.datasets.__version__,
            'gpu':c.torch.cuda.get_device_name(),'python_executable':d.sys.executable},
        'models': NAMES, 'training': training,
        'weight_sha256': {n:c.digest(P/f'out/{n}_768.pth') for n in NAMES},
        'files_sha256': {str(f.relative_to(P)):c.digest(f) for f in protocols},
        'primary_math_indices': plan['primary_math_indices'], 'primary_chat_indices': plan['primary_chat_indices'],
        'decision': 'Both fixed controls evaluated, no selection or further training based on final scores.'}
    save(path.name, record)
    print('Both control weights and evaluation protocol locked before final evaluation.', flush=True)


def verify():
    record = c.read(OUT/'evaluation_lock.json')
    for n,h in record['weight_sha256'].items():
        assert c.digest(P/f'out/{n}_768.pth') == h, n
    for f,h in record['files_sha256'].items():
        assert c.digest(P/f) == h, f
    return record


def replay():
    frozen = verify()
    path = OUT/'baseline_replay.json'
    record = c.read(path) if path.exists() else {}
    ds = c.AgentRLDataset(str(P/'dataset/agent_math_eval200.jsonl'), c.TOK)
    convs = c.prompts()
    chats = c.read(P/'my_eval/results/rm_bbrl_compare.json')
    assert chats['prompts'] == convs
    for name,filename in HISTORY.items():
        if name in record and record[name].get('passed'):
            continue
        old = c.read(P/'my_eval/results'/filename)[name]
        model = c.load(name, half=False)
        engine = c.engine_for(model)
        checks = []
        for i in [0, 99, 199]:
            for s in range(2):
                row = c.math_row(engine, ds[i], i, s)
                checks.append({'problem':i, 'sample':s, 'identical': row == old[2*i+s]})
        del engine, model
        c.release()
        model = c.load(name, half=True)
        texts = c.gen_chat(model, convs[0], 0)
        expected = [r['response'] for r in chats['results'][name] if r['prompt'] == 0]
        del model
        c.release()
        ok = all(r['identical'] for r in checks) and texts == expected
        record[name] = {'weight_sha256':frozen['weight_sha256'][name], 'math': checks,
            'chat_responses_equal':[a==b for a,b in zip(texts,expected)], 'passed':ok}
        save(path.name,record)
        assert ok, f'Historical replay mismatch for {name}; investigate before reusing outputs'
        print(f'{name}: six math rollouts and three chat responses exactly reproduced.',flush=True)
    from trainer.trainer_utils import LMForRewardModel
    if not record.get('reward_score_replay'):
        rm = LMForRewardModel('D:/minimind/internlm2-1_8b-reward',device='cuda',dtype=c.torch.float16)
        checks = []
        for name in HISTORY:
            for row in chats['results'][name][:3]:
                messages=[{'role':m['role'],'content':m['content']} for m in convs[row['prompt']]]
                values=c.score(row['response'],messages,rm)
                diffs={k:abs(values[k]-row[k]) for k in ['rm','total']}
                checks.append({'model':name,'prompt':row['prompt'],'absolute_differences':diffs})
                assert all(x < 1e-6 for x in diffs.values()), checks[-1]
        record['reward_score_replay']=checks
        save(path.name,record)
        del rm
        c.release()
        print('Six baseline reward scores also reproduced.',flush=True)


def import_history():
    frozen = verify()
    replay_record = c.read(OUT/'baseline_replay.json')
    assert all(replay_record[n]['passed'] for n in HISTORY) and replay_record.get('reward_score_replay')
    maths = c.read(OUT/'math_final.json') if (OUT/'math_final.json').exists() else {}
    chats = c.read(OUT/'chat_final.json') if (OUT/'chat_final.json').exists() else {'prompts':c.prompts(),'results':{}}
    old_chats = c.read(P/'my_eval/results/rm_bbrl_compare.json')
    assert chats['prompts'] == old_chats['prompts']
    for name,file in HISTORY.items():
        old=c.read(P/'my_eval/results'/file)[name]
        if name in maths: assert maths[name] == old
        maths[name]=old
        old=old_chats['results'][name]
        if name in chats['results']: assert chats['results'][name] == old
        chats['results'][name]=old
    save('math_final.json',maths)
    save('chat_final.json',chats)
    save('history_reuse.json',{'models':list(HISTORY),'math_files':HISTORY,
        'chat_file':'rm_bbrl_compare.json','weight_sha256':{n:frozen['weight_sha256'][n] for n in HISTORY},
        'validation':'Exact text/rollout replay and reward scoring checks in baseline_replay.json'})


def loss(name):
    verify()
    path=OUT/'loss_final.json'
    result=c.read(path) if path.exists() else {}
    if name in result:
        assert len(result[name]['losses'])==296
        return
    rows=d.read_rows(P/'dataset/dpo_eval300.jsonl')
    model=c.load(name,half=True)
    losses,indices=[],[]
    with c.torch.no_grad():
        for i,row in enumerate(rows):
            prompt=c.TOK(c.TOK.apply_chat_template(row['chosen'][:-1],tokenize=False,add_generation_prompt=True),add_special_tokens=False).input_ids
            answer=c.TOK(row['chosen'][-1]['content'],add_special_tokens=False).input_ids
            ids=(prompt+answer)[:768]
            if len(prompt)>=len(ids): continue
            x=c.torch.tensor([ids],device='cuda')
            lp=c.F.log_softmax(model(x).logits.float()[0,len(prompt)-1:-1],-1)
            losses.append(-lp.gather(1,x[0,len(prompt):,None]).mean().item())
            indices.append(i)
    assert len(indices)==296 and sorted(set(range(300))-set(indices))==[78,96,116,121]
    result[name]={'indices':indices,'losses':losses,'mean':st.mean(losses)}
    if name=='bb_rl':
        prior=c.read(P/'my_eval/results/identity_repair_20261024/loss_final.json')['bb_rl']
        assert indices==prior['indices'] and max(abs(a-b) for a,b in zip(losses,prior['losses']))<1e-6
    save(path.name,result)
    del model
    c.release()
    print(f'{name}: corrected answer-only loss {st.mean(losses):.8f}, n=296',flush=True)


def math_eval(name):
    verify()
    assert name in d.ARMS
    path=OUT/'math_final.json'
    result=c.read(path) if path.exists() else {}
    rows=result.setdefault(name,[])
    assert len(rows)%2==0 and [r['problem'] for r in rows]==[i for i in range(len(rows)//2) for _ in range(2)]
    if len(rows)==400:return
    ds=c.AgentRLDataset(str(P/'dataset/agent_math_eval200.jsonl'),c.TOK)
    model=c.load(name,half=False)
    engine=c.engine_for(model)
    start=time.perf_counter()
    for i in range(len(rows)//2,len(ds)):
        rows.extend(c.math_row(engine,ds[i],i,s) for s in range(2))
        save(path.name,result)
        if (i+1)%10==0 or i+1==len(ds):
            print(f'{name} math {i+1}/200; {time.perf_counter()-start:.0f}s',flush=True)
    del engine,model
    c.release()


def chat_eval(name):
    verify()
    assert name in d.ARMS
    from trainer.trainer_utils import LMForRewardModel
    convs=c.prompts()
    path=OUT/'chat_final.json'
    result=c.read(path) if path.exists() else {'prompts':convs,'results':{}}
    assert result['prompts']==convs
    rows=result['results'].setdefault(name,[])
    assert len(rows)%3==0 and [r['prompt'] for r in rows]==[i for i in range(len(rows)//3) for _ in range(3)]
    if len(rows)==90:return
    rm=LMForRewardModel('D:/minimind/internlm2-1_8b-reward',device='cuda',dtype=c.torch.float16)
    model=c.load(name,half=True)
    start=time.perf_counter()
    for i in range(len(rows)//3,len(convs)):
        messages=[{'role':m['role'],'content':m['content']} for m in convs[i]]
        for response in c.gen_chat(model,convs[i],i):
            rows.append({'prompt':i,'response':response,**c.score(response,messages,rm)})
        save(path.name,result)
        print(f'{name} chat {i+1}/30; {time.perf_counter()-start:.0f}s',flush=True)
    del model,rm
    c.release()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['lock','verify','replay','import_history','loss','math','chat'])
    ap.add_argument('--model',choices=NAMES)
    args=ap.parse_args()
    if args.stage in ['lock','verify','replay','import_history']:globals()[args.stage]()
    else:
        assert args.model
        {'loss':loss,'math':math_eval,'chat':chat_eval}[args.stage](args.model)
