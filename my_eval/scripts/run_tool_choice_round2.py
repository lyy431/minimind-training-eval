"""Sequential GPU stages for round2; same frozen base controls can be reused exactly."""
import argparse
import gc
import shutil
import time
import torch
import tool_choice_round2 as e
import eval_tool_choice_round2 as v
import eval_routing_round2 as r

def main(stage):
    started=time.perf_counter();old=e.P/'my_eval/results/tool_choice_20261026'
    if stage=='development':
        assert not (e.OUT/'final_lock.json').exists()
        for name in [e.BASE,'tool_choice2b_e1','tool_choice2b_e2','tool_choice2b_e3']:
            r.run('dev',name)
            if name==e.BASE:
                shutil.copyfile(old/f'loss_dev_{name}.json',e.OUT/f'loss_dev_{name}.json')
            else:v.nll('loss_dev',name)
            gc.collect();torch.cuda.empty_cache()
        v.select()
    else:
        lock=e.read(e.OUT/'final_lock.json');selected=lock['selected']
        for name in [e.BASE,selected]:
            r.run('final',name);gc.collect();torch.cuda.empty_cache()
            v.evaluate('fullmath',name);gc.collect();torch.cuda.empty_cache()
            if name==e.BASE:
                assert lock['weights'][name]==e.read(old/'final_lock.json')['weights'][name]
                for prefix in ['guard','loss_final','identity']:
                    shutil.copyfile(old/f'{prefix}_{name}.json',e.OUT/f'{prefix}_{name}.json')
                e.save('baseline_reuse.json',dict(source=str(old),model_sha256=lock['weights'][name],
                    scope='exact same historical197 math questions/seeds, NLL296 tokenization, identity14 seeds and unchanged shared generator',
                    source_sha256={prefix:e.sha(old/f'{prefix}_{name}.json') for prefix in ['guard','loss_final','identity']}))
            else:
                v.evaluate('guard',name);gc.collect();torch.cuda.empty_cache()
                v.nll('loss_final',name);gc.collect();torch.cuda.empty_cache()
                v.evaluate('identity',name);gc.collect();torch.cuda.empty_cache()
    e.save(f'{stage}_runtime.json',dict(completed=True,seconds=time.perf_counter()-started))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','final']);main(p.parse_args().stage)
