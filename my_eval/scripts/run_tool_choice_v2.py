"""Sequential GPU stages. No training is permitted after the final lock."""
import argparse
import gc
import time
import tool_choice_v2 as e
import eval_tool_choice_v2 as v
import torch

def main(stage):
    started=time.perf_counter()
    if stage=='development':
        assert not (e.OUT/'final_lock.json').exists()
        v.refresh_routes(e.BASE)
        for name in [e.BASE,'tool_choice_e1','tool_choice_e2','tool_choice_e3']:
            assert (e.P/f'out/{name}_768.pth').exists()
            if name!=e.BASE:v.evaluate('dev',name)
            v.nll('loss_dev',name)
            gc.collect();torch.cuda.empty_cache()
        v.select()
    else:
        selected=e.read(e.OUT/'final_lock.json')['selected']
        for name in [e.BASE,selected]:
            for step in ['final','guard','loss_final','identity']:
                print('Start',step,name,flush=True)
                if step=='loss_final':v.nll(step,name)
                else:v.evaluate(step,name)
                gc.collect();torch.cuda.empty_cache()
    e.save(f'{stage}_runtime.json',dict(seconds=time.perf_counter()-started,completed=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','final']);main(p.parse_args().stage)
