"""One GPU process at a time, resumable between recorded evaluation stages."""
import argparse
import subprocess
import sys
import chat_repair_common as e

def call(args,log):
    with (e.OUT/log).open('a',encoding='utf-8') as out:
        process=subprocess.Popen([sys.executable,'-u',*args],cwd=e.P,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        for line in process.stdout:print(line,end='',flush=True);out.write(line);out.flush()
        if process.wait():raise SystemExit(process.returncode)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['dev','dev_identity','final']);a=p.parse_args()
    if a.stage in ['dev','dev_identity']:
        assert not (e.OUT/'final_lock.json').exists()
        prefix='chat_repair_e' if a.stage=='dev' else 'chat_repair_id_e'
        for epoch in [1,2,3]:call(['my_eval/scripts/eval_chat_repair.py','dev','--model',f'{prefix}{epoch}'],'development.log')
        call(['my_eval/scripts/review_chat_repair.py','dev'],'development.log')
    else:
        name=e.read(e.OUT/'selection.json')['selected']
        for model in [e.BASE,name]:call(['my_eval/scripts/eval_chat_repair.py','final','--model',model],'final.log')
        call(['my_eval/scripts/review_chat_repair_safe.py','final'],'final.log')

if __name__=='__main__':main()
