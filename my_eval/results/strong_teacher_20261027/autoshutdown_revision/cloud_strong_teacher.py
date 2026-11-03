"""Run only on the user's AutoDL instance: bounded download + teacher generation.

Budget is enforced through a conservative wall-clock deadline and shutdown
request, not through a claim that Python controls the provider's billing.
Do not launch until the instance's shutdown command has been checked.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

def command(cmd,deadline,log):
    remaining=deadline-time.monotonic()
    if remaining<=90:raise TimeoutError('Too little time remains before shutdown deadline')
    with log.open('a',encoding='utf-8') as f:
        child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=child.wait(timeout=max(1,remaining-60))
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
    if code:raise RuntimeError(f'Child exited {code}; see {log.name}')

def main(a):
    root=Path(__file__).resolve().parent
    assert sys.platform=='linux' and Path('/root/autodl-tmp').is_dir(),'AutoDL data-directory check failed'
    assert root.is_relative_to(Path('/root/autodl-tmp')),'Place package on the AutoDL data disk'
    assert 0<a.max_seconds<=7200 and 0<a.hourly_cny<=2.08
    assert a.budget_remaining_cny>=a.max_seconds*a.hourly_cny/3600+.5,'Budget must include shutdown overhead'
    shutdown=shutil.which('shutdown');assert shutdown,'Verify AutoDL shutdown support before launching'
    assert a.shutdown_verified,'Read-only preflight plus explicit --shutdown-verified is required'
    manifest=json.loads((root/'package_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files_sha256'].items():
        assert Path(name).name==name
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected, f'Package integrity failed: {name}'
    assert not (root/'CLOUD_RUNNING.json').exists(),'A job marker already exists; inspect before restarting'
    runtime=root/'runtime';runtime.mkdir(exist_ok=True)
    os.environ.update(HF_HOME=str(runtime/'hf-cache'),TMPDIR=str(runtime),PIP_CACHE_DIR=str(runtime/'pip-cache'),
        PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    os.environ.pop('HF_HUB_OFFLINE',None);os.environ.pop('TRANSFORMERS_OFFLINE',None)
    start=time.time();deadline=time.monotonic()+a.max_seconds
    log=root/'cloud_job.log';status=dict(start_unix=start,hourly_cny=a.hourly_cny,budget_remaining_cny=a.budget_remaining_cny,max_seconds=a.max_seconds,completed=False)
    (root/'CLOUD_RUNNING.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
    try:
        if shutil.disk_usage(root).free<19*2**30:raise RuntimeError('Need at least19GiB free on data disk before downloading7B')
        # Reuse the CUDA torch installed in the image. A separate environment on
        # the data disk avoids replacing the provider's torch installation.
        env=root/'venv'
        if not (env/'bin/python').exists():command([sys.executable,'-m','venv','--system-site-packages',str(env)],deadline,log)
        py=str(env/'bin/python')
        command([py,'-m','pip','install','transformers==4.57.6'],deadline,log)
        download=root/'download_teacher.py'
        command([py,str(download)],deadline,log)
        generator=[py,str(root/'generate_strong_teacher.py'),'--requests',str(root/'teacher_requests.json'),
            '--settings',str(root/'teacher_settings.json'),'--model-dir',str(root/'Qwen2.5-7B-Instruct'),
            '--output',str(root/'teacher_strong.jsonl'),'--arm','strong']
        command(generator+['--limit','32'],deadline,log)
        trial=json.loads((root/'teacher_strong.runtime.json').read_text(encoding='utf-8'))
        estimated=trial['generation_seconds']/32*968+trial['load_seconds']+90
        status['pilot']=trial;status['estimated_remaining_generation_seconds']=estimated
        (root/'pilot_estimate.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
        if estimated>deadline-time.monotonic():raise TimeoutError('Pilot estimate exceeds remaining time; preserved pilot, no full run')
        command(generator,deadline,log)
        full=json.loads((root/'teacher_strong.runtime.json').read_text(encoding='utf-8'))
        assert full['completed'] and full['recorded']==1000
        status['completed']=True
    except BaseException as exc:
        status['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        status['elapsed_seconds']=time.time()-start
        status['job_compute_charge_estimate_cny']=status['elapsed_seconds']/3600*a.hourly_cny
        status['charge_note']='Excludes time before job start, shutdown latency and any separately billed storage.'
        (root/'cloud_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('Job ended; requesting AutoDL instance shutdown. Results remain on the data disk.',flush=True)
        # AutoDL supplies the instance shutdown command. Never issue this on the
        # local Windows machine or on an arbitrary Linux host.
        result=subprocess.run([shutdown],capture_output=True,text=True,timeout=15)
        if result.returncode:
            (root/'SHUTDOWN_FAILED.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
            print('SHUTDOWN FAILED: stop the instance in AutoDL now.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--max-seconds',type=int,default=7200)
    p.add_argument('--hourly-cny',type=float,required=True);p.add_argument('--budget-remaining-cny',type=float,required=True)
    p.add_argument('--shutdown-verified',action='store_true');main(p.parse_args())
