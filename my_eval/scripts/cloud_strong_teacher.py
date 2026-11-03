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
import zipfile

RESULT_FILES = [
    'teacher_strong.jsonl', 'teacher_strong.manifest.json',
    'teacher_strong.runtime.json', 'download_manifest.json',
    'pilot_estimate.json', 'cloud_status.json', 'cloud_job.log',
    'shutdown_request.json', 'SHUTDOWN_FAILED.txt',
]

def write_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)

def bundle_results(root):
    """Save only small result/log files, including partial answers on a failure."""
    files=[root/name for name in RESULT_FILES if (root/name).is_file()]
    manifest=dict(files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        weights_included=False,shutdown_note='Archive is saved before the shutdown request. Confirm power-off in AutoDL.')
    write_json(root/'results_manifest.json',manifest)
    path=root/'strong_teacher_results.zip';tmp=root/'strong_teacher_results.zip.tmp'
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files+[root/'results_manifest.json']:archive.write(file,file.name)
    tmp.replace(path)
    return path

def finish_job(root,status,shutdown):
    """A save/archive error must never prevent the instance shutdown attempt."""
    try:
        status['elapsed_seconds']=time.time()-status['start_unix']
        status['job_compute_charge_estimate_cny']=status['elapsed_seconds']/3600*status['hourly_cny']
        status['charge_note']='Excludes time before job start, shutdown latency and separately billed storage.'
        write_json(root/'cloud_status.json',status)
        write_json(root/'shutdown_request.json',dict(command=shutdown,requested_after_result_save=True,
            actual_power_off_confirmed=False,note='An intent record, not proof that the platform stopped billing.'))
        print('Saving result archive:',bundle_results(root),flush=True)
    except BaseException as exc:
        print(f'Result saving failed: {type(exc).__name__}: {exc}; still requesting shutdown.',flush=True)
    finally:
        print('Requesting AutoDL instance shutdown now.',flush=True)
        try:
            result=subprocess.run([shutdown],capture_output=True,text=True,timeout=15)
            if result.returncode:
                raise RuntimeError(f'shutdown exit {result.returncode}: {result.stdout}\n{result.stderr}')
        except BaseException as exc:
            print(f'SHUTDOWN FAILED: {exc}. Stop the instance in AutoDL now.',flush=True)
            try:
                (root/'SHUTDOWN_FAILED.txt').write_text(f'{type(exc).__name__}: {exc}\n',encoding='utf-8')
                bundle_results(root)
            except BaseException as save_error:
                print(f'Could not save shutdown failure details: {save_error}',flush=True)

def terminate(signum,frame):
    raise SystemExit(128+signum)

def command(cmd,deadline,log):
    remaining=deadline-time.monotonic()
    if remaining<=90:raise TimeoutError('Too little time remains before shutdown deadline')
    with log.open('a',encoding='utf-8') as f:
        child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=child.wait(timeout=max(1,remaining-60))
        except BaseException:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
    if code:raise RuntimeError(f'Child exited {code}; see {log.name}')

def main(a):
    root=Path(__file__).resolve().parent
    assert sys.platform=='linux' and Path('/root/autodl-tmp').is_dir(),'AutoDL data-directory check failed'
    assert root.is_relative_to(Path('/root/autodl-tmp')),'Place package on the AutoDL data disk'
    assert a.shutdown_verified,'Read-only preflight plus explicit --shutdown-verified is required'
    shutdown='/usr/bin/shutdown'
    assert Path(shutdown).is_file() and os.access(shutdown,os.X_OK),'AutoDL shutdown command is unavailable'
    assert not (root/'CLOUD_RUNNING.json').exists(),'A job marker already exists; inspect before restarting'
    start=time.time();deadline=time.monotonic()+a.max_seconds
    log=root/'cloud_job.log';status=dict(start_unix=start,hourly_cny=a.hourly_cny,budget_remaining_cny=a.budget_remaining_cny,max_seconds=a.max_seconds,completed=False)
    previous_sigterm=signal.signal(signal.SIGTERM,terminate)
    try:
        assert 0<a.max_seconds<=7200 and 0<a.hourly_cny<=2.08
        assert a.budget_remaining_cny>=a.max_seconds*a.hourly_cny/3600+.5,'Budget must include shutdown overhead'
        manifest=json.loads((root/'package_manifest.json').read_text(encoding='utf-8'))
        for name,expected in manifest['files_sha256'].items():
            assert Path(name).name==name
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected, f'Package integrity failed: {name}'
        write_json(root/'CLOUD_RUNNING.json',status)
        runtime=root/'runtime';runtime.mkdir(exist_ok=True)
        os.environ.update(HF_HOME=str(runtime/'hf-cache'),TMPDIR=str(runtime),PIP_CACHE_DIR=str(runtime/'pip-cache'),
            PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4')
        os.environ.pop('HF_HUB_OFFLINE',None);os.environ.pop('TRANSFORMERS_OFFLINE',None)
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
        try:finish_job(root,status,shutdown)
        finally:signal.signal(signal.SIGTERM,previous_sigterm)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--max-seconds',type=int,default=7200)
    p.add_argument('--hourly-cny',type=float,required=True);p.add_argument('--budget-remaining-cny',type=float,required=True)
    p.add_argument('--shutdown-verified',action='store_true');main(p.parse_args())
