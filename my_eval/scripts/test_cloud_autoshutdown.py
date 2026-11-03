"""Exercise shutdown paths using mocks only. Never issue a real shutdown."""
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock,patch
import zipfile
import strong_teacher_common as e
import cloud_strong_teacher as cloud

def main():
    runtime=e.P/'my_eval/runtime/strong_teacher';runtime.mkdir(parents=True,exist_ok=True)
    checks={}
    args=SimpleNamespace(max_seconds=6000,hourly_cny=2.08,budget_remaining_cny=5,shutdown_verified=True)
    if cloud.sys.platform!='linux':
        with patch.object(cloud.subprocess,'run') as shutdown:
            try:cloud.main(args)
            except AssertionError:pass
            else:raise AssertionError('Local Windows execution must be refused')
            shutdown.assert_not_called()
        checks['local_windows_never_shuts_down']=True
    original_is_dir=Path.is_dir;original_is_file=Path.is_file
    def is_dir(path):return True if path.as_posix().endswith('/root/autodl-tmp') else original_is_dir(path)
    def is_file(path):return True if path.as_posix()=='/usr/bin/shutdown' else original_is_file(path)
    for case in ['success','error','timeout','status_write_failure','archive_failure','shutdown_failure']:
        with tempfile.TemporaryDirectory(prefix='shutdown_mock_',dir=runtime) as folder:
            root=Path(folder)
            (root/'package_manifest.json').write_text('{"files_sha256":{}}',encoding='utf-8')
            (root/'teacher_strong.jsonl').write_text('{"id":"fake_fixture","answer":"fixture"}\n',encoding='utf-8')
            (root/'model.safetensors').write_bytes(b'NOT_FOR_RESULT_ARCHIVE')
            def work(command,deadline,log):
                if case=='error':raise RuntimeError('Simulated job error')
                if case=='timeout':raise subprocess.TimeoutExpired('simulated_job',1)
                if '--arm' in command:
                    pilot='--limit' in command
                    cloud.write_json(root/'teacher_strong.runtime.json',dict(recorded=32 if pilot else 1000,
                        completed=True,generation_seconds=1.,load_seconds=0.))
            result=Mock(returncode=1 if case=='shutdown_failure' else 0,stdout='',stderr='simulated refusal' if case=='shutdown_failure' else '')
            with patch.object(cloud,'__file__',str(root/'cloud_strong_teacher.py')),\
                 patch.object(cloud.sys,'platform','linux'),\
                 patch.object(Path,'is_dir',is_dir),\
                 patch.object(Path,'is_file',is_file),\
                 patch.object(Path,'is_relative_to',return_value=True),\
                 patch.object(cloud.os,'access',return_value=True),\
                 patch.object(cloud.signal,'signal'),\
                 patch.object(cloud.shutil,'disk_usage',return_value=SimpleNamespace(free=30*2**30)),\
                 patch.object(cloud,'command',side_effect=work),\
                 patch.object(cloud.subprocess,'run',return_value=result) as shutdown:
                error=None
                try:
                    if case=='status_write_failure':
                        with patch.object(cloud,'write_json',side_effect=OSError('Simulated full disk')):cloud.main(args)
                    elif case=='archive_failure':
                        with patch.object(cloud,'bundle_results',side_effect=OSError('Simulated archive error')):cloud.main(args)
                    else:cloud.main(args)
                except (RuntimeError,subprocess.TimeoutExpired,OSError) as exc:error=exc
                shutdown.assert_called_once_with(['/usr/bin/shutdown'],capture_output=True,text=True,timeout=15)
                assert (error is not None)==(case in ['error','timeout','status_write_failure']),(case,error)
            if case not in ['status_write_failure','archive_failure']:
                with zipfile.ZipFile(root/'strong_teacher_results.zip') as archive:
                    names=archive.namelist()
                    assert 'teacher_strong.jsonl' in names and 'cloud_status.json' in names and 'results_manifest.json' in names
                    assert not any(name.endswith('.safetensors') for name in names)
                    state=json.loads(archive.read('cloud_status.json'))
                    assert state['completed']==(case not in ['error','timeout'])
                    if case=='shutdown_failure':assert 'SHUTDOWN_FAILED.txt' in names
            checks[case+'_still_attempts_instance_shutdown']=True
    try:cloud.terminate(cloud.signal.SIGTERM,None)
    except SystemExit as exc:assert exc.code==128+cloud.signal.SIGTERM
    else:raise AssertionError('SIGTERM must enter the normal exception/finally path')
    checks['ordinary_termination_enters_finally_path']=True
    e.save('auto_shutdown_tests.json',dict(passed=all(checks.values()),checks=checks,actual_shutdown_executed=False,
        paid_cloud_started=False,cloud_integration_tested=False))
    print(e.read(e.OUT/'auto_shutdown_tests.json'))

if __name__=='__main__':main()
