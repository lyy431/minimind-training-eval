"""Focused regression tests for spend limits, grading and paired comparisons."""
from pathlib import Path
import subprocess
import tempfile
import time
from unittest.mock import Mock,patch
import strong_teacher_common as e
e.configure_local()
import cloud_strong_teacher as cloud
from analyse_strong_teacher import paired,score
from review_strong_teacher import content_key

def main():
    checks={}
    with tempfile.TemporaryDirectory(prefix='teacher_checks_',dir=e.P/'my_eval/runtime/strong_teacher') as folder:
        log=Path(folder)/'child.log'
        with patch.object(cloud.subprocess,'Popen') as spawn:
            try:cloud.command(['python','unused.py'],time.monotonic()+80,log)
            except TimeoutError:pass
            else:raise AssertionError('An almost exhausted deadline must not start a process')
            spawn.assert_not_called()
        checks['near_deadline_starts_no_chargeable_job']=True
        fake=Mock(pid=12345);fake.wait.side_effect=[subprocess.TimeoutExpired('fake',1),0]
        with patch.object(cloud.subprocess,'Popen',return_value=fake),patch.object(cloud.os,'killpg',create=True) as kill:
            try:cloud.command(['python','unused.py'],time.monotonic()+120,log)
            except subprocess.TimeoutExpired:pass
            else:raise AssertionError('Timeout must be propagated after terminating the child')
            kill.assert_called_once_with(12345,cloud.signal.SIGTERM)
        checks['deadline_terminates_child_group']=True
    task=dict(messages=[dict(role='user',content='只输出编号R-24')],exact_answer='R-24')
    for text,eligible in [('R-24，另外还有T-36',True),('R-24',False)]:
        row=dict(final=text,eligible=eligible,failure=None)
        try:score(task,row,{content_key(task,row):dict(score=2)})
        except AssertionError:pass
        else:raise AssertionError('Extra output or unfinished output cannot be fully usable')
    checks['whole_answer_and_completion_required']=True
    result=paired([1.,3.,2.],[1.,2.,3.],lower_is_better=True)
    assert result['n']==3 and result['better']==1 and result['worse']==1 and result['same']==1 and abs(result['difference'])<1e-12
    same=paired([.5,1.,1.5],[.5,1.,1.5]);assert same['ci95']==[0.,0.] and same['se']==0
    checks['paired_direction_and_identity_comparison_correct']=True
    e.save('behavior_tests.json',dict(passed=True,checks=checks,shutdown_executed=False,network_used=False))
    print(e.read(e.OUT/'behavior_tests.json'))

if __name__=='__main__':main()
