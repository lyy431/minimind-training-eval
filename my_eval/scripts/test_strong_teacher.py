"""Scientific-input checks before cloud or student training; no GPU required."""
import ast
from collections import Counter
import json
import runpy
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch
import zipfile
import strong_teacher_common as e

def main():
    plan=e.read(e.OUT/'plan.json');checks={}
    for name,h in plan['frozen_inputs'].items():assert e.sha(e.OUT/name)==h,name
    assert e.sha(e.P/f'out/{e.BASE}_768.pth')==e.BASE_SHA;checks['v4_and_predeclared_inputs_unchanged']=True
    data={s:e.read(e.OUT/('train_items.json' if s=='train' else s+'_tasks.json')) for s in ['train','dev','final']}
    question_sets={s:{e.norm(t['question']) for t in rr} for s,rr in data.items()}
    assert [len(data[s]) for s in ['train','dev','final']]==[1000,48,96]
    for s in data:assert len(question_sets[s])==len(data[s])
    assert not question_sets['train']&(question_sets['dev']|question_sets['final']) and not question_sets['dev']&question_sets['final']
    assert all(t['language'] in ['zh','en'] for rr in data.values() for t in rr);checks['fresh_splits_unique_explicit_language']=True
    requests=e.read(e.OUT/'teacher_requests.json');byid={t['id']:t for t in data['train']}
    assert set(r['id'] for r in requests)==set(byid)
    for r in requests:
        assert set(r)=={'id','question','messages','language','category'}
        assert r['messages']==[dict(role='user',content=r['question'])]
        assert r['question']==byid[r['id']]['question']
    checks['teacher_receives_no_reference_or_eval_prompts']=True
    retrieval=[t for t in data['train'] if t['category']=='retrieval']
    for k in range(10):
        rr=[t for t in retrieval if t['format_id']==k]
        assert len(rr)==40 and Counter(t['target_position'] for t in rr)=={0:20,1:20}
        assert Counter(t['fact_order'] for t in rr)=={0:20,1:20}
        en=[t for t in rr if t['language']=='en'];assert len(en)==8 and Counter(t['target_position'] for t in en)=={0:4,1:4}
    checks['retrieval_positions_and_language_not_confounded']=True
    held=question_sets['dev']|question_sets['final']
    for name in ['retention_math640.json','retention_kl600.json','retention_identity240.json']:
        for r in e.read(e.OUT/name):
            assert all(e.norm(m['content']) not in held for m in r['messages'] if m['role']=='user')
    checks['retention_has_no_new_dev_final_questions']=True
    root=e.OUT/'cloud_package';manifest=e.read(root/'package_manifest.json')
    for name,h in manifest['files_sha256'].items():assert e.sha(root/name)==h
    with zipfile.ZipFile(e.OUT/'strong_teacher_cloud.zip') as z:
        assert all(not any(x in name for x in ['final_tasks','dev_tasks','train_items','retention','tokenized','.pth','.safetensors']) for name in z.namelist())
        uploaded=json.loads(z.read('strong_teacher_20261027/teacher_requests.json'));assert uploaded==requests
    checks['cloud_archive_minimal_and_hash_validated']=True
    for f in (e.P/'my_eval/scripts').glob('*strong_teacher*.py'):ast.parse(f.read_text(encoding='utf-8'))
    for f in root.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
    checks['python_sources_parse']=True
    runtime=e.P/'my_eval/runtime/strong_teacher';runtime.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='download_manifest_check_',dir=runtime) as folder:
        testroot=e.Path(folder);script=testroot/'download_teacher.py'
        script.write_text((root/'download_teacher.py').read_text(encoding='utf-8'),encoding='utf-8')
        fake=ModuleType('huggingface_hub');fake.snapshot_download=lambda **kwargs:str(testroot/'fake_model')
        with patch.dict(sys.modules,{'huggingface_hub':fake}):runpy.run_path(str(script))
        downloaded=e.read(testroot/'download_manifest.json')
        assert downloaded['repo']==e.STRONG_REPO and downloaded['revision']==e.STRONG_REVISION
    checks['download_manifest_is_valid_json_without_network']=True
    old=e.read(e.P/'my_eval/results/chat_repair_20261027/final_lock.json')
    for name,h in old['files_sha256'].items():assert e.sha(e.P/name)==h,name
    checks['previous_published_experiment_unchanged']=True
    e.save('preflight_tests.json',dict(passed=all(checks.values()),checks=checks,cloud_runtime_tested=False,student_training_tested=False))
    print(e.read(e.OUT/'preflight_tests.json'))

if __name__=='__main__':main()
