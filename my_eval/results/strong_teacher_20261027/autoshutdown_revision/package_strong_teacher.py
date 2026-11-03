"""Create a minimal cloud upload: training questions only, no private held-out data."""
import shutil
import zipfile
import strong_teacher_common as e

def main():
    plan=e.read(e.OUT/'plan.json');root=e.OUT/'cloud_package';root.mkdir(exist_ok=True)
    for name in ['generate_strong_teacher.py','cloud_strong_teacher.py']:
        shutil.copy2(e.P/'my_eval/scripts'/name,root/name)
    shutil.copy2(e.OUT/'teacher_requests.json',root/'teacher_requests.json')
    (root/'teacher_settings.json').write_text(e.json.dumps(plan['teacher_generation'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    download=f'''from pathlib import Path
import json
from huggingface_hub import snapshot_download
root=Path(__file__).resolve().parent
path=snapshot_download(repo_id={e.STRONG_REPO!r},revision={e.STRONG_REVISION!r},local_dir=root/'Qwen2.5-7B-Instruct',allow_patterns=['*.safetensors','*.json','*.txt','LICENSE'],max_workers=4)
(root/'download_manifest.json').write_text(json.dumps(dict(repo={e.STRONG_REPO!r},revision={e.STRONG_REVISION!r},path=str(path)),indent=2)+'\\n',encoding='utf-8')
print('Pinned teacher download complete',flush=True)
'''
    (root/'download_teacher.py').write_text(download,encoding='utf-8',newline='\n')
    hashes={p.name:e.sha(p) for p in root.iterdir() if p.is_file() and p.name!='package_manifest.json'}
    (root/'package_manifest.json').write_text(e.json.dumps(dict(files_sha256=hashes,only_training_questions=True,no_local_weights=True,no_secrets=True),indent=2)+'\n',encoding='utf-8',newline='\n')
    archive=e.OUT/'strong_teacher_cloud.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.iterdir()):
            if p.is_file():z.write(p,'strong_teacher_20261027/'+p.name)
    e.save('cloud_package_manifest.json',dict(zip=str(archive),sha256=e.sha(archive),bytes=archive.stat().st_size,files=hashes,uploaded=False,executed=False))
    print(archive,archive.stat().st_size,'bytes')

if __name__=='__main__':main()
