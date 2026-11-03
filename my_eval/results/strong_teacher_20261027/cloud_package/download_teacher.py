from pathlib import Path
import json
from huggingface_hub import snapshot_download
root=Path(__file__).resolve().parent
path=snapshot_download(repo_id='Qwen/Qwen2.5-7B-Instruct',revision='a09a35458c702b33eeacc393d103063234e8bc28',local_dir=root/'Qwen2.5-7B-Instruct',allow_patterns=['*.safetensors','*.json','*.txt','LICENSE'],max_workers=4)
(root/'download_manifest.json').write_text(json.dumps(dict(repo='Qwen/Qwen2.5-7B-Instruct',revision='a09a35458c702b33eeacc393d103063234e8bc28',path=str(path)),indent=2)+'\n',encoding='utf-8')
print('Pinned teacher download complete',flush=True)
