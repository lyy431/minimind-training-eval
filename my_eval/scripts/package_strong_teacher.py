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
    instructions='''MiniMind 7B老师生成：运行后自动关机\n\n上传位置：/root/autodl-tmp/strong_teacher_cloud.zip\n解压后的目录：/root/autodl-tmp/strong_teacher_20261027/\n\n运行前：核实实例还是RTX4090、价格2.08元/小时、账户余额至少5元，\n并在平台设置从本次开机起两小时内自动关机。旧的关机时间不能复用。\n下面命令把任务时间限制为100分钟，任务内预计租金最多约3.47元；\n开机到启动任务之间的费用和关机延迟另计，本轮总预算仍为30元。\n\n在JupyterLab新建终端，解压后执行：\ncd /root/autodl-tmp/strong_teacher_20261027\npython -u cloud_strong_teacher.py --hourly-cny 2.08 --budget-remaining-cny 5 --max-seconds 6000 --shutdown-verified\n\n脚本会先下载固定版本老师，再用32条测速，然后生成1000条答案。\n完成、报错、超时或收到普通终止信号后，均尝试先保存结果再关机。\n结果文件：strong_teacher_results.zip；包含答案、设置、摘要和日志，不含15GB老师权重。\n\n自动关机会断开JupyterLab；稍后可使用平台无卡模式开机下载这个结果包。\n如失败，部分结果和错误日志也会保留在同一目录，结果包也可以发回分析。\n若已有CLOUD_RUNNING.json，请先核对之前任务状态，不要直接重复启动。\n\n脚本无法处理服务器断电、进程被强制杀死或平台关机命令失效，\n所以平台定时关机仍需设置，并在结束后确认实例显示“已关机”。\n'''
    (root/'运行说明.txt').write_text(instructions,encoding='utf-8',newline='\n')
    hashes={p.name:e.sha(p) for p in root.iterdir() if p.is_file() and p.name!='package_manifest.json'}
    (root/'package_manifest.json').write_text(e.json.dumps(dict(files_sha256=hashes,only_training_questions=True,no_local_weights=True,no_secrets=True),indent=2)+'\n',encoding='utf-8',newline='\n')
    archive=e.OUT/'strong_teacher_cloud.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.iterdir()):
            if p.is_file():z.write(p,'strong_teacher_20261027/'+p.name)
    e.save('cloud_package_manifest.json',dict(zip=str(archive),sha256=e.sha(archive),bytes=archive.stat().st_size,files=hashes,uploaded=False,executed=False))
    print(archive,archive.stat().st_size,'bytes')

if __name__=='__main__':main()
