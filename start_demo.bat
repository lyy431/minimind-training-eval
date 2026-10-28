@echo off
rem MiniMind web chat demo. Double-click to start; the browser opens http://localhost:8501 when ready.
rem Close this window (or press Ctrl+C) to stop. Models are the folders in scripts\ (made by my_eval\scripts\make_demo_models.py).
set HF_HOME=D:\conda\hf-cache
set PYTHONIOENCODING=utf-8
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TEMP=D:\minimind\minimind-master\my_eval\runtime
set TMP=D:\minimind\minimind-master\my_eval\runtime
if not exist "%TEMP%" mkdir "%TEMP%"
cd /d "%~dp0scripts"
if /i not "%~1"=="nobrowser" start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 180;$i++){try{Invoke-WebRequest http://localhost:8501 -UseBasicParsing -TimeoutSec 1 | Out-Null; Start-Process http://localhost:8501; break}catch{Start-Sleep 1}}"
D:\conda\envs\minimind\python.exe -m streamlit run web_demo.py --server.port 8501 --server.address localhost --server.headless true --browser.gatherUsageStats false --client.toolbarMode minimal
