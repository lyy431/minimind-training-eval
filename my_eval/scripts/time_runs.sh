#!/usr/bin/env bash
# Steps/second of plain SFT vs distillation (with/without TF32) at small batch, same seq len, on the laptop.
cd /d/minimind/minimind-master/trainer
PY=/d/conda/envs/minimind/python.exe
export HF_HOME=D:/conda/hf-cache PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--epochs 1 --batch_size 4 --max_seq_len 768 --num_workers 2 --log_interval 10 --save_interval 1000000"
last_step() { grep -oE "\([0-9]+/[0-9]+\)" "$1" 2>/dev/null | tail -1 | grep -oE "^.[0-9]+" | tr -d '(' ; }
measure() {  # measure <label> <command...>: steps/s between t=50s and t=100s (excludes model/data loading)
    local label=$1; shift
    local log=/d/conda/time_$label.log
    "$@" > "$log" 2>&1 &
    sleep 50; local a; a=$(last_step "$log"); a=${a:-0}
    sleep 50; local b; b=$(last_step "$log"); b=${b:-0}
    powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*timing_*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1
    wait 2>/dev/null
    grep -E "Traceback|Error:" "$log" | head -3
    echo "$label: steps $a -> $b in 50s = $(awk -v a="$a" -v b="$b" 'BEGIN{printf "%.2f", (b-a)/50}') steps/s"
}
measure sft "$PY" -u train_full_sft.py $COMMON --save_weight timing_sft
measure kd "$PY" -u train_distillation.py $COMMON --from_student_weight pretrain --save_weight timing_kd
measure kd_tf32 env TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 "$PY" -u train_distillation.py $COMMON --from_student_weight pretrain --save_weight timing_kd
