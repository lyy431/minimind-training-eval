#!/usr/bin/env bash
# MiniMind 云服务器一键训练脚本（AutoDL 等 Linux GPU 实例）
#
# 用法：把 minimind-master.zip 和本脚本上传到同一个目录（推荐 /root/autodl-tmp），然后运行
#     bash cloud_train.sh
# 脚本会在 tmux 后台会话里依次完成：解压 → 检查环境 → 装依赖 → 下数据 → 预训练 → SFT →（可选）GRPO → 测试对话。
# 中途断了就再运行一次同样的命令：已完成的阶段会跳过，未完成的阶段从最近的检查点续训。
#
# 可选环境变量：
#     EPOCHS=1          预训练 / SFT 各训练几轮（默认 1，作者原版默认 2）
#     TRAIN_ARGS="..."  额外传给预训练和 SFT 脚本的参数，例如 "--batch_size 64"
#     GRPO=1            SFT 之后再跑 GRPO 强化学习（会额外下载 rlaif.jsonl 和 3.4GB 的奖励模型）
#     GRPO_SAMPLES=1000 GRPO 用多少条 prompt（默认 1000 条 = 500 步；all = 全部 19502 条，耗时约为默认的 20 倍）
#     GRPO_ARGS="..."   额外传给 train_grpo.py 的参数，例如 "--loss_type grpo"（默认是 cispo）
#     GRPO_NAME=grpo    这一轮 GRPO 的名字：权重 out/<名字>_768.pth、完成标记、检查点、日志都按它区分。
#                       换个名字（如 grpo5k）就会从 full_sft 重新开一轮，不会续上或覆盖之前那轮
#     AGENT=1           再跑工具调用强化学习（train_agent.py + agent_rl_math.jsonl：调计算器算数学题，按答案对错给分）
#     AGENT_SAMPLES=3000  用多少道题（默认 3000 道 = 1500 步；all = 全部；固定留出的 200 道评测题永远不参与训练）
#     AGENT_EXCLUDE=0   先排除"AGENT_SAMPLES=这个数"时会抽到的那批题（续训时用，例如 3000 = 排除上一轮用过的题）
#     AGENT_ARGS="..."  额外传给 train_agent.py 的参数，例如 "--loss_type grpo"
#     AGENT_NAME=agent  这一轮的名字，作用同 GRPO_NAME
#     DISTILL=1         知识蒸馏：MoE 的 SFT 模型（full_sft_768_moe.pth）当老师，从稠密预训练权重开始，
#                       用和稠密 SFT 完全相同的数据和超参训练稠密学生（损失 = 0.5*CE + 0.5*T²*KL），原 full_sft 即对照组
#     DISTILL_NAME=full_dist  学生权重名（out/<名字>_768.pth）
#     DISTILL_ARGS="..." 额外传给 train_distillation.py 的参数，例如 "--alpha 0.3 --temperature 2.0"
#     MOE=1             用 MoE 结构（4 个专家、每个 token 选 1 个；198M 总参数 / 64M 激活）。每个阶段都加 --use_moe 1，
#                       权重自动带 _moe 后缀，完成标记和日志也带 _moe，和稠密版互不干扰
#     AUTO_SHUTDOWN=1   全部结束（或中途出错）后自动关机，停止计费
#     PYTHON=python     使用的 Python 解释器
# 例：EPOCHS=2 AUTO_SHUTDOWN=1 bash cloud_train.sh
#     GRPO=1 AUTO_SHUTDOWN=1 bash cloud_train.sh
#     AGENT=1 AGENT_ARGS="--loss_type grpo" AUTO_SHUTDOWN=1 bash cloud_train.sh

set -euo pipefail

EPOCHS="${EPOCHS:-1}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
GRPO="${GRPO:-0}"
GRPO_SAMPLES="${GRPO_SAMPLES:-1000}"
GRPO_ARGS="${GRPO_ARGS:-}"
GRPO_NAME="${GRPO_NAME:-grpo}"
AGENT="${AGENT:-0}"
AGENT_SAMPLES="${AGENT_SAMPLES:-3000}"
AGENT_ARGS="${AGENT_ARGS:-}"
AGENT_NAME="${AGENT_NAME:-agent}"
AGENT_EXCLUDE="${AGENT_EXCLUDE:-0}"
MOE="${MOE:-0}"
DISTILL="${DISTILL:-0}"
DISTILL_NAME="${DISTILL_NAME:-full_dist}"
DISTILL_ARGS="${DISTILL_ARGS:-}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
PYTHON="${PYTHON:-python}"
SESSION=minimind
DATA_FILES=(pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl)
[[ "$GRPO" == "1" ]] && DATA_FILES+=(rlaif.jsonl)
[[ "$AGENT" == "1" ]] && DATA_FILES+=(agent_rl_math.jsonl)
REWARD_MODEL_REPO=Shanghai_AI_Laboratory/internlm2-1_8b-reward

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$WORK_DIR/$(basename "${BASH_SOURCE[0]}")"
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "❌ $*"; exit 1; }

# ========== 0. 放进 tmux 后台运行，关掉网页终端也不会中断 ==========
if [[ "${1:-}" != "--inside" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        log "未找到 tmux，尝试安装..."
        (apt-get update -qq && apt-get install -y -qq tmux) >/dev/null 2>&1 || true
    fi
    printf -v inner 'EPOCHS=%q TRAIN_ARGS=%q GRPO=%q GRPO_SAMPLES=%q GRPO_ARGS=%q GRPO_NAME=%q AGENT=%q AGENT_SAMPLES=%q AGENT_ARGS=%q AGENT_NAME=%q AGENT_EXCLUDE=%q MOE=%q DISTILL=%q DISTILL_NAME=%q DISTILL_ARGS=%q AUTO_SHUTDOWN=%q PYTHON=%q bash %q --inside' \
        "$EPOCHS" "$TRAIN_ARGS" "$GRPO" "$GRPO_SAMPLES" "$GRPO_ARGS" "$GRPO_NAME" \
        "$AGENT" "$AGENT_SAMPLES" "$AGENT_ARGS" "$AGENT_NAME" "$AGENT_EXCLUDE" "$MOE" \
        "$DISTILL" "$DISTILL_NAME" "$DISTILL_ARGS" "$AUTO_SHUTDOWN" "$PYTHON" "$SCRIPT_PATH"
    if command -v tmux >/dev/null 2>&1; then
        tmux has-session -t "$SESSION" 2>/dev/null && die "tmux 会话 '$SESSION' 已经在运行，查看进度：tmux attach -t $SESSION"
        tmux new-session -d -s "$SESSION" "$inner 2>&1 | tee -a '$LOG_DIR/all.log'; echo; echo '脚本已结束，按回车关闭窗口'; read"
        log "已在 tmux 后台启动。"
        log "查看实时进度：tmux attach -t $SESSION    （离开但不停止：先按 Ctrl+B，再按 D）"
    else
        nohup bash -c "$inner" >>"$LOG_DIR/all.log" 2>&1 &
        log "tmux 不可用，已用 nohup 在后台启动。"
        log "查看实时进度：tail -f $LOG_DIR/all.log    （按 Ctrl+C 只退出查看，不会停止训练）"
    fi
    log "全部日志在：$LOG_DIR"
    exit 0
fi

# ========== 以下在后台会话里执行 ==========
T_START=$(date +%s)
finish() {
    local code=$?
    if [[ $code -eq 0 ]]; then
        log "🎉 全部完成，总用时 $(( ($(date +%s) - T_START) / 60 )) 分钟"
    else
        log "❌ 中途出错退出（退出码 $code），日志在 $LOG_DIR。修好后重新运行 bash $SCRIPT_PATH 即可续跑"
    fi
    if [[ "$AUTO_SHUTDOWN" == "1" && "$(uname -s)" == "Linux" ]]; then
        log "AUTO_SHUTDOWN=1：60 秒后自动关机（按 Ctrl+C 取消）..."
        sleep 60 && shutdown
    fi
}
trap finish EXIT

# ---------- 1. 找到项目目录，没有就解压 ----------
find_project() {
    local d
    for d in "$WORK_DIR" "$WORK_DIR/minimind-master"; do
        if [[ -f "$d/model/model_minimind.py" && -d "$d/trainer" ]]; then echo "$d"; return 0; fi
    done
    return 1
}
if ! PROJECT="$(find_project)"; then
    zips=("$WORK_DIR"/minimind*.zip)
    [[ -f "${zips[0]}" ]] || die "在 $WORK_DIR 里既没有项目目录，也没有 minimind*.zip，请把压缩包和本脚本放在同一目录"
    log "解压 ${zips[0]} ..."
    "$PYTHON" -m zipfile -e "${zips[0]}" "$WORK_DIR"
    PROJECT="$(find_project)" || die "解压后没找到项目文件（model/model_minimind.py），请检查压缩包"
fi
log "项目目录：$PROJECT"

# ---------- 2. 检查 Python / GPU / 磁盘 ----------
"$PYTHON" - <<'EOF' || die "环境检查没通过（原因见上一行）"
import sys
v = sys.version_info[:2]
if not ((3, 10) <= v <= (3, 12)):
    sys.exit(f"当前是 Python {v[0]}.{v[1]}：requirements.txt 锁定的 numpy 1.26.4 需要 Python 3.10–3.12，请换镜像")
try:
    import torch
except ImportError:
    sys.exit("没装 PyTorch：请选择预装 PyTorch 2.x 的镜像")
if not torch.cuda.is_available():
    sys.exit("PyTorch 检测不到 GPU：请确认实例是有卡模式开机，且镜像是 CUDA 版 PyTorch")
p = torch.cuda.get_device_properties(0)
print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | GPU {p.name} {p.total_memory / 1024**3:.0f}GB | bf16 {torch.cuda.is_bf16_supported()}")
EOF

DTYPE_ARG=""
if ! "$PYTHON" -c "import sys, torch; sys.exit(0 if torch.cuda.is_bf16_supported() else 1)"; then
    DTYPE_ARG="--dtype float16"
    log "这张卡不支持 bf16，改用 float16 混合精度"
fi

free_gb=$(df -Pk "$WORK_DIR" | awk 'NR==2 {print int($4 / 1024 / 1024)}')
(( free_gb >= 12 )) || die "$WORK_DIR 只剩 ${free_gb}GB，数据+缓存+检查点至少要 12GB，请放到数据盘（如 /root/autodl-tmp）"

# 数据集缓存、模型缓存都放在工作目录（数据盘），不占系统盘
export HF_HOME="${HF_HOME:-$WORK_DIR/hf-cache}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$WORK_DIR/modelscope-cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # 减少显存碎片导致的 OOM

# ---------- 3. 安装依赖（锁住镜像自带的 CUDA 版 torch，防止被换掉） ----------
TORCH_VER=$("$PYTHON" -c "import torch; print(torch.__version__)")
echo "torch==$TORCH_VER" >"$LOG_DIR/keep_torch.txt"
log "安装依赖（保持 torch==$TORCH_VER 不变，可能要几分钟）..."
# requirements.txt 锁定 numpy==1.26.4，而 pyarrow>=26 导入时要求 numpy>=2（它没声明这个依赖，pip 发现不了），所以锁在 26 以下
"$PYTHON" -m pip install -q -r "$PROJECT/requirements.txt" "pyarrow<26" -c "$LOG_DIR/keep_torch.txt" 2>&1 | tee "$LOG_DIR/pip.log" \
    || die "依赖安装失败，详见 $LOG_DIR/pip.log"
"$PYTHON" -c "import numpy, pyarrow, datasets, transformers, torch" \
    || die "依赖装完了但导入失败（报错见上方），多半是镜像自带的包和 requirements.txt 的版本冲突"
log "依赖安装完成"

# ---------- 4. 下载数据 ----------
cd "$PROJECT"
need=()
for f in "${DATA_FILES[@]}"; do
    if [[ -s "dataset/$f" ]]; then log "数据已存在，跳过：$f"; else need+=("$f"); fi
done
if (( ${#need[@]} )); then
    log "从 ModelScope 下载：${need[*]} ..."
    "$PYTHON" -m modelscope.cli.cli download --dataset gongjy/minimind_dataset "${need[@]}" --local_dir ./dataset
fi
for f in "${DATA_FILES[@]}"; do
    [[ -s "dataset/$f" ]] || die "数据文件缺失：dataset/$f"
done

if [[ "$DISTILL" == "1" ]]; then  # 开训前就检查，免得先跑完别的阶段才报错
    [[ "$MOE" == "1" ]] && die "DISTILL=1 不能和 MOE=1 一起用（蒸馏的学生/老师结构由 train_distillation.py 自己的参数决定）"
    [[ -f "$PROJECT/out/full_sft_768_moe.pth" ]] || die "蒸馏需要 MoE 老师 out/full_sft_768_moe.pth，先用 MOE=1 跑完预训练和 SFT"
fi

if [[ "$GRPO" == "1" || "$AGENT" == "1" ]]; then
    # 奖励模型放在工作目录（数据盘），开训前先下好，免得跑完 SFT 才发现下载失败
    REWARD_MODEL_DIR="$WORK_DIR/internlm2-1_8b-reward"
    if [[ -f "$REWARD_MODEL_DIR/model.safetensors.index.json" && -f "$REWARD_MODEL_DIR/model-00002-of-00002.safetensors" ]]; then
        log "奖励模型已存在，跳过：$REWARD_MODEL_DIR"
    else
        log "从 ModelScope 下载奖励模型 $REWARD_MODEL_REPO（3.4GB）..."
        "$PYTHON" -m modelscope.cli.cli download --model "$REWARD_MODEL_REPO" --local_dir "$REWARD_MODEL_DIR"
    fi
fi

if [[ "$AGENT" == "1" ]]; then
    # 固定留出 200 道评测题（和本地评测用的同一份切分）；可选先排除上一轮用过的题，再从剩下的里抽 AGENT_SAMPLES 道训练
    AGENT_DATA="$PROJECT/dataset/agent_math_train_${AGENT_SAMPLES}_excl${AGENT_EXCLUDE}.jsonl"
    "$PYTHON" -c "import hashlib, random, sys
src, train_out, eval_out, n, excl = sys.argv[1:6]
lines = [l.rstrip('\n') + '\n' for l in open(src, encoding='utf-8') if l.strip()]
held = set(random.Random(123).sample(range(len(lines)), 200))
evals = [l for i, l in enumerate(lines) if i in held]
open(eval_out, 'w', encoding='utf-8', newline='\n').writelines(evals)
train = [l for i, l in enumerate(lines) if i not in held]
if int(excl):
    random.seed(42)  # 和 AGENT_SAMPLES=excl 那一轮的抽样完全相同，正好排除掉那批题
    used = set(random.sample(train, int(excl)))
    train = [l for l in train if l not in used]
if n != 'all':
    random.seed(42)
    train = random.sample(train, min(int(n), len(train)))
eval_set = set(evals)
leaked = sum(l in eval_set for l in train)
train = [l for l in train if l not in eval_set]  # 数据里有重复行：评测题的副本可能落在训练集里，按内容剔除
open(train_out, 'w', encoding='utf-8', newline='\n').writelines(train)
digest = lambda xs: hashlib.sha256(''.join(xs).encode()).hexdigest()[:16]
print(f'评测集 {len(evals)} 道（校验 {digest(evals)}）| 训练 {len(train)} 道（校验 {digest(train)}，已剔除 {leaked} 道评测题副本）')" \
        "$PROJECT/dataset/agent_rl_math.jsonl" "$AGENT_DATA" "$PROJECT/dataset/agent_math_eval200.jsonl" "$AGENT_SAMPLES" "$AGENT_EXCLUDE"
    log "AGENT 数据：$AGENT_DATA（$(wc -l <"$AGENT_DATA") 道题；评测集 dataset/agent_math_eval200.jsonl 不参与训练）"
fi

if [[ "$GRPO" == "1" ]]; then
    if [[ "$GRPO_SAMPLES" == "all" ]]; then
        GRPO_DATA="$PROJECT/dataset/rlaif.jsonl"
    else
        GRPO_DATA="$PROJECT/dataset/rlaif_${GRPO_SAMPLES}.jsonl"
        "$PYTHON" -c "import random, sys
lines = [l.rstrip('\n') + '\n' for l in open(sys.argv[1], encoding='utf-8') if l.strip()]  # 保证每行都以换行结尾
random.seed(42)
open(sys.argv[2], 'w', encoding='utf-8', newline='\n').writelines(random.sample(lines, min(int(sys.argv[3]), len(lines))))" \
            "$PROJECT/dataset/rlaif.jsonl" "$GRPO_DATA" "$GRPO_SAMPLES"
    fi
    log "GRPO 数据：$GRPO_DATA（$(wc -l <"$GRPO_DATA") 条 prompt）"
fi

# ---------- 5. 预训练 → SFT →（可选）GRPO ----------
mkdir -p "$PROJECT/out"
MOE_SUFFIX="" MOE_ARG=""
if [[ "$MOE" == "1" ]]; then MOE_SUFFIX="_moe" MOE_ARG="--use_moe 1"; fi
run_stage() {  # run_stage <名字> <脚本> <传给脚本的参数...>；MOE=1 时名字自动加 _moe（标记、日志都按它区分）
    local name=$1$MOE_SUFFIX script=$2 t0
    local marker="$PROJECT/out/.${name}_done"
    shift 2
    set -- $MOE_ARG "$@"  # shellcheck disable=SC2086  # 写在最前面，用户参数在后，可覆盖
    if [[ -f "$marker" ]]; then
        log "⏭  $name 已完成，跳过（想重跑就删掉 $marker）"
        return
    fi
    log "▶  开始 $name（参数：$*）..."
    t0=$(date +%s)
    cd "$PROJECT/trainer"
    # shellcheck disable=SC2086  # DTYPE_ARG 需要按空格拆成多个参数
    "$PYTHON" -u "$script" --from_resume 1 $DTYPE_ARG "$@" 2>&1 | tee -a "$LOG_DIR/$name.log"
    touch "$marker"
    log "✅ $name 完成，用时 $(( ($(date +%s) - t0) / 60 )) 分钟"
}
# shellcheck disable=SC2086  # TRAIN_ARGS / GRPO_ARGS 需要按空格拆成多个参数
run_stage pretrain train_pretrain.py --epochs "$EPOCHS" $TRAIN_ARGS
# shellcheck disable=SC2086
run_stage sft train_full_sft.py --epochs "$EPOCHS" $TRAIN_ARGS
if [[ "$DISTILL" == "1" ]]; then
    [[ -f "$PROJECT/out/pretrain_768.pth" ]] || die "蒸馏需要学生起点 out/pretrain_768.pth（稠密预训练）"
    # 学生：稠密预训练权重；老师：MoE SFT。数据和超参与 train_full_sft.py 默认值一致（batch 16、768 token、lr 1e-5、1 轮），
    # 所以原来的 full_sft 就是"不用老师"的对照组。默认每 100 步存一次（约 1GB）太频繁，改成 1000。
    # 老师在 autocast 外面用 fp32 前向，打开 TF32 后在本机快约 15%
    export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
    # shellcheck disable=SC2086
    run_stage "$DISTILL_NAME" train_distillation.py --epochs 1 --from_student_weight pretrain --student_use_moe 0 \
        --from_teacher_weight full_sft --teacher_use_moe 1 --batch_size 16 --max_seq_len 768 --learning_rate 1e-5 \
        --save_weight "$DISTILL_NAME" --save_interval 1000 $DISTILL_ARGS
    unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
fi
if [[ "$GRPO" == "1" ]]; then
    # 默认每 10 步存一次（每次约 1GB），改成 100 步；GRPO_ARGS 写在后面，可以覆盖这里的设置
    # max_gen_len 默认 1024 时序列最长 768+1024，带 padding 的注意力走非 flash 路径，fp32 分数矩阵在 24GB 卡上会 OOM；
    # 512 时显存约减半，也和奖励函数"回答 800 字以内"的长度分一致
    # shellcheck disable=SC2086
    run_stage "$GRPO_NAME" train_grpo.py --epochs 1 --data_path "$GRPO_DATA" --reward_model_path "$REWARD_MODEL_DIR" \
        --save_weight "$GRPO_NAME" --save_interval 100 --max_gen_len 512 $GRPO_ARGS
fi
if [[ "$AGENT" == "1" ]]; then
    # max_gen_len 默认 768（每轮），3 轮 + 工具说明可达 2500 token；一条跑长就会把整批 padding 拉长而 OOM。
    # 工具调用和最终回答通常不到 100 token，256 足够
    # shellcheck disable=SC2086
    run_stage "$AGENT_NAME" train_agent.py --epochs 1 --data_path "$AGENT_DATA" --reward_model_path "$REWARD_MODEL_DIR" \
        --save_weight "$AGENT_NAME" --save_interval 100 --max_gen_len 256 $AGENT_ARGS
fi

# ---------- 6. 用训练好的模型试答几个问题 ----------
cd "$PROJECT"
if [[ "$AGENT" == "1" && "$GRPO" != "1" && "$DISTILL" != "1" ]]; then
    log "⏭  工具调用模型不做普通对话测试（eval_llm.py 不调用工具），正确率请用工具调用评测脚本测"
else
    if [[ "$GRPO" == "1" ]]; then
        EVAL_ARGS=(--weight "$GRPO_NAME" --open_thinking 1 --max_new_tokens 400)
        EVAL_LOG="$LOG_DIR/eval_${GRPO_NAME}${MOE_SUFFIX}.log"
    elif [[ "$DISTILL" == "1" ]]; then
        EVAL_ARGS=(--weight "$DISTILL_NAME" --max_new_tokens 120)
        EVAL_LOG="$LOG_DIR/eval_${DISTILL_NAME}.log"
    else
        EVAL_ARGS=(--weight full_sft --max_new_tokens 120)
        EVAL_LOG="$LOG_DIR/eval${MOE_SUFFIX}.log"
    fi
    [[ "$MOE" == "1" ]] && EVAL_ARGS+=(--use_moe 1)
    log "用训练好的模型回答几个测试问题（${EVAL_ARGS[*]}）..."
    echo 0 | "$PYTHON" eval_llm.py "${EVAL_ARGS[@]}" 2>&1 | tee "$EVAL_LOG" \
        || log "⚠ 测试对话失败（不影响已保存的权重），详见 $EVAL_LOG"
fi

log "权重文件如下，把需要的 *.pth（pretrain / full_sft / grpo / agent）下载到本地项目的 out/ 目录："
ls -lh "$PROJECT"/out/*.pth
log "下载完记得去控制台关机，停止计费"
