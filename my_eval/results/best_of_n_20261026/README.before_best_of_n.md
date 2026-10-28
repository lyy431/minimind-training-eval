# MiniMind 实验记录

最新评测更新（2026-10-26）：数学已补充完整回答一致性评分；旧数字包含率保留为历史口径。新题组、复习组、原RL暂不能可靠排出先后，详见[修正报告](results/strict_math_20261026/REPORT.md)。

作者：Yuyang LU　时间：2026-10-20 ～ 2026-10-25　（交接说明见同目录 `HANDOFF.md`，给 AI 助手读的技术档案见 `SUMMARY_FOR_CHATGPT.md`）

基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)，从预训练一路做到 SFT、LoRA、GRPO、工具调用 RL、DPO、MoE、知识蒸馏和"蒸馏 + RL"。
这个文件夹保存了所有**自己写的评测脚本、评测结果和本地训练日志**。项目自带的代码只改了一处：`scripts/web_demo.py` 的工具定义（见 4.10）；另外新增了根目录的 `start_demo.bat` 和 `scripts/` 下的 4 个 demo 模型文件夹。

---

## 1. 一页总结

最新补充（2026-10-25）：待办1、2、3均已完成。身份改进见第10节；新增蒸馏对照和测试同文问题修正见[完整对照实验报告](results/distill_control_20261025/REPORT.md)及第11节。主要数学/聊天比较改为197/29题；下面早期200/30题表保留历史口径。

### 数学工具调用（200 道留出题，每题 2 次）

| 模型 | 怎么来的 | 全对率 | 1 / 2 / 3 个算式 |
|---|---|---|---|
| `full_sft` | 预训练 + SFT（官方流程） | 2.8% | 10 / 0 / 0% |
| `full_dist` | MoE 老师白盒蒸馏的稠密学生 | 1.5% | |
| `full_sft_moe` | MoE 结构的 SFT | 6.2% | |
| `agent` | `full_sft` + 工具调用 RL 1500 步 | 19.0% | 65.5 / 4.2 / 0% |
| `bb_dist` | `full_sft` + Qwen2.5-1.5B 黑盒蒸馏 | 68.5% | 86.4 / 71.9 / 56.7% |
| **`bb_rl`** | **`bb_dist` + 工具调用 RL 1500 步** | **81.8%** | **98.2 / 86.5 / 70.1%** |
| `bb_rl_id` | `bb_rl` + 身份 LoRA（演示小助手，见 4.10） | 73.0% | 85.5 / 70.8 / 67.0% |
| Qwen2.5-1.5B（老师） | — | 80.5%* | |

\* 老师用的是贪心解码、每题 1 次，和学生的测法不同，只能说两者"差不多"。

### 聊天质量（30 个留出问题 × 3 次，用 GRPO 的奖励模型打分，相对 `full_sft`）

| 模型 | 总奖励差 | 其中奖励模型分差 | 结论 |
|---|---|---|---|
| `grpo`（500 步） | −0.06 ± 0.07 | −0.10 ± 0.05 | 没变好 |
| `grpo5k`（2500 步） | **+0.46 ± 0.13** | +0.11 ± 0.08 | 主要靠格式分（思考长度合适、更短、少重复） |
| `dpo` | −0.02 ± 0.06 | −0.04 ± 0.04 | 没变化 |
| `dpo10x`（学习率 ×10） | −0.22 ± 0.10 | −0.15 ± 0.08 | 变差：回答长 30%，更多重复 |
| `full_sft_moe` | −0.34 ± 0.08 | −0.26 ± 0.09 | 变差 |
| `full_dist` | −0.20 ± 0.08 | −0.21 ± 0.08 | 变差，介于对照组和老师之间 |
| `bb_dist` | −0.07 ± 0.08 | −0.03 ± 0.07 | 不变 |
| `bb_rl` | −0.05 ± 0.10 | −0.04 ± 0.09 | 不变 |
| `bb_rl_id` | +0.19 ± 0.11 | +0.06 ± 0.10 | 奖励模型分不变；总奖励的提升来自回答短了一半（但留出 loss 明显变差，见 4.10） |

（± 都是 1 个标准误，按题配对计算。所有数字都可以从 `results/*_compare.json` 重新算出来，2026-10-24 复核过。）

### 最重要的几条结论
1. **小模型自己做 RL，起点太低就学不动**：`full_sft` 几乎答不对（2.8%），RL 1500 步只到 19%，3 个算式的题始终是 0。
2. **换一个强老师做黑盒蒸馏，效果远超 RL**：Qwen 解的 2514 道正确答案，训练 8 分钟就到 68.5%，200 题里比 SFT 好 154 题、差 0 题。
3. **历史阶段先蒸馏再RL取得最高数学分**：原200题81.8%（逐题+13.2 ± 2.4分，好50题、差11题）；表达式结果匹配率91.2% → 97.7%。该阶段比较尚未控制额外训练和新增数据；现已补对照，不能把全部提升视为RL的独有优势，见第11节。
4. **用弱老师（自己的 MoE）白盒蒸馏几乎没用**：留出文本上的 loss 降了一点，但聊天打分和数学都没变好。
5. **训练奖励涨不等于真的变好**：grpo5k 的奖励涨了，但涨的主要是格式分，回答里仍有自信的胡编；dpo10x 偏好准确率最高，生成质量反而变差。
6. **小模型学到的是"格式"而不只是"能力"**：把在 `full_sft` 上训练的 LoRA 直接合并进 `bb_rl`，数学从 81.8% 掉到 8.8%；工具描述换一种措辞掉到 54.5%；工具列表只放 1 个（训练时总是 2 个）掉到约 52%。

---

## 2. 文件夹说明

```
my_eval/
├── README.md               ← 本文件（实验报告）
├── scripts/                ← 自己写的评测/数据/训练辅助脚本
│   └── prompts/            ← eval_llm.py 手动测试用的问题（从标准输入喂进去）
├── results/                ← 每次评测输出的 JSON（每道题、每次采样的完整生成文本都在里面）
└── logs/                   ← 在笔记本上跑的训练/评测日志（云上跑的日志在 ../out/*.log）
```

### 脚本一览

| 脚本 | 作用 | 用法 |
|---|---|---|
| `make_datasets.py` | 从官方数据**原样重建**所有派生数据集，并和 `dataset/` 里的文件逐字节比对 | `make_datasets.py 项目路径 输出目录` |
| `heldout_loss.py` | 通用语言能力：在 300 段留出对话上每个回答 token 的 loss（和第一个模型配对比较） | `heldout_loss.py 项目 权重1,权重2,...` |
| `tool_choice_probe.py` | 把同一份工具列表给模型，看它第一轮回复会不会选对工具（数学/天气/汇率/时间/翻译/换算/不需要工具） | `tool_choice_probe.py 项目 工具json 输出json 权重1,权重2 [每题次数]` |
| `make_demo_models.py` | 把 LoRA 合并进权重；把权重导出成 Hugging Face（Qwen3）格式放进 `scripts/` 给网页 demo 用，并核对导出前后 logits 一致 | `make_demo_models.py 项目 merge 基座 LoRA 新名字` / `make_demo_models.py 项目 export 权重名 文件夹名` |
| `eval_agent_math.py` | 数学工具调用评测：复用 `train_agent.py` 的多轮 rollout 和判分 | `eval_agent_math.py 项目 题目文件 输出json 权重名[,权重名] 题数 每题次数` |
| `compare_agent.py` | 两个模型的数学错误分类 + 逐题配对比较（同一个模型两种测法也行，列名会标 [A]/[B]） | `compare_agent.py 项目 A.json B.json 权重A 权重B` |
| `score_rm.py` | 聊天质量：用 GRPO 的完整奖励（奖励模型 + 格式项）给多个模型打分 | `score_rm.py 项目 奖励模型路径 输出json 权重1,权重2,...` |
| `score_grpo3.py` | 同上，GRPO 那次专用（固定比较 full_sft / grpo / grpo5k） | `score_grpo3.py 项目 奖励模型路径 输出json` |
| `dpo_pref_eval.py` | 留出偏好对上，模型是否更偏向 chosen（DPO 的隐式奖励） | `dpo_pref_eval.py 项目 偏好对文件 权重1,权重2 [cuda] [条数]` |
| `compare_arch.py` | 稠密 vs MoE：留出文本 loss（配对）+ 生成速度 | `compare_arch.py 项目` |
| `expert_usage.py` | MoE 路由分析：每层专家负载、各类 token 去了哪个专家 | `expert_usage.py 项目 权重名 文本文件 [条数]` |
| `compare_kd.py` | 白盒蒸馏：学生/对照/老师的留出 loss、到老师的 KL、生成速度 | `compare_kd.py 项目` |
| `score_lora.py` | LoRA 身份：问身份时答对几次、问别的问题时"漏"出几次 | `score_lora.py 项目 [权重或"权重+LoRA",...]`（默认测 full_sft 和两个 LoRA） |
| `make_mix.py` | 身份数据 ×2 + 200 条不含身份内容的通用对话 | `make_mix.py sft文件 身份文件 输出文件` |
| `teacher_gen.py` | 黑盒蒸馏数据：Qwen 用同样的工具解题，只保留答对的，转成 MiniMind 的 SFT 格式 | `teacher_gen.py 项目 老师路径 题目文件 输出文件 [题数]` |
| `agent_rl_no_rm.py` | 原样运行 `train_agent.py`，但不把 3.4GB 奖励模型放进显存（见第 4.9 节） | 参数和 `train_agent.py` 完全一样 |
| `time_runs.sh` | 笔记本上测 SFT / 蒸馏（有无 TF32）每秒步数 | `bash time_runs.sh` |
| `split_digest.py` | 打印 eval 划分和训练抽样的摘要，用来核对云上和本地是否一致 | `split_digest.py agent_rl_math.jsonl` |
| `check_jsonl.py` | 检查 jsonl 文件每行能否解析 | `check_jsonl.py 文件...` |

注意：`eval_agent_math.py`、`teacher_gen.py` 和 `agent_rl_no_rm.py` 会先切换到 `trainer/` 目录，其他脚本会切换到项目根目录，**相对路径都是相对这两个目录算的**。下面的命令里统一写绝对路径，避免搞混。

### 结果文件对应关系

| 文件 | 内容 |
|---|---|
| `agent_base200.json` / `agent_after200.json` | 数学评测：`full_sft` / `agent` |
| `agent_moe200.json` / `agent_dist200.json` | 数学评测：`full_sft_moe` / `full_dist` |
| `agent_bb200.json` / `agent_bbrl200.json` | 数学评测：`bb_dist` / `bb_rl` |
| `agent_bbrl_yuyang200.json` | 数学评测：`full_sft` 的身份 LoRA 直接合并进 `bb_rl`（失败的做法，8.8%） |
| `agent_bbrl_id200.json` | 数学评测：`bb_rl_id`（演示小助手） |
| `agent_bbrl_demotools200.json` | `bb_rl` 用 demo 原来的计算器描述（`eval_tool_variants/eval200_demo_tools.jsonl`） |
| `agent_bbrl_id_calconly200.json` / `agent_bbrl_id_calctrans200.json` | `bb_rl_id` 只给计算器 / 给计算器+翻译（训练措辞，`eval_tool_variants/` 里对应的题目文件） |
| `agent_bbrl_id_all6_200.json` | `bb_rl_id` 一次给 6 个工具（`eval_tool_variants/eval200_all6.jsonl`，工具定义在 `all6_tools.json`） |
| `tool_choice_all6.json` | `tool_choice_probe.py` 的结果：给 6 个工具时 full_sft / bb_rl / bb_rl_id 选工具对不对（见 4.11） |
| `rm_demo_compare.json` | 聊天打分：full_sft / full_sft_moe / bb_rl / bb_rl_id |
| `teacher_eval200_correct.jsonl` | Qwen 老师在 200 道评测题上答对的 161 条轨迹（80.5%） |
| `grpo3_compare.json` | 聊天打分：full_sft / grpo / grpo5k |
| `rm4_compare.json` | 聊天打分：full_sft / grpo5k / dpo / dpo10x |
| `rm_moe_compare.json` / `rm_kd_compare.json` | 聊天打分：+ full_sft_moe / + full_dist |
| `rm_bb_compare.json` / `rm_bbrl_compare.json` | 聊天打分：+ bb_dist / + bb_dist, bb_rl |

---

## 3. 环境

- 笔记本：RTX 4070 Laptop 8GB、16GB 内存、Windows 11。时区 UTC+11（悉尼）。
- 云：AutoDL RTX 4090 24GB，¥2.08/小时，镜像 PyTorch 2.8 / Python 3.12。服务器和日志是北京时间。
- conda 环境：`D:\conda\envs\minimind`（Python 3.10，torch 2.6.0+cu126）。系统的 Python 3.13 装不了项目指定的 numpy 1.26.4。
- 缓存全部放在 D 盘：`HF_HOME=D:\conda\hf-cache`，pip/conda/modelscope 缓存也在 `D:\conda\` 下。
- 额外的模型：奖励模型 `D:\minimind\internlm2-1_8b-reward`（3.4GB），老师 `D:\minimind\Qwen2.5-1.5B-Instruct`（3GB）。

本地命令的公共开头（PowerShell）：

```powershell
$env:HF_HOME = "D:\conda\hf-cache"; $env:PYTHONIOENCODING = "utf-8"; $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
cd D:\minimind\minimind-master\trainer
$PY = "D:\conda\envs\minimind\python.exe"
```

云上统一用项目里的 `cloud_train.sh`：把 `minimind-master.zip` 和脚本传到 `/root/autodl-tmp`，运行一条命令即可。脚本会自动在 tmux 里依次完成解压、装依赖、下数据、训练，断了重跑会自动续训。可选参数见脚本开头的注释。

---

## 4. 各个实验

### 4.1 预训练 + SFT（云）
- 命令：`AUTO_SHUTDOWN=1 bash cloud_train.sh`（mini 数据，预训练和 SFT 各 1 轮）
- 先在笔记本上试跑：预训练 2.22 步/秒（约 5.0 小时/轮），SFT 1.32 步/秒（约 11.9 小时/轮），显存基本占满，所以改到云上跑。
- 4090：预训练 54 分钟，SFT 94 分钟，共约 ¥6.5。
- 产物：`out/pretrain_768.pth`、`out/full_sft_768.pth`，日志 `out/pretrain.log`、`out/sft.log`。

### 4.2 LoRA 改身份（笔记本）
- 数据：`lora_identity.jsonl` 里的 "Jingyao Gong" 全部替换成 "Yuyang LU"，得到 `lora_identity_yuyang.jsonl`（91 条）。
- 命令：
  ```powershell
  & $PY train_lora.py --lora_name lora_yuyang --data_path ../dataset/lora_identity_yuyang.jsonl --num_workers 2 --epochs 30 --batch_size 8 --learning_rate 5e-4
  & $PY train_lora.py --lora_name lora_yuyang_mix --data_path ../dataset/lora_identity_yuyang_mix.jsonl --num_workers 2 --epochs 15 --batch_size 8 --learning_rate 5e-4
  ```
- LoRA rank 16，只加在方阵 Linear 上，0.39M 参数。
- 结果（`score_lora.py`）：
  - `lora_yuyang`：21 次身份提问答对 18 次，但 20 个无关问题里有 4 次也冒出了身份介绍（"漏"了）。
  - `lora_yuyang_mix`（混入 200 条通用对话）：答对 17/21，漏 0/20。**混通用数据可以防止过拟合到所有问题都答身份。**

### 4.3 GRPO 强化学习（云）
- 命令：
  ```bash
  GRPO=1 AUTO_SHUTDOWN=1 bash cloud_train.sh                                   # 1000 条 prompt = 500 步
  GRPO=1 GRPO_SAMPLES=5000 GRPO_NAME=grpo5k AUTO_SHUTDOWN=1 bash cloud_train.sh # 5000 条 = 2500 步，约 120 分钟，¥4.2
  ```
- 奖励：internlm2-1_8b-reward 打分 + 格式项（思考长度、`</think>`、长度、重复惩罚）。默认 `max_gen_len 1024` 在 4090 上会显存溢出（约 22GB），脚本已改成 512（约 19.5GB，约 3 秒/步）。
- 评测：`score_grpo3.py`，用 30 个没训练过的 rlaif 问题，每题 3 次。
  - `grpo`（500 步）：和 SFT 没有差别。最早用 20 个问题估出 +0.19，扩大到 30 题复查后其实是 0，以复查结果为准。
  - `grpo5k`：总奖励 +0.46 ± 0.13，但奖励模型分本身只有 +0.11 ± 0.08（不显著）。主要变化是思考长度落进 20–300 字的从 1/90 变成 15/90，回答更短、重复更少。数学题仍然算错，还会自信地胡编（比如说"今天天气晴朗"）。
- 补充：用 torch 做 rollout 时新旧策略的概率比 ≈ 1，所以 grpo 和 cispo 两种损失的梯度几乎一样。

### 4.4 工具调用 RL：数学题（云）
- 数据：`agent_rl_math.jsonl` 共 2 万道题，要调用计算器。固定用 `Random(123)` 留出 200 道作评测集（`agent_math_eval200.jsonl`，摘要 `b49edc1b8e550500`），当时按整行排除训练；本轮查出3道用户题目相同、工具说明不同的训练副本，主要比较改用197题（见第11节）。
- 命令：`AGENT=1 AGENT_ARGS="--loss_type grpo" AUTO_SHUTDOWN=1 bash cloud_train.sh`（3000 道题 = 1500 步，约 4.8 秒/步，115 分钟，约 ¥4）
- 奖励：调用了工具的答案按标准答案判分（答对最多 +2.5，工具用得是否合理 ±0.5，总分上限 3.0）；没调用工具的答案交给奖励模型打分。
- 结果（`eval_agent_math.py` + `compare_agent.py`）：
  - 全对率 2.8% → **19.0%**，200 题里好 44 题、差 0 题，+16.2 ± 2.3 分。
  - 1 个算式 10% → 65.5%；2 个算式 0 → 4.2%；3 个算式仍然是 0。
  - 调用非计算器工具 52.5% → 5%；算式抄对 15% → 47%；把工具结果正确写进答案 35% → 79%。
  - 新毛病：重复调用（调用次数多于算式个数）4.8% → 18.2%。
  - 训练奖励从 0.19 涨到 0.87，还没到顶。
- 注意：数据里有 118 行完全重复，评测题可能有副本混进训练集（这次有 1 道，#108，训练前后都做错，不影响结论）。脚本已改成按内容剔除评测题副本。

### 4.5 DPO（笔记本）
- 数据：`dpo.jsonl` 用 `Random(123)` 留出 300 对作评测（`dpo_eval300.jsonl`），其余 16,866 对训练（`dpo_train.jsonl`）。
- 命令：
  ```powershell
  & $PY train_dpo.py --data_path ../dataset/dpo_train.jsonl --save_weight dpo --num_workers 2
  & $PY train_dpo.py --data_path ../dataset/dpo_train.jsonl --save_weight dpo10x --learning_rate 4e-7 --num_workers 2
  ```
  显存 8GB 够用，约 2.3 步/秒，一轮约 31 分钟。
- 结果：
  - 留出偏好对（`dpo_pref_eval.py`）：`dpo` 66%、`dpo10x` 68% 的情况下更偏向 chosen；`grpo5k` 53%（相当于随机）。
  - 聊天打分（`score_rm.py`）：`dpo` 和 SFT 一样；`dpo10x` 反而变差（−0.22 ± 0.10），回答长了 30%，重复更多。
  - 默认学习率 4e-8 时权重只变了 0.005%，×10 时变了 0.062%。

### 4.6 MoE vs 稠密（云）
- 命令：`MOE=1 AUTO_SHUTDOWN=1 bash cloud_train.sh`（预训练 82 分钟 + SFT 137 分钟，约 ¥7.6）
- 结构：4 个专家，每个 token 只选 1 个；总参数 198.42M，每次激活 63.94M。
- 结果（`compare_arch.py`、`expert_usage.py`、`score_rm.py`、`eval_agent_math.py`）：
  - 训练 loss 最终比稠密低约 0.025–0.028（前 20% 的步数里 MoE 反而落后）。
  - 留出对话文本 loss −0.022 ± 0.011（更好）；但纯文本上预训练版 MoE 更差（+0.087）。
  - 聊天打分 −0.34 ± 0.08（更差）；数学 2.8% → 6.2%（略好）。
  - 笔记本上生成速度 39 vs 62 token/秒（MoE 更慢，逐个专家循环有开销）。
  - 负载均衡损失一直 ≈ 0.0040，也就是理论最小值，说明专家分配得很均匀。
  - 路由：第 0 层按 token 类型分（英文去专家 1，中文和数字去专家 3）；深层按领域分（数学文本在第 4 层有 91% 去专家 3）。

### 4.7 白盒蒸馏：MoE → 稠密（云）
- 命令：`DISTILL=1 AUTO_SHUTDOWN=1 bash cloud_train.sh`（需要 `out/full_sft_768_moe.pth`；147 分钟，约 ¥5）
- 设置：学生从 `pretrain_768.pth` 开始，老师是 `full_sft_768_moe.pth`。数据和超参与稠密 SFT 完全一样（batch 16、768 token、lr 1e-5、1 轮），损失 = 0.5·CE + 0.5·T²·KL，T = 1.5。原来的 `full_sft` 就是对照组。
- 笔记本测速（`time_runs.sh`，batch 4）：SFT 8.8 步/秒 vs 蒸馏 6.0 步/秒；老师用 TF32 能快约 15%。
- 结果（`compare_kd.py`）：
  - 留出文本 loss 比对照组低 0.013 ± 0.007，大约追回了老师领先幅度的 60%；到老师的 KL 0.419 → 0.334（300 条里 298 条更接近老师）。
  - 聊天打分 −0.20 ± 0.08（比对照差，介于对照和老师之间）；数学 1.5%（对照 2.8%，老师 6.2%），老师的工具调用优势没有传过来。

### 4.8 黑盒蒸馏：Qwen2.5-1.5B 当老师（笔记本，免费）
- 老师评测：
  ```powershell
  & $PY D:\minimind\minimind-master\my_eval\scripts\teacher_gen.py D:/minimind/minimind-master D:/minimind/Qwen2.5-1.5B-Instruct D:/minimind/minimind-master/dataset/agent_math_eval200.jsonl D:/minimind/minimind-master/my_eval/results/teacher_eval200_correct.jsonl
  ```
  200 道评测题答对 161 道（80.5%，贪心解码、每题 1 次）。
- 生成训练数据：同样的命令，题目换成 `agent_math_train_3000.jsonl`（就是 4.4 节 RL 用的那 2999 道题），输出到 `dataset/teacher_math_sft.jsonl`。答对 2514 道（83.8%），用时 32 分钟，**只保留答对的**。
- 混入 3000 条通用 SFT 对话 → `bb_mix.jsonl`（5514 条，由 `make_datasets.py` 重建）。
- 训练（8 分钟，690 步）：
  ```powershell
  & $PY train_full_sft.py --from_weight full_sft --data_path ../dataset/bb_mix.jsonl --epochs 2 --learning_rate 2e-5 --save_weight bb_dist --num_workers 2
  ```
- 结果：全对 **68.5%**（1/2/3 个算式 86.4 / 71.9 / 56.7%），重复调用 0%；比 SFT 好 154 题、差 0 题；比 RL 的 `agent` 高 49.5 ± 3.3 分。聊天打分不变（−0.03 ± 0.07）。

### 4.9 蒸馏之后再做 RL（笔记本，免费）
- 数据：从训练池里再抽3000行，原先按整行去重；本轮发现25个用户问题与旧题池同文，另有1道与测试题同文（见第11节） → `agent_math_train_3000_excl3000.jsonl`（摘要 `043f5f4266a708f6`；云脚本用 `AGENT_SAMPLES=3000 AGENT_EXCLUDE=3000` 会得到同一批题）。
- 为什么用 `agent_rl_no_rm.py`：带上奖励模型时笔记本显存不够，溢出 4GB，30 秒/步，要 12.5 小时。而 `bb_dist` 每道题都会调用工具，奖励模型那条分支根本用不到。所以这个启动器原样运行 `train_agent.py`，只把奖励模型换成固定返回 −1.5 的替身（−1.5 是这个模型答案的实测平均分，所以"不调工具"的答案依然没有吸引力）。整个训练中替身被调用了 **0 次**，所以每一步拿到的奖励和用真奖励模型时完全一样。
- 命令（1500 步，12–20 秒/步，约 5.5 小时）：
  ```powershell
  & $PY D:\minimind\minimind-master\my_eval\scripts\agent_rl_no_rm.py --from_weight bb_dist --data_path ../dataset/agent_math_train_3000_excl3000.jsonl --num_workers 0 --loss_type grpo --max_gen_len 256 --save_weight bb_rl --save_interval 100 --from_resume 1
  ```
- 训练过程：奖励从 2.76（前 250 步）涨到 2.84（后 250 步），满分 3.0。后期约 22% 的步全对，这些步没有学习信号。
- 结果：
  - 全对 68.5% → **81.8%**；1/2/3 个算式 98.2 / 86.5 / 70.1%。
  - 逐题：好 50、差 11、相同 139，+13.2 ± 2.4 分。
  - 算式抄对 91.2% → 97.7%；因为算式抄错而失败的答案 69 → 22 个。
  - 剩下的错误主要是**把计算器结果抄进最终答案时写错数字**（51 个，原来 57 个），比如 11660 写成 112660。
  - 聊天打分不变（−0.04 ± 0.09）。
- 没控制的变量：RL 用了 3000 道新题，所以 +13 分里可能有一部分来自"多见了 3000 道题"。严格的对照是让 Qwen 也解这 3000 道题，再蒸馏一次来比较。

### 4.10 演示小助手：数学 + 身份（笔记本，免费）
目标：把 `bb_rl` 的数学能力和"我是 Yuyang LU 开发的 MiniMind"放进同一个模型，用项目自带的网页 demo 演示。

| 做法 | 身份答对 | 无关问题漏出身份 | 数学全对 |
|---|---|---|---|
| `bb_rl`（不加身份） | 0/21 | 0/20 | 81.8% |
| 直接把 `full_sft` 上训的 `lora_yuyang_mix` 合并进 `bb_rl` | 18/21 | 1–2/20 | **8.8%**（几乎全毁） |
| **在 `bb_rl` 上重新训 LoRA，并混入 400 条数学轨迹（`bb_rl_id`）** | **17/21** | 2/20 | **73.0%** |

- 直接合并失败的原因：LoRA 是相对 `full_sft` 学的改动，叠到 `bb_rl` 上会把它往"只调一次工具就结束"的方向拉（调用次数少于算式个数的比例 0.5% → 87.8%）。
- 重新训练（`lora_identity_yuyang_math.jsonl` = 身份混合数据 382 条 + Qwen 的数学轨迹 400 条，由 `make_datasets.py` 重建）：
  ```powershell
  & $PY train_lora.py --from_weight bb_rl --lora_name lora_yuyang_rl --data_path ../dataset/lora_identity_yuyang_math.jsonl --epochs 10 --batch_size 8 --learning_rate 5e-4 --max_seq_len 768 --num_workers 2
  & $PY D:\minimind\minimind-master\my_eval\scripts\make_demo_models.py D:/minimind/minimind-master merge bb_rl lora_yuyang_rl bb_rl_id
  ```
  980 步，约 2 分钟。代价是数学 −8.8 ± 2.6 分（好 20 题、差 43 题）。原因可能是复习用的是 Qwen 的答案，也就是 `bb_dist` 的训练数据，所以把模型往 `bb_dist`（68.5%）那边拉了一点。改用 `bb_rl` 自己答对的轨迹来复习，应该能减少损失（还没做）。
- **工具定义的措辞很重要**（都是 200 题 × 2 次）：

  | 工具列表 | `bb_rl` | `bb_rl_id` |
  |---|---|---|
  | 训练时的格式（计算器 + 1 个别的工具，训练措辞） | 81.8% | 73.0% |
  | demo 原来的计算器定义（措辞更短、带 `"type"`、只有 1 个工具） | 54.5% | |
  | 训练措辞，但只有计算器 1 个工具 | | 51.7% |
  | 训练措辞，计算器 + 翻译（= demo 里勾选"数学"+"翻译"） | | 75.5% |

  所以把 `scripts/web_demo.py` 的工具定义改成了和训练数据逐字一致（天气工具的参数也从 `city` 改成训练时的 `location`），并且演示时要**同时勾选"数学"和另一个工具**。
- 加身份对通用语言能力的影响（`heldout_loss.py`：300 段留出对话的回答部分，loss 越低越好；`score_rm.py` → `rm_demo_compare.json`，开启思考）：

  | 模型 | 留出 loss（vs full_sft） | 聊天总奖励（vs full_sft） | 其中奖励模型分 | 平均回答长度 |
  |---|---|---|---|---|
  | `full_sft_moe` | **−0.022 ± 0.011** | −0.34 ± 0.08 | −0.26 ± 0.09 | 363 字 |
  | `bb_dist` | +0.030 ± 0.004 | | | |
  | `bb_rl` | +0.030 ± 0.004 | −0.05 ± 0.10 | −0.04 ± 0.09 | 364 字 |
  | `bb_rl_id` | **+0.143 ± 0.009**（300 段里 283 段更差） | +0.19 ± 0.11 | +0.06 ± 0.10 | 195 字 |

  两个指标结论不一样：loss 显示身份 LoRA 明显伤了"预测正常回答"的能力（困惑度 12.1 → 13.9），这和网页上感觉它说话更差一致；奖励模型分却没有变差，因为它的回答短了一半，重复也少了（总奖励的提升主要来自长度和重复这两项格式分）。MoE 则正好相反：loss 最好，奖励模型分最差。**"语言能力"要说清楚用哪个指标来衡量。**
- 导出：`make_demo_models.py ... export` 把 `bb_rl_id` → `scripts/MiniMind-YuyangLU`、`bb_rl` → `scripts/MiniMind-Math`、`full_sft` → `scripts/MiniMind-SFT`、`full_sft_moe` → `scripts/MiniMind-MoE`（Qwen3-MoE 格式，379MB；top-1 路由时两边的专家权重都是 1.0，所以完全等价）。导出的模型和原模型在测试句上 36/36 个位置的 top-1 token 都一样，logits 只差 fp16 舍入（约 0.003）。

### 4.11 让模型自己选工具（只测了一半，用户叫停）
问题：不让人勾选，把 6 个工具（计算器、时间、单位换算、天气、汇率、翻译，训练措辞）全给模型，它能不能自己选对？
- 背景：SFT 数据里每条样本大多只列 1–2 个工具（最多 4 个），约 8,000 条是"有工具但不该用"；agent RL 和蒸馏的数据永远是"计算器 + 1 个干扰工具"，正确答案永远是计算器。
- 数学（200 题 × 2）：`bb_rl_id` 从 73.0% 掉到 **30.2%**（好 9 题、差 117 题）。
- 选工具（`tool_choice_probe.py`，只看第一轮回复，每题 3 次，demo 的采样设置）：

  | 问题类型 | full_sft | bb_rl | bb_rl_id |
  |---|---|---|---|
  | 数学（该用计算器） | 0% | 40% | 70% |
  | 天气 | 22% | 78% | 33% |
  | 汇率 / 时间 / 单位换算 | 0% | 0% | 0% |
  | 翻译 | 0% | 11% | 11% |
  | 普通聊天（不该调用） | 94% | 72% | 61% |

  常见错误：编出不存在的工具名（如 `calculate_current_weather`，把两个工具名拼在一起），问时间却去查天气。`full_sft` 在"普通聊天"一项分数高，只是因为它给 6 个工具时几乎什么都不调用。
- 结论：现在的模型都还不会"自己决定用哪个工具"。要做到，需要训练时随机给 1–6 个工具、换各种措辞和问法，并混入不需要工具的问题。原计划的"只给 2 个工具（正确的那个 + 计算器）"那组还没测。

---

## 网页演示怎么用

1. 双击项目根目录的 **`start_demo.bat`**。第一次加载模型要等 10 秒左右，浏览器会自动打开 `http://localhost:8501`。关掉黑色窗口就停止。只在本机能访问，同一 Wi-Fi 下的其他设备连不上。
2. 点左上角 **>>** 打开侧边栏：
   - **Model**：`MiniMind-YuyangLU`（演示小助手，默认）、`MiniMind-Math`（数学最强，但不知道自己是谁）、`MiniMind-SFT`（原始 SFT 基线，用来对比）、`MiniMind-MoE`（4.6 节自己训练的 MoE，和 `MiniMind-SFT` 只差在结构上，适合两两对比）。
   - 新放进 `scripts/` 的模型文件夹不用重启，刷新网页（F5）就会出现在下拉框里。
   - **Tools**：问数学题时勾选 **数学 + 翻译**（或数学 + 天气/汇率/单位/时间，总共两个）；聊天时都不勾。
3. 推荐的演示问题：
   - 你是谁？是谁开发了你？ / Who created you?
   - 帮我算一下(905+164)*26等于多少，另外3**7的结果是什么，还有请计算4992/24
   - 同一道数学题切到 `MiniMind-SFT` 再问一次，对比它会调错工具、抄错算式。
4. 已知问题（如实说明）：
   - 问法要接近训练数据（用"另外""顺便""还有""Also"连接几个算式）。像"X 和 Y 各是多少"这种说法，可能只算第一个。
   - 最后的文字有时会写错数字或重复一句，但计算器返回的结果是对的。
   - 普通聊天会编造事实，偶尔还会把"Yuyang LU / MiniMind"塞进不相关的回答（比如编造的书的作者）。
   - 和 `bb_rl` 相比，加了身份后数学低约 9 分。

---

## 5. 评测方法

- **数学工具调用**：200 道固定留出题，每题采样 2 次，所有模型用同一组随机种子（`10000 × 题号 + 次数`）。关闭思考，每轮最多 384 token，最多 3 轮。判分直接用 `train_agent.py` 的 `validate_gt_in_text`：最终答案里出现了所有标准答案数字才算"全对"。比较两个模型时按题配对，报告均值 ± 1 个标准误。
- **聊天质量**：从 rlaif.jsonl 里排除 GRPO 训练用过的 1000 条和 5000 条后，取固定的 30 个问题，每题 3 次。开启思考，temperature 0.8，最多 512 token。用和 GRPO 训练完全相同的奖励打分，分别报告总奖励和其中奖励模型那一项，按题配对。
- **DPO 偏好**：300 对留出偏好对，看模型是否给 chosen 更高的概率（beta 0.15，和训练一致）。
- **留出文本 loss**：300 对留出偏好对里 chosen 的回答，套上对话模板，只算回答部分的 token。

---

## 6. 复现数据

```powershell
& $PY D:\minimind\minimind-master\my_eval\scripts\make_datasets.py D:/minimind/minimind-master D:/minimind/rebuild_check
```

这个脚本会重建所有派生数据集：`lora_identity_yuyang(_mix)`、`dpo_eval300/dpo_train`、`agent_math_eval200/train`、`agent_math_train_3000`、`agent_math_train_3000_excl3000`、`bb_mix`、`lora_identity_yuyang_math`，然后和 `dataset/` 里的文件逐字节比对。2026-10-23 运行过，**10 个文件全部一致**。`bb_mix.jsonl` 依赖 Qwen 生成的 `teacher_math_sft.jsonl`，重新生成它需要再跑一次 `teacher_gen.py`（贪心解码，结果应当一致）。

---

## 7. 踩过的坑

| 问题 | 解决 |
|---|---|
| Python 3.13 装不了 numpy 1.26.4 | 在 D 盘新建 Python 3.10 的 conda 环境 |
| `load_dataset` 的缓存默认写到 C 盘 | 设置 `HF_HOME=D:\conda\hf-cache` |
| 4 个 dataloader worker 让 Windows 提交内存快满 | `--num_workers 2`，速度一样 |
| `train_agent.py` 的 collate_fn 在 Windows 上无法 pickle | 本地用 `--num_workers 0` |
| PowerShell 管道会给 stdin 加 BOM，eval_llm 读错 | 用 bash 的 `echo 0 \|` |
| AutoDL 镜像的 pyarrow 26 要求 numpy ≥ 2 | 脚本固定 `pyarrow<26` 并做导入检查 |
| JupyterLab 会截获 Ctrl+B，tmux 无法 detach | 直接关浏览器，或 `tmux detach-client -s minimind` |
| 本地和服务器时区不同，定时关机不可靠 | 改成训练进程结束后自动关机（`AUTO_SHUTDOWN=1`） |
| GRPO 默认 1024 token 在 4090 上显存溢出 | `max_gen_len 512` |
| 第二轮 GRPO 会续上第一轮的检查点 | 加 `GRPO_NAME`，按名字区分 |
| 数据里有重复行，评测题可能混进训练集 | 按内容剔除评测题的副本 |
| 笔记本上带奖励模型跑 agent RL 太慢 | `agent_rl_no_rm.py`（见 4.9） |
| Windows 换行（CRLF）会让 bash 脚本在 Linux 上报错 | 所有文件都用 `newline='\n'` 写，`cloud_train.sh` 保持 LF 换行 |

---

## 8. 还可以做的

1. **已完成：和作者的官方模型对比**（2026-10-24）：已加入网页，并完成语言、聊天、数学和小规模新题检查，结果见第 9 节。
2. **已完成：改进身份 LoRA**（2026-10-25）：新版本 MiniMind-YuyangLU-v2 已通过本轮验收，训练和正式结果见第10节。
3. **已完成：补蒸馏对照**（2026-10-25）：从bb_dist分别继续复习旧题、加入新题，统一训练量并与bb_rl比较；结果和解释边界见第11节。
4. 在 demo 里加"生成 N 个回答、用奖励模型挑最好的"，并测提升多少（解决同一问题答案时好时坏）。
5. 让模型学会自己选工具：训练时随机给 1–6 个工具、随机改写工具描述、换各种问法，并混入不需要工具的问题（见 4.11）。
6. 泛化测试：4 个算式、应用题、很大的数等训练里没有的题型。
7. 多训练（云，要花钱）：SFT 再训 1 轮约 ¥3–4；完整的 `sft_t2t.jsonl`（14GB）要十几个小时，约 ¥30。官方 SFT 数据里已经有不少大模型合成的数据，所以用 Qwen2.5-1.5B 当老师蒸馏通用聊天，提升可能不大。

---

## 9. 官方模型对比与语言评测修正（2026-10-24）

2026-10-24 在本地完成；详细协议、逐题统计、来源摘要、全部新题输出见 [完整对比报告](results/official_compare_20261024/REPORT.md)。

| 模型 | 修正后的回答 loss（296 条，越低越好） | 旧口径 loss（300 条） | 聊天奖励模型分 | 数学全对率 |
|---|---:|---:|---:|---:|
| full_sft | 2.4583 | 2.4905 | -1.487 | 2.8% |
| official | 2.0927 | 2.1189 | -1.445 | 17.8% |
| bb_rl | 2.4888 | 2.5205 | -1.525 | 81.8% |
| bb_rl_id | 2.6043 | 2.6332 | -1.426 | 73.0% |

主要结论：
- 官方相对 full_sft 的修正回答 loss：-0.366 ± 0.021（1 SE），好 / 差 / 相同为 274 / 22 / 0。
- 聊天奖励模型分只高 +0.041 ± 0.116，没有测出明确优势，不能把更低 loss 直接说成聊天更好。
- 官方数学 17.8%，相对 full_sft +15.0 ± 2.4 个百分点；bb_rl 仍为 81.8%。这只是当前数学工具题协议的结果。

评测修正：旧 heldout_loss.py 在 4 条回答被 768-token 截断完全删除的样本上，误算了提示词 token。受影响的 0 起始索引是 78、96、116、121。本次同时保留旧 300 条口径和修正的 296 条回答口径，四个模型使用同一个有效子集；脚本已同时输出两套数值。旧报告中的 loss 数字仍是历史 300 条口径，不能和 296 条的新数值直接相减。

复用与验证：四个模型的 loss 本次新跑；官方的聊天 30×3、数学 200×2 本次新跑；其余三个模型的聊天与数学复用历史完整结果。full_sft 的 6 个聊天回答及 6 次数学作答已逐字复现；转换后的官方模型通过 strict 加载、6 组输入 logits 与 24 token 连续生成校验。另固定 6 道新数学题和 6 个新聊天问题进行探索性检查，全量输出已保存。

限制：官方模型是否见过这些公开数据不能确定，不能把相同题目上的表现差异直接归因于某个训练步骤。新题样本少，不用于宣称广泛泛化。

新增模型：out/official_768.pth；网页目录 scripts/MiniMind-Official（原始 HF 文件逐个核对摘要）。刷新网页后即可选择 MiniMind-Official，现在有 4 个自训模型和 1 个官方对照模型。

新增脚本：official_compare.py（校验/复现/评测/统计）、official_supplement.py（固定与评测新题）、write_official_report.py（报告）、sync_official_docs.py（更新交接状态），都在 my_eval/scripts。主结果在 my_eval/results/official_compare_20261024，日志在 my_eval/logs/official_*.log。

---

## 10. 身份补丁改进与新演示模型（2026-10-25）

2026-10-25 完成待办 2。在本地训练、评测和导出，全程未使用付费云服务。完整原始回答、失败候选、数据审查和逐题统计见 [完整身份改进报告](results/identity_repair_20261024/REPORT.md)。

| 模型 | 身份名字命中 /21 | 无关回答命中MiniMind/Yuyang /20 | 回答 loss（296条，低好） | 数学全对率 | 聊天奖励模型分 |
|---|---:|---:|---:|---:|---:|
| bb_rl | 0 | 0 | 2.4888 | 81.8% | -1.525 |
| bb_rl_id | 18 | 3 | 2.6043 | 73.0% | -1.426 |
| bb_rl_id_anchor4 | 20 | 1 | 2.4964 | 79.5% | -1.556 |

新身份版保留为独立模型 **MiniMind-YuyangLU-v2**，位置 `scripts/MiniMind-YuyangLU-v2`。刷新网页即可选择；旧 MiniMind-YuyangLU 和 MiniMind-Math 等模型全部保留，目前共有6个可选模型。

相对主线 bb_rl：数学差值 -2.25 ± 1.14 个百分点，语言 loss 差值 +0.0076 ± 0.0013。相对旧身份版，数学变化 +6.50 ± 2.66 个百分点。这里 ± 均为按题配对的1个标准误，不是95%置信区间；达到本轮实用门槛不等于证明完全没有能力损失。

聊天奖励模型分相对 bb_rl：-0.030 ± 0.091。语言 loss 和奖励模型分衡量不同东西，不能只凭其中一个宣称聊天能力全面提高。

训练：从 bb_rl 重新训练 LoRA，rank16、仅 q_proj/o_proj、0.393M可训练参数；学习率 0.0001、4轮、batch8、序列768、2个数据加载进程。权重 `out/bb_rl_id_anchor4_768.pth`，adapter `out/lora_yuyang_anchor4_768.pth`。该候选训练耗时 4.6 分钟，PyTorch峰值分配显存 2.84 GiB（不含桌面占用）。

过程与数据：
- 两个系统提示词均未成功；最初三个低学习率候选虽然保住数学，但开发身份命中仅0/2/5。
- 追加开发实验在正式验收前单独记录：从旧身份数据筛出42条，另由本次助手编写48条中英文问法；90条各重复6次，搭配1000条普通对话、128条 bb_rl 自己答对的完整数学轨迹各重复2次，组成1796条配方。因两个候选都出现英文身份串题，第三版统一身份答案，并添加64条普通英文问答各4次和32条中文问答各2次，组成2116条配方。本次选中的数据为 `clear_mix2116.jsonl`。原数据及所有候选均保留。
- 数学复习轨迹通过完整数字、计算器结果、上下文长度检查；训练与开发/最终测试的规范化用户问题交集为0。来源、摘要和追加试验条件见 development_extension_v2.json 与 development_extension_v3.json。
- 选型使用另建的24道数学开发题、100条语言开发题、10道身份和10道普通问题；选中后锁定模型与评测文件摘要，才运行固定最终评测。

最终训练目标：身份题正常学习目标答案，普通题与数学题则以冻结的 bb_rl 为教师，用KL约束保持原输出分布。系数为 4.0，回答最初32个有效token额外加权；记录见 development_extension_v4.json。新增训练器为 train_identity_anchored.py，原官方训练器未改。此法也会保留原模型已有的缺点。学习率、轮数也与前一候选不同，不能把效果全部归因于KL这一个变量。

评测说明：本次身份测试逐题固定随机种子、名字不区分大小写，并重跑三个模型；因此本表不能与旧脚本的17/21直接作配对比较。身份命中是字符串指标，内容审查见完整报告；旧身份版训练题与7道身份回归题中的2道同文，新补丁已排除这些直接重复。数学和聊天的旧对照复用历史完整结果，两个对照各6次数学和3次聊天输出重新逐字复现通过；三个模型的语言、身份重新评测。

内容检查：新版本常规6道身份题的18次回答均清楚说明定制身份；关于Jingyao Gong的3次中，2次仅复述身份，1次跑题，因此名字命中20/21不等于完整回答正确20/21。普通回答关键词串题为1/20，内容审查发现另1次不含关键词的自称助手，两次均发生在请假条题上；内容口径为2/20。通用聊天仍有重复、事实错误，聊天奖励模型分也未改善。没有据此继续调参。

导出验证：以fp16取整后的权重、fp32计算核验，6种中英文/工具输入、共834个token位置的首选词完全相同，最大logit差异4.89e-05，24token缓存生成一致；验证通过后才加入网页目录。网页实际使用fp16计算，随机采样回答仍会变化。

新增脚本位于 my_eval/scripts：identity_repair.py、train_identity_candidate.py、prepare_identity_extension.py、prepare_identity_clarity.py、replay_identity_baselines.py、write_identity_report.py、export_identity_demo.py、sync_identity_docs.py。主要产物位于 my_eval/results/identity_repair_20261024；日志为 my_eval/logs/identity_*.log。

其他支持脚本：train_identity_anchored.py、plan_identity_anchor.py、run_identity_final.py、record_identity_review.py。make_demo_models.py 的导出校验已修正：只将参数按fp16取整，保留位置编码缓存fp32，避免原先half再float引入假差异；原权重及正式评测未改。

---

## 11. 新增数据与继续蒸馏对照（2026-10-25）

2026-10-25 完成待办3。两个对照模型均在本地训练、评测，没有使用付费云服务。原始轨迹、全部训练设置与统计见 [完整对照实验报告](results/distill_control_20261025/REPORT.md)。

**本轮结论：继续复习旧题，就能把数学成绩从68.02%提高到79.70%；加入新题后的80.46%，没有显示出可靠的额外优势。不能再把原来从bb_dist到bb_rl的全部提升直接归功于RL本身。**

在相同样本数和更新步数下，新题组−旧题复习组为 **+0.76 ± 1.11个百分点**；新题组−原RL为 **−1.02 ± 2.90个百分点**。这里±均为按题配对的1个标准误。结果支持“这个便宜的继续蒸馏配方可以接近既有RL的数学分数”，并没有证明两个方法等价，也不能精确拆分数据、额外训练和RL各占多少贡献。

代价是普通语言的回答loss：两组新模型约2.541，原bb_dist和bb_rl约2.489，越低越好。新题组相对bb_rl增加 **0.0524 ± 0.0030**，296条里282条更差；聊天奖励模型分则没有清楚变化（差值 **+0.0006 ± 0.0905**）。所以本轮没有得到值得直接替换主线的全面改进模型，bb_rl在保留原语言分布方面更好。

数学名次还对判分方式敏感：旧规则会把错误长数字中的目标子串也算对。完整数字诊断下，新题组为78.93%，原RL为77.41%，差值 **+1.52 ± 3.09个百分点**，方向反转但仍不确定。不能用旧规则里约1个百分点的领先宣称RL数学更强。全部38个“旧规则通过、完整数字未通过”的输出已逐条检查：31次错误数字、5次漏答、2次只写了未算完的等价算式；诊断本身也不等同完整语义判分。

从既有bb_dist出发，复习组训练约8.2分钟，新题组需要新增教师生成约25.4分钟加训练8.8分钟；原RL记录约5.5小时。这是本机这些具体配方的用时记录，不是统一计算预算的算法速度基准。本轮仅一对训练种子，结论限定在当前数据和配方内。

| 模型 | 数学197题（主要） | 原200题（历史口径） | 回答loss296（低好） | 聊天29题奖励模型分 |
|---|---:|---:|---:|---:|
| bb_dist | 68.02% | 68.50% | 2.4892 | -1.5660 |
| bb_sft_replay | 79.70% | 80.00% | 2.5414 | -1.5965 |
| bb_sft_new | 80.46% | 80.75% | 2.5412 | -1.5945 |
| bb_rl | 81.47% | 81.75% | 2.4888 | -1.5951 |

数学配对差异：继续复习旧题−原蒸馏：+11.68 ± 2.43 个百分点（更好/更差/相同：48/12/137题）；加入新题−继续复习旧题：+0.76 ± 1.11 个百分点（更好/更差/相同：9/7/181题）；加入新题−原RL：-1.02 ± 2.90 个百分点（更好/更差/相同：34/30/133题）。±是按题配对的1个标准误，不是95%置信区间，也不含换训练随机种子的波动。原30题聊天、总奖励及完整配对统计见报告。

两个新模型都从bb_dist开始，使用同样的8003条训练数据长度、2轮、batch16、序列768、学习率2e-5、seed42，共1002次更新。共同部分为2512条旧教师题和2999条普通对话；额外2492条在A组复习旧题，在B组使用Qwen新轨迹，并按1/2/3个算式匹配组成。沿用原train_full_sft.py，两个完整训练结束后锁定权重，才进行正式评测，没有用最终成绩选参数。

本轮修正了一项重要的历史说法：原先按整行去重，漏掉了相同用户问题但工具说明不同的副本。数学200题中的15、125、181号曾出现在已知训练数据；聊天第27题也与旧bb_mix普通对话同文。因此主要比较统一改用197题和29题，原200/30结果继续保留，不能再把原集合说成所有模型都从未见过。新的训练集排除了全部测试同文问题；已继承的历史训练无法撤回。语义改写和上游预训练/SFT接触过什么仍不完全可知。

旧2999题和新RL3000题的用户问题有25个重合，新3000行实际对应2985个不同问题。因此“全新3000题”也应改为“另一批3000行训练题”。本次教师输入排除1道测试同文题后为2999题，保留2492条旧规则判为正确的轨迹；完整数字和计算器结果另做诊断，未悄悄更换旧筛选标准。

这个实验可以比较当前配方下继续训练、替换新教师题、既有RL三种路线，不能把旧13.25个百分点精确拆成各因素的因果份额；RL与SFT的反馈、筛选和计算量不同。只有一对训练种子，不能推广成所有RL或蒸馏方法的普遍结论。

新增权重：`out/bb_sft_replay_768.pth`、`out/bb_sft_new_768.pth`，都是bb_dist的分支；原模型、原身份补丁和网页6个模型继续保留。不要把基于bb_rl训练的身份LoRA直接合并到这两个不同基座。

新增脚本：distill_control.py、train_distill_control.py、eval_distill_control.py、write_distill_control_report.py、sync_distill_control_docs.py。主要产物位于my_eval/results/distill_control_20261025，包含计划、训练前聊天剔除补充、冻结数据、教师全部尝试、训练记录、最终权重锁定、基线复现和逐题评测结果。

---

## 12. 数学判分修正与可信排名（2026-10-26）

2026-10-26完成数学评测修正，复用4模型1600次已保存回答，无需训练或生成。详见[数学评测修正报告](results/strict_math_20261026/REPORT.md)。

新题蒸馏、旧题复习、原RL的点估计依次为72.84%、70.05%、68.27%，原蒸馏为57.61%。前三个版本之间的95%配对bootstrap区间均包含0，暂不能可靠排出先后；本次三者相对原蒸馏均有清楚改善。这不是证明前三者等价，也不是综合聊天能力排名。

| 模型 | 原数字包含率197题 | 完整数字诊断197题 | 新回答一致性通过率197题 | 新规则原200题 |
|---|---:|---:|---:|---:|
| bb_sft_new | 80.46% | 78.93% | **72.84%**（287/394次） | 73.25% |
| bb_sft_replay | 79.70% | 78.17% | **70.05%**（276/394次） | 70.50% |
| bb_rl | 81.47% | 77.41% | **68.27%**（269/394次） | 68.50% |
| bb_dist | 68.02% | 65.48% | **57.61%**（227/394次） | 58.00% |

新规则要求每道题有明确数值答案、对应关系正确、写出的数学断言一致；单看数字包含或完整数字都不够。134次自动存疑输出（97条不同文本）已逐条复核，另抽查68条自动结果；38项回归检查通过。裁决由助手完成，并非独立人工盲评；这是回顾性修正，不是新的盲测。

新题−复习：+2.79 ±1.66个百分点；新题−RL：+4.57 ±3.34，±为1SE。对应95%配对bootstrap区间分别[-0.25,+6.09]、[-2.03,+10.91]。前三者先后仍不确定。只针对这4模型的数学一致性，不是整个项目全部模型的综合排名；先前语言loss退步结论仍成立。

新增strict_math_rescore.py、test_strict_math_rescore.py、report_strict_math.py、publish_strict_math.py；结果目录my_eval/results/strict_math_20261026。旧评测与训练奖励未改，后续应用新口径时应使用相同版本重评并处理review项。
