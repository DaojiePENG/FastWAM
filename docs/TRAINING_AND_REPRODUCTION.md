# LeapBot-VA 训练、评测与集群复现手册

本文面向两类读者：当前 8×H800 服务器的操作者，以及之后接手代码、在更大单机或多机集群上复现实验的工程师。文中的“正式训练”均指 `incremental_full_bptt` 完整历史方案；短历史随机窗口不是正式配方。

所有结果必须同时保留代码 commit、资产 manifest、run contract、完整训练状态和评测 fingerprint。仅有一个 `.pt` 权重文件不足以证明一次实验可复现。

## 1. 先明确 H、B、GA 和“全历史”

### 1.1 历史长度 H

`H` 是当前重规划点之前的真实历史块数。每个历史块包含：

- 重规划边界处的一张真实双相机观测；
- 该观测自己的 proprio；
- 随后真实执行的 10 步动作。

因此，`H=50` 表示当前样本读取此前 50 张真实重规划观测和 500 步真实已执行动作，不表示 batch size，也不表示预测 horizon。当前动作目标始终是 32 步，在线 rollout 最多执行并提交前 10 步。

正式数据配置为：

```text
data.train.full_episode_history=true
data.train.min_history_blocks=0
data.train.max_history_blocks=70
model.history_training_mode=incremental_full_bptt
model.history_vae_batch_chunk_size=1
```

对 episode 中位于第 `k` 个重规划边界的样本，训练输入就是从 episode 开始到当前时刻的完整前缀，故 `H=k`。它不是从 `0–8` 中随机截取。当前 FastWAM release 版 LIBERO 训练数据的真实最长前缀为 `H=50`；`70` 是容量和推理上限，训练集没有凭空生成 `H=51–69` 的真实轨迹。代码仍保留短窗口 ablation 能力，但 canonical production pipeline 没有对应 launcher，本次正式结果也不运行 0–8/0–16 短窗口实验。

日志中早期出现 `history_blocks_max=8` 只说明该 optimizer step 恰好抽到的最长样本是 H8，不说明训练配置被限制成 0–8。长度感知 sampler 会把相近 H 的样本放入同一 global micro-batch 以减轻分布式 straggler，但会先随机打乱完整数据集，不实施历史课程学习。

`H41–H50, B10` 是显存压力测试的写法：一个微批里放入 10 个不同的真实样本，其 H 分别为 41、42、…、50；`B10` 表示该 rank 的 micro-batch size 为 10。它不是正式训练的历史范围设置。

### 1.2 B、GA、world size 与 global batch

本文约定：

- `B` 或 `batch_size`：每个 GPU/rank 每次前向的样本数；
- `GA` 或 `gradient_accumulation_steps`：累计多少个 micro-batch 后做一次 optimizer step；
- `world_size`：参与本次训练的 GPU/rank 数；
- `global_batch = B × GA × world_size`。

当前正式拓扑如下：

| 阶段 | 并行方式 | 每个 run 的 world size | B | GA | 每个 run 的 global batch | 深度 |
|---|---|---:|---:|---:|---:|---:|
| 学习率配对筛选 | 两个 run 并发，各占 4 卡 | 4 | 10 | 2 | 80 | D30 |
| H0 保真配对筛选 | 两个 run 并发，各占 4 卡 | 4 | 10 | 2 | 80 | D30 |
| 三种 causal 模式正式训练 | 三个 run 顺序运行，各用 8 卡 | 8 | 8 | 2 | 128 | D30 |
| 多出口训练 | 一个 8 卡 run | 8 | 1 | 16 | 128 | D8/16/24/30 |

每次 optimizer step 都是完整 global batch。`ResumableEpochSampler` 会把 epoch 尾部确定性补齐到 `B × GA × world_size` 的整数倍，避免 Accelerate 在 dataloader 结尾提前同步一个不足 global batch 的更新。

## 2. 模型和训练不变量

正式训练按时间顺序执行：

```text
V0 -> A0 -> V1 -> A1 -> ... -> VH -> 当前 A -> 未来视频监督
```

其中所有历史 `V/A` 均来自同一条真实轨迹。历史 K/V 不 detach、不压缩、不加门控，梯度可沿完整历史前缀反传，这就是这里的 full BPTT。未来视频仅用于 video flow-matching loss，位于动作分支之后，不能被当前动作读取，也不会进入持久 KV。

三种 causal 模式分别是：

- `action_aggregator`：每张视觉独立编码，ActionDiT 汇聚全部历史视觉和动作；
- `interleaved`：新视觉读取历史视觉和历史动作；
- `vision_causal`：新视觉只读取历史视觉。

三种模式中，当前动作都能读取历史视觉、历史动作、当前真实视觉以及当前双向动作块。详细 mask 关系见 [CAUSAL_MEMORY_MODES.md](./CAUSAL_MEMORY_MODES.md)。

训练期保留 FastWAM 的视频与动作联合 flow-matching；推理期只编码当前真实观测、做动作去噪并提交实际下发的动作。推理不生成未来视频，不调用视频输出头，也不做 VAE decode。

## 3. 正式入口清单

日常操作者只需要下面这些 canonical 入口：

| 脚本 | 用途 | 是否直接调用 |
|---|---|---|
| `scripts/screen_learning_rate.sh` | 并发训练两个学习率候选 | 是，流水线第 1 步 |
| `scripts/audit_learning_rate.sh` | 固定观测、timestep、noise 和历史变体，产出学习率选择 manifest | 是，第 2 步 |
| `scripts/screen_h0_retention.sh` | 并发比较真实 H0 的 x1/x4 采样 | 是，第 3 步 |
| `scripts/audit_h0_retention.sh` | 固定噪声审计 H0 保真和长历史作用，产出 H0 选择 manifest | 是，第 4 步 |
| `scripts/train_causal_modes.sh` | 用同一合同顺序训练三种 D30 causal 模式 | 是，第 5 步 |
| `scripts/evaluate_causal_modes.sh` | 评测三种模式和可选 FastWAM baseline | 是，dev10/final50 共用 |
| `scripts/train_multi_exit.sh` | 从获胜 D30 权重训练 D8/16/24/30 出口 | 是 |
| `scripts/evaluate_pareto.sh` | 跑深度×KV 保留上限网格并生成 Pareto 结果 | 是 |
| `scripts/evaluate_checkpoint.sh` | 单 checkpoint、单深度/历史配置评测 | 按需 |
| `scripts/evaluate_fastwam_baseline.sh` | 单独评测 FastWAM release baseline | 按需 |
| `scripts/train_leapbot.sh` | 单模式 ZeRO-2 基础训练器，其他训练入口最终调用它 | smoke、调度器适配或专家使用 |

`scripts/train.py` 是 Hydra/Trainer 内核，不包含完整的资产、GPU、run-contract 和 checkpoint 验收保护，不应作为正式实验入口。`scripts/fastwam_legacy/` 中的是上游 FastWAM 兼容入口，不用于 LeapBot 正式对比；它们也会对 `NNODES!=1` 或非零 `NODE_RANK` fail-fast，不能当作 LeapBot/论文规模的多机 launcher。

## 4. 环境准备

当前服务器的所有训练、评测和资产命令应由用户 `sheng` 执行；root 只用于安装驱动、系统库等系统级操作。这样可保证 checkpoint、W&B cache 和数据文件的属主一致。如果当前已经是 `sheng`，跳过第一行；只有处于 root shell 时才执行 `su - sheng`。

```bash
su - sheng
cd /home/sheng/workspace/leapbot-va
export ROOT_DIR="$(git rev-parse --show-toplevel)"

uv venv --python /usr/bin/python3.10 .venv
uv sync --dev
```

核心版本由 `pyproject.toml`/`uv.lock` 固定，包括 PyTorch 2.7.1+cu128、Accelerate 1.12.0、DeepSpeed 0.18.5 和 W&B 0.23.1。新机器还需要：

- 能运行 CUDA 12.8 构建 PyTorch 的 NVIDIA 驱动；
- GCC/G++ 和 DeepSpeed 扩展所需的开发工具；
- LIBERO 评测环境、MuJoCo 3.3.2 和可用的 EGL；
- 当前 wrapper 默认期望 LIBERO 在 `/home/sheng/workspace/LIBERO`。迁移到其他路径时设置 `LIBERO_ROOT=/path/to/LIBERO`；三套 canonical 评测 wrapper 会用它统一构造 fingerprint 和实际 evaluator 的 `PYTHONPATH`。

先做非 GPU 验收：

```bash
git status --short                 # 正式运行前必须为空
git rev-parse HEAD
uv run pytest -q
```

正式训练器默认拒绝 dirty worktree，也拒绝所选 GPU 已使用超过 2048 MiB 的情况。

## 5. 资产、T5 cache 与 provenance

### 5.1 下载固定 revision 的 FastWAM 资产

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT_DIR/checkpoints"
uv run python scripts/download_leapbot_assets.py --root "$ROOT_DIR"
```

该脚本固定下载：

- `yuanty/LIBERO-fastwam` 的四套 LeRobot 数据；
- `yuanty/fastwam` 的 LIBERO release checkpoint 和 dataset stats；
- 六个下载文件的 revision、字节数和 SHA-256 manifest。

它不读取本地 RLDS。四套训练数据目录是 `libero_spatial`、`libero_object`、`libero_goal` 和 `libero_10`，评测使用 `libero_10` 的 10 个任务。

### 5.2 下载 Wan VAE、T5 和 tokenizer

T5 与 tokenizer 会在下一步预计算时由与模型构造相同的 resolver 自动下载。正式 launcher 在启动前要求 Wan2.2 VAE 已存在，因此建议先显式解析并下载它：

```bash
DIFFSYNTH_MODEL_BASE_PATH="$ROOT_DIR/checkpoints" \
uv run python - <<'PY'
from fastwam.models.wan22.helpers.loader import _resolve_configs

_, _, vae, _ = _resolve_configs(
    model_id="Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
    redirect_common_files=True,
)
vae.download_if_necessary()
print(vae.path)
PY
```

默认下载源是 ModelScope；需要 Hugging Face 时可设置 `DIFFSYNTH_DOWNLOAD_SOURCE=huggingface`。最终 VAE 必须解析到：

```text
checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
```

### 5.3 预计算并严格验证 T5 cache

```bash
export LEAPBOT_DATASET_STATS="$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"

CUDA_VISIBLE_DEVICES=0 \
DIFFSYNTH_MODEL_BASE_PATH="$ROOT_DIR/checkpoints" \
uv run python scripts/precompute_text_embeds.py task=libero_leapbot_2cam224

CUDA_VISIBLE_DEVICES=0 \
DIFFSYNTH_MODEL_BASE_PATH="$ROOT_DIR/checkpoints" \
uv run python scripts/verify_text_cache_provenance.py \
  task=libero_leapbot_2cam224 \
  +text_cache_verification_device=cuda
```

第二条命令不是可选的 checksum 检查。它会用已解析的 T5/tokenizer 在线重新编码所有正式 prompt，并要求 cache 中 BF16 context 和 bool mask 在 shape、dtype、数值上逐元素完全一致，然后写入：

```text
data/text_embeds_cache/libero/.leapbot_text_cache_provenance.json
```

正式训练要求 provenance 的方法为 `online_source_forward_cache_tensor_exact`。生成/验证 cache 时取得独占文件锁；训练全程持有共享锁，防止 cache 在 epoch 中途被替换。

2026-07-31 在当前正式资产上已验证 40/40 个 cache 文件。可用于跨机器核对的身份是：

| 资产 | 已验证值 |
|---|---|
| training asset manifest | `886da5d91d6497bffb6b68de674cae5457175bbf9432bfd17c2aa794d019e977` |
| extracted dataset content | `aa5f87acb4df51ba3ff9aa4a671f64efd891bd17ddfc4912006f9b4d541bb361` |
| text cache | `4fad91546fe15c9fa04cb8d4ea08e8a758aead8c4273e87aaaff203621211332` |
| text cache provenance | `f4dfb1a72cfebb90c92557ba68d746c3e716bf66e1e65de0c1a0a4a673c5d646` |
| text encoder | `d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23` |
| tokenizer tree | `a8bc717cf013b7790af3b115681470a445fd2ac2b8e5ba750f1041f13ac54279` |
| VAE | `0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996` |

这些值只对相同正式资产成立；不要为了“匹配表格”而复制 provenance。重新生成 cache 后必须重新在线验证，并以新 manifest 为准。

每次 canonical 训练启动时都会在启动 Accelerate/W&B 之前重新散列正式数据、cache、VAE、release 权重和来源 manifest。因此命令刚启动后的几分钟内 W&B 还没有 run/curve 可能是正常的；应先看 shell 输出和 output directory 中的资产检查进度。散列完成且 rank 0 初始化 W&B 后才会出现线上 run。

### 5.4 哪些内容预计算，哪些内容在线计算

当前实现确实沿用 FastWAM 的静态编码缓存思路，但只缓存不会随训练更新的语言条件。当前 40 个 T5 `.pt` cache 的精确总大小为 42,074,760 bytes：

| 内容 | 预计算/在线 | 原因 |
|---|---|---|
| T5 prompt context 与 mask | 离线预计算 | 训练时 `load_text_encoder=false`，避免每步加载和前向 11.36 GB T5 权重 |
| release dataset stats | 官方预计算并复用 | 保持动作/proprio 归一化与 FastWAM 一致 |
| 当前及每个历史真实观测的 VAE latent | 每步在线 | 严格使用 rollout 同构的 batch-one、T=1 编码；不使用历史 latent 离线近似 |
| 未来视频监督 latent | 每步在线 | 对当前 T9 clip 在线编码，只服务 video flow-matching loss，不进入动作历史/KV |
| MP4 解码、双相机拼接、resize 和 normalize | 每步在线 | 数据 loader 的实际训练输入变换 |
| 历史动作归一化、proprio 处理 | 每步在线 | 必须与对应真实轨迹和 release stats 一致 |
| 历史视觉/动作 K/V | 训练时在线且保留计算图 | ActionDiT/VideoDiT 会更新；预计算 K/V 会使其 stale，并破坏 full BPTT |
| 推理期历史 K/V | episode 内在线增量缓存 | 只来自真实观测和实际下发动作，episode 结束 reset |

所以，当前没有预计算历史图像特征或动作特征。训练慢的主要代价正是对完整真实前缀做在线 VAE 编码和全梯度 causal 前向；这是模型语义的一部分，不应通过 detached cache 偷换。

canonical launcher 会强制注入并散列 release dataset stats。若绕过它直接运行 Trainer，且 `pretrained_norm_stats`/`LEAPBOT_DATASET_STATS` 为空，会触发全数据 normalization stats 重算：这既耗时，也改变实验身份。正式流程禁止绕过 launcher 或在 stats 为空时开跑。

因为 VAE 冻结，未来可以研究“严格绑定资产的 VAE latent cache”，但当前未实现、未验收，不能计入现有速度。按当前唯一数据帧粗算，T1 latent 约 1.0 GiB、T9 clip latent 约 3.0 GiB，raw 总量约 4.0 GiB；真正启用前必须把数据字节、像素变换、视频 decoder 版本、VAE 权重/hash 和 dtype 写入 manifest，并通过 online-vs-cache 的 loss 与全梯度等价测试。即使未来缓存 VAE latent，也不能缓存训练中的视觉/动作 K/V，因为 DiT 权重在更新。

## 6. DeepSpeed/Accelerate 拓扑

正式训练由 `scripts/train_leapbot.sh` 调用 Accelerate 和 `scripts/accelerate_configs/accelerate_zero2_ds.yaml`，再读取 `scripts/ds_configs/ds_zero2_config.json`：

- DeepSpeed ZeRO stage 2；
- optimizer state 和 gradient 在 ranks 间分片，模型参数仍在每卡复制；
- 不使用 CPU offload，也不使用 NVMe offload；
- BF16；
- ActionDiT 全量训练，VideoDiT 使用 rank-16 LoRA；
- DiT/MoT 开启 gradient checkpointing；
- AdamW，betas 0.9/0.95，weight decay 0.01，global grad clip 1.0。

当前可训练参数实测为 1.033B / 总 6.740B，其中 ActionDiT 与辅助参数约 1.0216B，VideoDiT LoRA 约 11.8M。

DeepSpeed JSON 的 batch 字段为 `auto`，但这不意味着拓扑未经检查。`Accelerator.prepare()` 后，Trainer 从实际 DeepSpeed engine 读取 micro-batch、GA、global batch 和 ZeRO stage，与 Hydra 配置逐项比较；任一不一致立即失败。日志中必须出现类似：

```text
Verified DeepSpeed topology: micro_batch_per_gpu=8 grad_accum=2 global_batch=128 world_size=8 zero_stage=2
```

每个 optimizer step 前还会收集所有 rank 的 gradient norm；任一 rank 非有限时在 `optimizer.step()` 前失败。

## 7. W&B 设置与曲线

canonical 训练入口默认 `WANDB_ENABLED=true`、`WANDB_MODE=online`。先以运行训练的同一用户登录：

```bash
uv run wandb login
export WANDB_ENTITY="<your-entity>"
export WANDB_PROJECT="leapbot-va"
export WANDB_ENABLED=true
export WANDB_MODE=online
```

W&B 由 rank 0 写入，每个 optimizer step 记录：

- `train/loss`、`train/loss_video_d30`、`train/loss_action_d30`；
- 多出口时各深度的 video/action loss；
- `train/history_blocks_mean`、`train/history_blocks_max` 及 EMA；
- `train/grad_norm`、各 optimizer group 的 LR；
- step/s、sample/s、峰值 allocated/reserved GPU memory。

launcher 用稳定的 `WANDB_RUN_ID` 和 `WANDB_RESUME=allow`。相同 output directory、相同 run contract 的断点恢复会接到同一条 W&B 曲线。若没有线上曲线，先检查对应 `train.log` 中是否出现 `Initialized wandb run`，再检查登录用户、entity/project 和网络；不要用一个新 run 手工转录旧日志冒充连续训练。

需要离线运行时可显式设置 `WANDB_MODE=offline`，之后用同版本 W&B CLI 同步本地 run。`WANDB_ENABLED=false` 只适合 smoke，不适合正式筛选。

## 8. 单机 8 卡完整流水线

以下命令均从仓库根目录执行，并使用显式输出目录，便于交接。先统一环境：

```bash
export ROOT_DIR="$(git rev-parse --show-toplevel)"
export DATASET_STATS="$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
export LEAPBOT_DATASET_STATS="$DATASET_STATS"
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT_DIR/checkpoints"
export LIBERO_ROOT="/home/sheng/workspace/LIBERO"
export WANDB_ENTITY="<your-entity>"
export WANDB_PROJECT="leapbot-va"
export WANDB_ENABLED=true
export WANDB_MODE=online
```

### 8.1 学习率配对筛选

两个候选都是 `action_aggregator`、完整历史、D30、100 optimizer steps、constant LR；它们使用相同 seed、初始化、数据顺序和 noise 顺序，仅 LR 不同。

```bash
export LR_SCREEN_ROOT="$ROOT_DIR/runs/lr_screen_full_history_s100"
SCREEN_ROOT="$LR_SCREEN_ROOT" \
MAX_STEPS=100 \
bash scripts/screen_learning_rate.sh
```

该入口同时启动：

- GPU 0–3：LR `1e-5`；
- GPU 4–7：LR `1e-4`。

筛选结束后，在一张空闲 GPU 上跑固定噪声审计：

```bash
export LR_AUDIT_ROOT="$ROOT_DIR/runs/lr_audit_full_history_s100"
SCREEN_ROOT="$LR_SCREEN_ROOT" \
OUTPUT_DIR="$LR_AUDIT_ROOT" \
FINAL_STEP=100 \
GPU_ID=0 \
bash scripts/audit_learning_rate.sh

export LR_SELECTION_MANIFEST="$LR_AUDIT_ROOT/learning_rate_selection.json"
```

审计覆盖 `H={0,1,4,8,16,32,50}`、correct/masked/cross-episode-shuffled 历史、固定 flow timestep 和固定 Gaussian noise，并用 bootstrap 生成选择 manifest。W&B 上的随机 minibatch loss 只能用于诊断，正式 LR 选择以该固定审计为准。

### 8.2 H0 保真筛选

H0 筛选不丢弃任何非零历史。`x4` 只重复真正的 episode-start/H0 样本；所有 H>0 样本仍带从 episode 开始的完整前缀。

```bash
export H0_SCREEN_ROOT="$ROOT_DIR/runs/h0_screen_full_history_s100"
LR_SELECTION_MANIFEST="$LR_SELECTION_MANIFEST" \
SCREEN_ROOT="$H0_SCREEN_ROOT" \
MAX_STEPS=100 \
bash scripts/screen_h0_retention.sh

export H0_AUDIT_ROOT="$ROOT_DIR/runs/h0_audit_full_history_s100"
LR_SELECTION_MANIFEST="$LR_SELECTION_MANIFEST" \
SCREEN_ROOT="$H0_SCREEN_ROOT" \
OUTPUT_DIR="$H0_AUDIT_ROOT" \
FINAL_STEP=100 \
GPU_ID=0 \
bash scripts/audit_h0_retention.sh

export H0_SELECTION_MANIFEST="$H0_AUDIT_ROOT/initial_block_oversample_selection.json"
```

### 8.3 三种 D30 causal 模式正式训练

```bash
export CAUSAL_TRAIN_ROOT="$ROOT_DIR/runs/causal_full_history_d30_s1115_seed42"

LR_SELECTION_MANIFEST="$LR_SELECTION_MANIFEST" \
H0_SELECTION_MANIFEST="$H0_SELECTION_MANIFEST" \
TRAIN_ROOT="$CAUSAL_TRAIN_ROOT" \
MAX_STEPS=1115 \
SAVE_EVERY=223 \
bash scripts/train_causal_modes.sh
```

该脚本用全部 8 张 GPU 顺序训练 `action_aggregator`、`interleaved`、`vision_causal`。每种模式都从同一个 FastWAM release 权重独立初始化；一个模式的终点不会作为下一个模式的起点。scheduler 为 5% warmup + cosine，seed 为 42。

需要分阶段占用集群时，可用同一个 `CAUSAL_TRAIN_ROOT` 和完全相同的两个选择 manifest，设置例如 `MODES_CSV=action_aggregator` 只跑一个模式；后续再跑其余模式。不要改变 topology、MAX_STEPS、SAVE_EVERY、seed 或资产。

### 8.4 dev10 与 final50

评测前安装 LIBERO/MuJoCo/EGL。开发筛选在 10 个 `libero_10` 任务上每任务 10 次，默认同时包含 FastWAM release baseline：

```bash
export DEV_EVAL_ROOT="$ROOT_DIR/evaluate_results/causal_full_history_dev10"
TRAIN_ROOT="$CAUSAL_TRAIN_ROOT" \
FINAL_STEP=1115 \
NUM_TRIALS=10 \
GPU_IDS_CSV=0,1,2,3,4,5,6,7 \
EVAL_ROOT="$DEV_EVAL_ROOT" \
bash scripts/evaluate_causal_modes.sh
```

最终表必须使用新的结果目录，每任务 50 次：

```bash
export FINAL_EVAL_ROOT="$ROOT_DIR/evaluate_results/causal_full_history_final50"
TRAIN_ROOT="$CAUSAL_TRAIN_ROOT" \
FINAL_STEP=1115 \
NUM_TRIALS=50 \
GPU_IDS_CSV=0,1,2,3,4,5,6,7 \
EVAL_ROOT="$FINAL_EVAL_ROOT" \
bash scripts/evaluate_causal_modes.sh
```

入口会验证三种 run contract 和 checkpoint，给每个 checkpoint/task/config 建 fingerprint，拒绝把不同 commit、权重、任务初始状态或运行参数的结果混在同一个 `EVAL_ROOT`。汇总结果位于 `EVAL_ROOT/pareto/results.csv`。

canonical 评测在 fingerprint 和实际 evaluator 两侧都固定 `EVALUATION.save_rollout_video=false`，避免 500/10,000-episode 评测被视频 I/O 和磁盘占用污染；如需定性视频，应在独立结果目录另跑，不得与正式 latency 树混用。

causal-mode winner 不是训练脚本自动指定的。应根据 dev/final 结果记录获胜 `MODE`、checkpoint step、结果目录和选择理由，再进入多出口阶段。

### 8.5 多出口训练

多出口从获胜 D30 `.pt` 权重开始一个新的 optimizer/scheduler，训练 step 从 0 重新计数。一次 30 层前向计算：

```text
L30 + (L8 + L16 + L24) / 3
```

每个 `Ld` 都含 video 和 action flow-matching loss。

```bash
export WINNER_MODE="<action_aggregator|interleaved|vision_causal>"
export MULTI_EXIT_STEPS="<freeze-this-before-running>"
export MULTI_EXIT_ROOT="$ROOT_DIR/runs/multi_exit_full_history"

SOURCE_TRAIN_ROOT="$CAUSAL_TRAIN_ROOT" \
MODE="$WINNER_MODE" \
SOURCE_STEP=1115 \
MAX_STEPS="$MULTI_EXIT_STEPS" \
TRAIN_ROOT="$MULTI_EXIT_ROOT" \
bash scripts/train_multi_exit.sh
```

`MAX_STEPS` 当前无隐式默认，必须在实验协议中先冻结。若决定给多出口与 D30 相同的 1115 次更新，应明确写 `MULTI_EXIT_STEPS=1115`；不要事后根据曲线改口径。

### 8.6 深度×KV 保留上限 Pareto

开发阶段可先设 `NUM_TRIALS=10`，最终报告使用 50：

```bash
export PARETO_ROOT="$ROOT_DIR/evaluate_results/depth_history_pareto_final50"

TRAIN_ROOT="$MULTI_EXIT_ROOT" \
MODE="$WINNER_MODE" \
FINAL_STEP="$MULTI_EXIT_STEPS" \
NUM_TRIALS=50 \
GPU_IDS_CSV=0,1,2,3,4,5,6,7 \
GRID_ROOT="$PARETO_ROOT" \
BASELINE_RESULTS_ROOT="$FINAL_EVAL_ROOT" \
bash scripts/evaluate_pareto.sh
```

默认网格为 `D={8,16,24,30} × KV-retention={0,8,16,32,full}`，共 20 个 LeapBot 配置；每配置 10 任务×50 次即 500 episodes，总计 10,000 个 LeapBot episodes。`kvret8` 表示只物理保留最近 8 个块的 K/V tensor；较新的 K/V 在生成时已经读取过更老前缀，因此它不是严格的“最后 8 块信息”消融。

汇总保留成功率—P50/P95 重规划延迟—峰值显存的全部非支配点。默认 LeapBot 只能从 memory-enabled LeapBot 行中选：先限制在成功率距最佳不超过 1 个百分点且置信区间重叠的配置，再取 P50 最低者；FastWAM baseline 不能被误标成 LeapBot 默认。若没有浅出口满足条件，就保留 D30 并如实报告速度目标未达到。重规划 profiler 分别记录真实观测前推、动作去噪和实际动作 K/V commit，不能用只计模型 kernel 的局部计时替代。

## 9. Run contract、checkpoint 与断点恢复

每个训练目录至少包含：

```text
run_contract.txt
training_asset_manifest.json
train.log
config.yaml
checkpoints/
  weights/step_XXXXXX.pt
  state/step_XXXXXX/
    trainer_state.json
    ... DeepSpeed optimizer/model shards and RNG state ...
checkpoint_validation.json
```

`run_contract.txt` 绑定：

- commit、release checkpoint、dataset stats；
- 四套数据、下载 manifest、T5 cache/provenance、T5/tokenizer/VAE hash；
- causal mode、B/GA/world/global batch、max steps；
- LR/scheduler、seed、optimizer、历史和出口配置；
- 上游 LR/H0 选择 manifest 的 SHA-256。

`.pt` 是便于评测/下游初始化的 portable weights；`state/step_*` 才包含 ZeRO-2 optimizer、scheduler、所有 rank RNG 和精确 dataloader cursor。需要续训时必须保留后者。

安全恢复方法是：在相同 commit、相同资产、相同输出目录上，重新执行逐字相同的 canonical 命令。launcher 只选择已经完整写出 `trainer_state.json` 的最新 state，忽略半写入目录；Trainer 再检查 run contract 和 dataloader topology。不要手工把 `.pt` 当作“断点恢复”，因为这只加载权重，不恢复 optimizer/scheduler/step。

`max_steps` 是 run contract 的组成部分。当前实现不支持把一个 1115-step run 原地改成 2000 steps 并声称是同一实验；应在开跑前冻结预算。不要用 `ALLOW_CROSS_CONTRACT_RESUME`、`ALLOW_DIRTY` 等逃生开关构造正式结果。

验证某个 final checkpoint 的关键证据：

```bash
grep '^run_contract_sha256=' "$CAUSAL_TRAIN_ROOT/action_aggregator/run_contract.txt"
grep 'Verified DeepSpeed topology' "$CAUSAL_TRAIN_ROOT/action_aggregator/train.log"
grep 'max_steps reached' "$CAUSAL_TRAIN_ROOT/action_aggregator/train.log"
```

canonical launcher 在结束时会运行 `validate_leapbot_checkpoint.py` 并生成 `checkpoint_validation.json`。缺少该文件、final state 或 `max_steps reached` 日志的 run 不应进入正式评测。

## 10. H800 实测资源与时长

以下数据均为 2026-07-31 当前实现。表中明确区分“实测”和“估计”；历史长度对时间/显存影响很大，不能用两个低 H step 外推完整训练承诺。

### 10.1 GPU、RAM 与吞吐

| 场景 | 状态 | 观测 |
|---|---|---|
| 8卡 ZeRO-2，B8/GA2/global128，两步 smoke | 实测 | step1 Hmean=5.7109/Hmax=8，113s，日志累计 1.13 samples/s；step2 Hmean=2.0938/Hmax=4，46s，累计 1.61 samples/s |
| 同一两步 smoke | 实测 | 峰值每卡 allocated 21.00 GiB，reserved 22.07 GiB；含资产合同、模型加载和 checkpoint 保存的总墙钟 5m31s |
| 双 4 卡 LR 正式筛选，B10/GA2/global80 | 2026-07-31 step8 运行中快照 | 当时见到 Hmean 最高 18.125；峰值每卡 allocated 32.66 GiB、reserved 34.51 GiB，`nvidia-smi` 约 36.9 GiB；不是 100-step 最终峰值 |
| 上述两个 run 同时运行 | 2026-07-31 step8 整机快照 | `free -h` 当时显示全机 used 158–159 GiB / total 1.8 TiB / available 1.6 TiB / 无 swap；这是系统级占用，不能全部归因到训练进程，也不是训练完成值 |
| 单卡 full-optimizer B10，真实 H41–H50，2 updates×2 microbatches | 显存压力实测 | action_aggregator 47.01/49.93 GiB、interleaved 47.63/50.10 GiB、vision_causal 49.02/51.21 GiB（allocated/reserved） |

上表的单卡 `full_prefix_smoke.py` 会完整执行真实图和 AdamW，但它不是
DeepSpeed engine，因此只能用作保守趋势证据，不能代替最终 8 卡拓扑验收。正式
扩大 micro-batch 前运行 checkpoint-free 容量探针：

```bash
su sheng -c '
  cd /home/sheng/workspace/leapbot-va &&
  MODE=action_aggregator \
  BATCH_SIZE=20 \
  OUTPUT_DIR=/home/sheng/workspace/leapbot-va/runs/acceptance/zero2_h41_h50_action_aggregator_b20 \
  bash scripts/probe_zero2_high_history_capacity.sh
'
```

该入口固定为 8 卡、ZeRO-2、BF16、D30、chunk1、
`incremental_full_bptt`、正式 ActionDiT/VideoLoRA optimizer，并执行恰好两次
optimizer update。每个 rank 的一个 micro-batch 包含 B 个不同的真实数据行；每行
都是同一 episode 内未截断、未合成的 H41–H50 完整前缀。固定批次只在 rank 和
update 之间重复，历史张量本身绝不复制延长。探针禁用 W&B，且不得写 portable
weights 或 DeepSpeed state；结果只写 `capacity_probe.json`、小型 config/contract
和日志。

探针和正式训练一样，对 T5 cache 持有整个进程生命周期的 shared `flock`，并在
启动 GPU 进程前生成 `training_asset_manifest.json`。完整数据、T5 cache/provenance、
VAE 和 pinned download manifest 的 hash 由该文件覆盖，其 `manifest_sha256` 同时
写入 `probe_contract.txt` 及成功/失败报告。optimizer 报告分别注明
`action_and_aux: weight_decay=0.01` 与 `video_lora: weight_decay=0`，不能笼统描述成
所有参数都使用 wd0.01。

默认 `PROBE_TIMEOUT_SECONDS=7200`，只允许正整数；超时会终止整个 Accelerate
进程组并写 `status=timeout`。每次 update 都检查 DeepSpeedEngine 的
`global_steps` 恰好增加 1、`_step_applied=true`、`skipped_steps` 不增加且
Accelerate 未报告 skipped；最终还断言 `global_steps` 总 delta 恰好为 2。报告会
扫描 `checkpoints/` 的实际文件列表，任何文件都会把结果升级为
`status=contract_violation`，即使训练计算本身成功也不能作为容量验收。

`MODE` 默认是 `action_aggregator`，且只接受三种正式值：
`action_aggregator`、`interleaved`、`vision_causal`。首个策略先按默认模式测 B20；
在另外两种模式正式开跑前，可用独立输出目录重复同一探针，尤其建议复验已有
单卡压力结果中峰值略高的 `vision_causal`。报告、输出目录默认名和
`probe_contract.txt` 都绑定实际 `MODE`，不同模式的结果不得覆盖或混用。

当前 release 数据对 B20 的确定性选择为 H44×8、H45×6、H46×2、
H47/H48/H49/H50 各 1（20 个不同数据行，平均 H45.4）；实际选择和每 rank
收到的数据行/H 列表都会写入报告，不能只凭配置推断。

先测 B20。若报告 `status=passed` 且最坏 rank 的 `global_peak_reserved_gib` 落在
约 70–75 GiB，可把 B20/GA1 作为候选。若 launcher OOM，它仍会聚合 rank OOM
证据并明确写出 `status=oom`；随后必须换全新目录回退 B18：

```bash
su sheng -c '
  cd /home/sheng/workspace/leapbot-va &&
  MODE=action_aggregator \
  BATCH_SIZE=18 \
  OUTPUT_DIR=/home/sheng/workspace/leapbot-va/runs/acceptance/zero2_h41_h50_action_aggregator_b18 \
  bash scripts/probe_zero2_high_history_capacity.sh
'
```

容量通过不等于可以悄悄改变实验合同。8 卡 B20/GA1 的 global batch 是 160；
B18/GA1 是 144；原 B8/GA2 和 B16/GA1 都是 128。最终选择若改变 global batch，
三种 causal mode 必须统一采用同一 B/GA/world、重新冻结每 epoch optimizer step、
scheduler/save 间隔和 run contract，不能从 global128 的 optimizer state 续训。

ZeRO-2 没有 CPU/NVMe offload。主机内存主要来自 8 个 rank、每 rank 3 个 dataloader workers、视频解码/预取、Python/模型元数据和系统 page cache。迁移到较小 RAM 节点时应实测，不应把 1.8 TiB 机器上的 available 数字当作最低要求。

### 10.2 时间规划

| 阶段 | 当前估计 |
|---|---|
| 两个 100-step LR run 并发 | 运行早期动态 ETA 约 3.5–4.5 小时；尚不是完成实测，后续高 H batch 可能改变 ETA |
| 100-step H0 配对筛选 | 拓扑相同，预计同量级；需用实际前 20–50 step 校准 |
| D30 正式 1115-step 单模式，8×H800 | 约 36–60 小时/模式，仅为容量规划估计 |
| 三模式顺序训练 | 约 108–180 小时，即约 4.5–7.5 天，不含筛选、审计和评测 |
| 多出口训练 | 尚无可负责的端到端实测；四深度 loss 和 B1/GA16 会改变吞吐，应在前 50 step 后重新估计 |
| LIBERO 评测/Pareto | 强依赖 simulator reset、任务 episode 长度和 inference steps；以 episode 数规划，不给未经实测的小时数 |

正式训练启动后，应在 step 50 用 `train.log` 的累计 step/s、Hmean/Hmax 和最近窗口墙钟重新给 ETA。首个 step 包含 kernel/allocator warmup，不能单独用于外推。

### 10.3 磁盘

当前输入资产的已验证字节量大致为：

- 四套 extracted 数据 4.73 GB，下载 tar 另约 4.69 GB；
- FastWAM release checkpoint 12.04 GB；
- T5 11.36 GB、VAE 1.41 GB、tokenizer 约 21 MB；
- 40 个 T5 cache 文件约 42 MB。

合计约 34 GB，不含 `.venv`、Hugging Face/ModelScope 元数据和文件系统开销。

8卡 ZeRO-2 两步 smoke 的一个完整 checkpoint 实测约 46 GiB：portable weights 约 12.1 GB，DeepSpeed model state 约 24.9 GB，8 个 optimizer shards 合计约 12.4 GB。按 `SAVE_EVERY=223` 保存 1115 steps 会有 5 个 checkpoint，约 230 GiB/模式，三模式约 690 GiB。正式集群至少应在此基础上为多出口、筛选和评测留余量。

可以在确认 final checkpoint 已复制、验证且不再需要从中间 step 恢复后归档旧 state，但 final `state/` 和 `checkpoint_validation.json` 必须保留；评测中间 checkpoint 时也需要对应 state 通过身份验证。不要在训练写 checkpoint 时并发移动或删除 shard。

## 11. 更大 GPU 集群的扩展边界

当前 canonical reference 是单机 8 卡，Accelerate 配置明确为 `num_machines: 1`。更大集群可提升并行实验数，但不能直接把环境变量改成 16/32 卡后仍称为同一 reference。

当前代码没有完成多节点端到端验收，因此本节是扩展检查表，不是可直接复制运行的多机命令。完成 rank-0 合同/W&B、rendezvous、共享存储与跨节点资产验证工程并跑通验收之前，不应宣称“支持多节点正式训练”。

### 11.1 最安全的扩展：实验级并行

优先保持每个 run 的 8卡/B8/GA2/global128 不变，在不同节点并行跑三种 causal mode、不同 seed 或评测配置。这样单个 run 的数值拓扑最接近当前 reference。

需要为每个 job 保证：

- 相同 commit、release 权重、dataset stats、LR/H0 选择 manifest 和资产 manifest；
- 独立 GPU allocation、端口、output subdirectory 和 W&B run ID；
- 三个 causal mode 最终仍通过 `validate_run_contract_group.py` 的共同字段验证；
- 若共享一个 `TRAIN_ROOT` 并发建 contract，要避免共同 validation 文件的写入竞争。生产调度适配器应为每个 mode 做原子目录管理，或先逐个创建 contract 再并发训练。

### 11.2 扩大单个 run 的 world size

`train_causal_modes.sh` 当前故意强制 8 GPU×B8×GA2。要改成多机或更多卡，必须作为一个新的训练协议修改并重新验收：

1. 创建多机 Accelerate 配置，正确设置 `num_machines`、`machine_rank`、主节点地址/端口和 rendezvous；
2. 调整 B/GA，使目标 global batch 明确；
3. 更新 canonical wrapper 的 topology guard 和 run-contract 预期；
4. 在所有被比较模式上使用完全相同的新 topology；
5. 重做两步 ZeRO-2 smoke、resume smoke、sampler 尾部和 checkpoint 验收；
6. 重新做 LR 筛选。即使 global batch 不变，world-size/sharding 改变也会改变每 rank 数据/noise 分配，不能默认沿用 8 卡最优 LR。

ZeRO-2 仍会在每卡复制全部模型参数，增加 GPU 数主要缩小 optimizer/gradient shard，不会按 GPU 数线性消除 activation 和完整历史的显存。单卡显存预算仍应按 B、D 和最坏 H 做验收。

### 11.3 数据、cache、NCCL 和共享存储

- 预先把四套数据、release 权重、T5/tokenizer/VAE 和 cache stage 到每个节点的相同逻辑路径；不要让所有节点在作业开始时同时下载。
- 数据和 text cache 在训练期间必须只读。共享文件系统需支持可靠的 POSIX `flock`；若不支持，应在调度层冻结资产并逐节点验证 manifest。
- 多机上要配置 NCCL 网卡、IB/RoCE、firewall 和 rendezvous 端口，并先跑两步 smoke。
- checkpoint 应写到吞吐足够的共享存储；单 checkpoint 约 46 GiB，保存时会产生明显 I/O 停顿。
- `CUDA_VISIBLE_DEVICES` 应由调度器分配；不要在多个同节点 job 中复用相同 `MAIN_PROCESS_PORT`。
- 训练 wrapper 的本地 `nvidia-smi` preflight 只能看到本机，不能替代集群调度器的全局资源隔离。

## 12. 验收与实现检查入口

建议接手者按下面路径审代码：

| 关注点 | 实现/测试 |
|---|---|
| 完整 episode 历史采样、真实动作和 padding 拒绝 | `src/leapbot_va/data.py`、`tests/test_incremental_full_bptt_training.py` |
| causal mask、未来信息隔离、完整梯度 | `src/leapbot_va/training.py`、`tests/test_causal_masks.py`、`tests/test_packed_training_masks.py` |
| episode 位置和 RoPE | `src/leapbot_va/positions.py`、`tests/test_hierarchical_positions.py` |
| 显式 KV memory 状态机和事务回滚 | `src/leapbot_va/memory.py`、`src/leapbot_va/runtime.py`、`tests/test_memory.py`、`tests/test_inference_contract.py` |
| 实际执行动作的后处理/重归一化 | `src/leapbot_va/libero.py`、`tests/test_libero_action_commit.py` |
| ZeRO-2 拓扑、sampler、resume、W&B | `src/fastwam/trainer.py`、`src/fastwam/utils/samplers.py`、对应 trainer/sampler tests |
| T5/VAE/tokenizer/cache 身份 | `src/leapbot_va/conditioning_assets.py`、`tests/test_conditioning_assets.py` |
| 评测 fingerprint 和 Pareto | `src/leapbot_va/eval_fingerprint.py`、`experiments/leapbot/pareto.py` |

每个新 commit 的最低 CPU 验收是：

```bash
uv run pytest -q
bash -n scripts/*.sh scripts/fastwam_legacy/*.sh
```

有正式资产和空闲 H800 时，建议每个硬件/依赖栈至少跑一次真实 6B 训练—在线 KV 等价验证：

```bash
for mode in action_aggregator interleaved vision_causal; do
  CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/validate_real_6b_runtime_training_equivalence.py \
    --checkpoint "$ROOT_DIR/checkpoints/fastwam_release/libero_uncond_2cam224.pt" \
    --dataset-stats "$DATASET_STATS" \
    --causal-mode "$mode" \
    --history-blocks 8 \
    --device cuda:0 \
    --dtype bf16 \
    --output-json "$ROOT_DIR/runs/acceptance/${mode}_h8.json"
done
```

再用 canonical base 做 8卡 ZeRO-2 两步 smoke：

```bash
SMOKE_ROOT="/tmp/leapbot-zero2-w8-b8-ga2-$(git rev-parse --short HEAD)"
MODE=vision_causal \
NUM_PROCESSES=8 \
GPU_IDS_CSV=0,1,2,3,4,5,6,7 \
BATCH_SIZE=8 \
GRAD_ACCUM=2 \
MAX_STEPS=2 \
SAVE_EVERY=2 \
LEARNING_RATE=1.0e-5 \
LR_SCHEDULER_TYPE=constant \
OUTPUT_DIR="$SMOKE_ROOT" \
WANDB_ENABLED=false \
WANDB_MODE=disabled \
REQUIRE_SELF_IDENTIFYING_CHECKPOINT=true \
bash scripts/train_leapbot.sh
```

检查 `run_contract.txt` 的 `8/8/2/128`、日志中的 ZeRO stage 2、有限 grad norm、峰值显存、`trainer_state.json` 的 step/cursor，以及 `checkpoint_validation.json`。

## 13. 常见误判和故障定位

### W&B 没曲线

先看训练进程是否真正到 optimizer step，而不是还在 hash 资产、加载 6B 模型或保存 checkpoint。再检查 `train.log` 的 W&B 初始化、当前用户的登录凭证和 `WANDB_MODE`。smoke 若显式关闭 W&B，本来就不会有线上曲线。

### action loss 不能和 FastWAM 曲线直接对齐

`train/loss_action_d30` 只有在相同样本、历史、noise、flow timestep、mask、global batch 和权重初始化下才可直接做数值归因。全历史训练的随机 W&B step 与 FastWAM H0 reference 不满足这些条件。保真比较应使用 `audit_learning_rate.sh`/`audit_h0_retention.sh` 的固定噪声、native H0 和 masked/shuffled controls。

### video loss 初值在不同 causal mode 差异大

`action_aggregator` 的视觉块独立，最接近 release 初始化的视觉路径；`interleaved`/`vision_causal` 让未适配的视觉专家首次读取长前缀，初始 video loss 可能更大。应比较控制训练后的效果，不应把不同 H 分布的第一个随机 step 当作架构错误结论。

### OOM

先同时查看 `history_blocks_max`、allocated/reserved 和 `nvidia-smi`。显存随 H、B 和出口数变化。新的非正式拓扑可降低 B、提高 GA 以保持 global batch，但三种正式模式必须一起采用同一新拓扑并重做 LR/验收。不要把 `history_vae_batch_chunk_size` 改为 2；正式 runtime 同构合同固定为 1。

### 资产或 contract mismatch

不要覆盖旧输出目录。检查是否换了 commit、stats、cache、VAE、选择 manifest、MAX_STEPS 或 topology；使用新的 output root。cache 发生任何重算后重新跑在线 provenance 验证。

### 评测目录被拒绝

这通常意味着目录里已有不同 checkpoint/config/task 初始状态的结果。保留旧目录用于审计，换一个全新的 `EVAL_ROOT`；不要删除 fingerprint 后强行混合结果。

## 14. 交付清单

一次可交接的完整实验至少应打包或登记：

- 仓库 commit 和干净 worktree 证明；
- `uv.lock`、GPU/驱动、PyTorch/DeepSpeed/Accelerate 版本；
- 下载 manifest、training asset manifest、T5 cache provenance；
- LR/H0 固定噪声 audit JSON 与两个 selection manifests；
- 每个 causal mode 的 `run_contract.txt`、`train.log`、W&B URL、final `.pt`、final full state 和 validation JSON；
- dev10、final50 的完整 fingerprint/result tree 与汇总表；
- multi-exit source lineage、训练合同和 final state；
- D×KV 网格结果、逐任务成功率、P50/P95 latency、峰值显存、cache 大小和最终 Pareto 图；
- 所有“实测”和“估计”时长/资源的明确标注。

只有这些证据齐全，后续团队才能区分“模型效果差异”“训练拓扑差异”“资产漂移”和“评测环境漂移”。
