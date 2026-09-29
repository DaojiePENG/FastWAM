# LeapBotCE 全 LIBERO 训练进度跟踪

> 用途：跟踪全 LIBERO（四套件联合）CE 训练的进度、配置、时间与检查点位置。
> 由训练看守脚本启动，人工/助手按训练节点回填"进度记录"表格。
> 最后更新：2026-09-22 19:00（训练启动时）

## 1. 实验概览

| 项 | 值 |
| --- | --- |
| 任务 | LeapBotCE 全 LIBERO 四套件联合训练（P0 主实验） |
| 计划 | 先 `d_train=20`，正常结束后串行 `d_train=40`；各 50,000 步，seed=42 |
| 数据 | spatial + object + goal + 10（no_noops），共 277,713 帧 / 1,712 条 demo |
| 数据来源 | `/mnt/c2-worldmodel/training_data/LIBERO-fastwam/`（与 FastWAM 官方 `libero_uncond_2cam224_1e-4` 检查点同源） |
| 有效 batch | 16（= 卡数 × 每卡 batch × 累积；每卡 batch 由启动探针标定，预期 8×1×2） |
| 学习率 | 1e-4，cosine，weight decay 1e-2，grad clip 1.0，bf16，ZeRO-1 |
| stale 损失 | `stale_loss_lambda_max=0.5`，warmup 自动 = max_steps/2 = 25,000 步线性爬升 |
| 云端 KV 适配器 | 关闭（基线配方；KV 适配器实验另开配置） |
| wandb | project=`leapbotce`，name=`libero_all_leapbotce_dtrain20_seed42` / `..._dtrain40_seed42`（走临时代理 127.0.0.1:33716 在线上报） |

## 2. 脚本与配置文件

| 文件 | 说明 |
| --- | --- |
| `scripts/train_leapbotce_libero_2653853.sh` | 训练启动脚本（`--task all` 全 LIBERO；含代理与 wandb 开关，`--help` 看全部参数） |
| `scripts/watch_and_train_libero_all_2653853.sh` | 看守脚本：等空闲卡 → 预计算文本嵌入 → 显存标定（bs 8→4→2）→ 启动 → 等结束 → 下一轮 |
| `configs/data/libero_all_cloudedge.yaml` | 四套件 CloudEdge 数据配置 |
| `configs/task/libero_all_leapbotce.yaml` | 全 LIBERO 训练任务配置 |
| `configs/task/libero_all_leapbotce_smoke.yaml` | 冒烟配置（10 步） |

## 3. 时间估算

- 数据 277,713 帧 / 有效 batch 16 ≈ **17,357 步/epoch**；50k 步 ≈ **2.9 epoch**
- 吞吐：以探针实测为准（写在下方进度表），估算参考 ~29 h/epoch（2 卡）
- 单轮 50k 步 ≈ **3.5 天**；两轮串行 ≈ **7 天**（若第二轮开始时无空闲卡，看守会自动等待）

## 4. 检查点与日志位置

| 内容 | 路径 |
| --- | --- |
| d=20 运行目录 | `/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce/dtrain20_seed42/` |
| d=40 运行目录 | `/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce/dtrain40_seed42/` |
| 权重 delta（评测用，约 2GB） | `<运行目录>/checkpoints/weights/step_050000.pt` |
| DeepSpeed 可恢复状态（约 27GB，保留最近 1 份） | `<运行目录>/checkpoints/state/` |
| 看守总日志 | `/mnt/c2-worldmodel/2653853/leapbot-ce/runs/watch_launch.log` |
| 每轮训练日志 | `/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce/train_<run_name>_<时间戳>.log` |
| 显存标定日志 | `/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce/probe_d*_bs*_<时间戳>.log` |
| 文本嵌入缓存 | `/mnt/c2-worldmodel/2653853/leapbot-ce/text_embeds/libero/`（40 条指令已就绪） |

注意：评测 LeapBotCE delta checkpoint 时必须设 `model.skip_dit_load_from_pretrain=false`
（见 LEAPBOTCE.md），评测入口 `scripts/eval_leapbotce_delay_sweep.sh`。

## 5. 进度记录

| 轮次 | 状态 | 启动时间 | 完成时间 | GPUs | 每卡 bs×累积 | 实测吞吐 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| d=20 | 排队/标定中 | - | - | - | - | - | 已确认 50k 步，等待看守脚本选中空闲卡 |
| d=40 | 待 d=20 完成 | - | - | - | - | - | 串行 |

## 6. 训练完成后评测入口（备忘）

```bash
CKPT=/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce/dtrain20_seed42/checkpoints/weights/step_050000.pt
OUT=/mnt/c2-worldmodel/2653853/leapbot-ce/evaluate_results/leapbotce_all_dtrain20_seed42
for delay in 0 5 10 20 40 80; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/libero/run_libero_manager.py \
    task=libero_all_leapbotce ckpt="${CKPT}" seed=42 \
    EVALUATION.num_trials=50 EVALUATION.max_delay_steps="${delay}" \
    EVALUATION.action_horizon=32 EVALUATION.num_inference_steps=10 \
    EVALUATION.replan_steps=10 \
    EVALUATION.output_dir="${OUT}/delay_${delay}" MULTIRUN.num_gpus=4 \
    model.load_text_encoder=true model.skip_dit_load_from_pretrain=false
done
python scripts/summarize_leapbotce_delays.py "${OUT}"
```
