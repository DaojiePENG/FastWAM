# LeapBotCE 全 LIBERO 训练进度跟踪

> 用途：跟踪全 LIBERO（四套件联合）CE 训练的进度、配置、时间与检查点位置。
> 由训练看守脚本启动，人工/助手按训练节点回填"进度记录"表格。
> 最后更新：2026-09-30（训练迁移公司 AIFlow 平台，见第 7 节；110k 检查点与 50k dtrain20 全量评测并行进行中）

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

## 3. 时间估算（2026-09-22 实测修正）

- 数据 277,713 帧 / 有效 batch 16 ≈ **17,357 步/epoch**；50k 步 ≈ **2.9 epoch**
- 实测吞吐（已完结）：d=20 约 **10.5 samples/s**（单轮 ≈21 h）；d=40 约 **7.7 samples/s**（窗口更大，单轮 ≈29 h）；两轮串行合计约 50 h
  （第二轮开始时若空闲卡不足，看守脚本会每 5 分钟自动重试等待）

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
| d=20 | ✅ 完成 | 2026-09-22 18:59 | 2026-09-23 16:03（约 21 h） | 2,3 | 8×1（探针峰值 64.2GB，余量 17.3GB） | 10.55 samples/s（收尾） | 50k/50k 步，最终 loss 0.0454，delay 均值 11.1；wandb [run 5t78mito](https://wandb.ai/yabi/leapbotce/runs/5t78mito) |
| d=40 | ✅ 完成 | 2026-09-23 16:12 | 2026-09-24 21:21（约 29 h） | 看守自动重选（bs=8 重新标定通过） | 8×1 | 7.67 samples/s（收尾，d=40 窗口更大故稍慢） | 50k/50k 步，最终 loss 0.0557，delay 均值 21.3；wandb [run v5je6j5q](https://wandb.ai/yabi/leapbotce/runs/v5je6j5q) |
| d=20 加长 | ⛔ 中止（09-29 23:56 NCCL watchdog SIGABRT，rank 0 崩溃；约 step 110k+） | 2026-09-28 13:52 | 09-29 23:56（运行 ~34 h） | 0,1,2,3 | 4×1（有效 batch 16） | 收尾 0.80 step/s（12.9 samples/s；节点负载高所致） | 200k 计划中止于 ~110k 步（≈5.5 epoch，stale warmup 100k 已过）；崩溃前最后检查点 **step_110000.pt**（09-29 23:23）；90k/100k/110k 的 DeepSpeed state 均在，可 --resume 续训（需 4 卡）；wandb name=`libero_all_leapbotce_dtrain20_seed42_200k` |

最终检查点（均已确认落盘，各 2.0GB delta 权重）：
- `/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce/dtrain20_seed42/checkpoints/weights/step_050000.pt`
- `/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce/dtrain40_seed42/checkpoints/weights/step_050000.pt`
- d=20 加长轮（进行中）：`/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce/dtrain20_seed42_200k/checkpoints/weights/`（每 10k 步一存，保留最近 3 个 + 最终 step_200000）

启动流程回放（2026-09-22）：18:53 看守启动 → 18:53 选中空闲卡 2,3 → 18:54 文本嵌入确认就绪（40 条全缓存）→ 18:55-18:57 bs=8 标定通过（3 步冒烟，峰值 64.2GB）→ 18:59 正式训练启动，wandb 在线同步成功 → 19:04 确认稳定运行（step 180，loss 0.234，delay 均值 11.8，stale 权重爬升正常）。

d=20 于 09-23 16:03 正常完成后，看守自动按 d=40 窗口重新标定（bs=8 通过）并于 16:12 启动第二轮，09-24 21:21 两轮计划全部完成，看守脚本正常退出。

## 6. 全量时延评测（对齐 baseline 协议，进行中）

**2026-09-28 启动**：两个已训完模型（dtrain20_seed42、dtrain40_seed42，各 50k 步）的
完整 delay 鲁棒性评测，协议与 `docs/baseline_results_libero.md` 中各 baseline 对齐：

| 项 | 值 |
| --- | --- |
| 时延上限 D | 0 / 5 / 10 / 20 / 40（D>0 时每 episode 从 U{1..D} 采样，只延迟云端流） |
| 规模 | 每模型 × 每 D：4 套件 × 10 任务 × 50 trials = 500 episodes/套件 |
| 种子 | seed=1234（对齐 baseline 的 delay_seed 约定） |
| 其它 | replan=10、wait=30、action_horizon=32、num_inference_steps=10、binarize_gripper=true |
| GPU | 09-30 起改为：110k 检查点评测在 GPU 0-3（每卡 1 worker），50k dtrain20 sweep 迁到 GPU 1-3 第二槽位（GPU 4-7 被其它任务占满，原驻留的 3 个 worker 已迁出） |
| 预计时长 | 110k 评测（D∈0,10,20,40，160 任务单元）：~26 h @4 workers；50k dtrain20 补漏+后续（~126 单元）：~27 h @3 workers；之后 dtrain40-50k 继续 |
| 驱动脚本 | 110k：`scripts/eval_leapbotce_110k_2653853.sh`（日志 `driver_110k.log`）；50k：`scripts/eval_leapbotce_full_sweep_2653853.sh`（日志 `driver3.log`）——均幂等断点续跑 |
| 结果目录 | 50k：`leapbotce/dtrain20/`、`leapbotce/dtrain40/`；110k 加长轮：`leapbotce/dtrain20_seed42_110k/`（报告生成器新增 "LeapBotCE (d_train=20, 110k)" 行与 50k 并列对比） |
| 汇总 | 每 sweep 结束自动跑 `scripts/summarize_leapbotce_delays.py` → `delay_sweep_summary.json`；全部结束后自动重生成 `docs/baseline_results_libero.md`（LeapBotCE 行与其它 baseline 并列） |
| 监控 | `tail /mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce/driver2.log`；单 delay 日志在 `leapbotce/logs/` |

**10-01 11:40 状态快照（200k 加长训练未完成，评测未扩容）**：200k 加长训练已于
09-29 23:56 NCCL SIGABRT 中止（见第 5 节，⛔ 行），未触发评测扩容条件；当前
110k 检查点评测**已全部结束**（driver_110k.log 10-01 09:40，报告已重生成），
50k dtrain20 sweep 在 delay_5（30/40，GPU 1-3 每卡 1 worker，管理进程 662678），
dtrain40 50k sweep 仍未启动（SKIP_D40=1，等空闲卡）。空闲卡实况：GPU 4/5/6 全空
（RoboTwin 文本嵌入预计算 10-01 04:10 完成），GPU 0 有 110k 残留 26.5GB（本容器
命名空间无对应进程，约 55GB 可用），GPU 7 桌面驻留 7.8GB。

**09-28 晚间事故与恢复**：首跑 driver 在 22:03-23:22 期间因节点负载过高（load>300）
+ 一次存储卡顿（manager 进程 D 状态挂死、worker 状态文件 14h 不更新）级联失败：
dtrain20 delay_0 完成 38/40、delay_20 完成 3/40，dtrain40 delay_0 完成 30/40，其余空。
09-29 处理：杀掉挂死进程组；`run_libero_manager.py` 加了断点续跑（建队列时跳过
已有结果的任务，缺多少补多少）；结果目录改为 `leapbotce/dtrain20`、`leapbotce/dtrain40`
（报告生成器按 dtrain20/dtrain40 子串匹配，不受影响）；GPU 4-7 目前有其它会话的任务
驻留（约 14-50GB/卡），故以 GPUS=4,5,6、NUM_GPUS=3（每卡 1 worker）重启，先跑
dtrain20（SKIP_D40=1），dtrain40 等 10-01 训练结束、卡空闲后再补跑。

评测入口命令备忘（单模型手动版，驱动脚本内部等价实现）：

```bash
CKPT=/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce/dtrain20_seed42/checkpoints/weights/step_050000.pt
OUT=/mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce/dtrain20_seed42/delay_0
CUDA_VISIBLE_DEVICES=4,5,6,7 python experiments/libero/run_libero_manager.py \
  task=libero_all_leapbotce ckpt="${CKPT}" seed=1234 \
  model.load_text_encoder=true model.skip_dit_load_from_pretrain=false \
  EVALUATION.num_trials=50 EVALUATION.max_delay_steps=0 \
  EVALUATION.action_horizon=32 EVALUATION.num_inference_steps=10 \
  EVALUATION.replan_steps=10 EVALUATION.output_dir="${OUT}" \
  'MULTIRUN.task_suite_names=[libero_spatial,libero_object,libero_goal,libero_10]' \
  MULTIRUN.num_gpus=4
python scripts/summarize_leapbotce_delays.py /mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce/dtrain20_seed42
```

注意：200k 加长模型训完后需对其再跑一轮同协议评测（预计再 ~15 h @4 卡），
结果目录约定 `baseline-evaluations/leapbotce/dtrain20_seed42_200k/`。

## 7. 训练迁移：公司 AIFlow 平台（2026-09-30 起）

调试服务器不再续训；后续加长训练（11.5 epoch 全 LIBERO）改走公司 AIFlow 自动
申请显卡训练平台。配置文件：**`FastWAM/AIFlow_config.py`**（放在仓库根目录，
申请资源时工作目录选到 FastWAM 根目录即可，平台按固定文件名自动拉起训练）。

| 项 | 值 |
| --- | --- |
| 默认资源 | 8×H100/H200 自动适配（每卡 batch 按显存阈值表选：≥128GB→16，≥72GB→8；卡数读平台 GPU_NUM） |
| 有效 batch | H100: 64（8×bs8）/ H200: 128（8×bs16）；H100-80G bs8 实测峰值 64.2GB，H200 bs16 预计 ~115-130GB |
| 总步数 | 自动按 11.5 epoch × 277,713 帧 ÷ 有效 batch 折算：H100 ≈49.9k，H200 ≈25.0k（可 AI_FLOW_MAX_STEPS 覆盖） |
| 学习率 | 自动按 √batch 从官方 1e-4@16 折算：有效 64→2e-4，128→2.8e-4（可 AI_FLOW_LR 固定，如复现 bs8/2e-4 配方） |
| 其余超参 | cosine、wd 1e-2、clip 1.0、stale λ=0.5、warmup 自动=总步数/2；checkpoint 按个数自动均分：`AI_FLOW_NUM_SAVE=5`（含最终 1 个），间隔=总步数÷5（H100≈9,981 步/个，H200≈4,991 步/个），最终步必存（trainer.py 结尾无条件 save） |
| 训练入口 | 复用 `scripts/train_leapbotce_libero_2653853.sh`（同一入口；多机经 NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT 透传给 train_zero1.sh） |
| wandb | 打开 `--wandb`，project=leapbotce，name=libero_all_dtrain20_seed42_h100；代理经 PROXY_ENABLED/PROXY_HOST/PROXY_PORT 透传（启动脚本新增 PROXY_HOST 支持，默认 127.0.0.1 仅调试机有效；训练节点需设调试机内网 IP 或直连外网） |
| NCCL | 集群稳定性四件套（CUMEM=0、ASYNC_ERROR_HANDLING=1、TIMEOUT=1200、GDR_LEVEL=LOC）已内置，勿动 |
| 平台日志 | 工作目录下 `jinnTrainResult/<job>/train.log`（rank0）；checkpoint 仍在 `/mnt/c2-worldmodel/2653853/leapbot-ce/runs/libero_all_leapbotce/libero_all_dtrain20_seed42_h100/` |

已验证：模拟平台 8 卡调用生成的 CMD 通过 `--dry-run` 全链路解析；断点续训用
`AI_FLOW_RESUME` 填 DeepSpeed state 目录。

**09-30 首次提交（8×H200，leapbot_ce_train-8329b664）失败与修复**：训练本体正常
（DeepSpeed ZeRO-1 8 卡、bs16/有效 128 起来并跑了部分步），但 rank0 的
`wandb.init` 在线模式被死代理（节点上 127.0.0.1:33716 拒连）卡 90s 后抛
CommError，其余 7 rank 在梯度 ALLREDUCE 上等满 torch 默认 NCCL 600s 超时被
watchdog 全杀 → 平台判 failed。细节：wandb 0.23 的 online init 失败会留下僵死
服务状态，rank0 连退出清理都挂死（事故时 ps 里 9 个 train.py 的谜底）。
三层修复：
1. `src/fastwam/trainer.py::_init_wandb`：进 online 前先探活 api.wandb.ai
   （2×10s，不校验证书，urllib 自动走代理环境变量），探不通直接 offline——
   根本不给 wandb 在死网络里重试的机会；wandb.init 仍失败则禁用 wandb 并
   teardown 清理，训练永不因遥测中断；`WANDB_MODE` 环境变量优先于 config
   （显式 online 不拦；空串/非法值会被清掉——wandb 0.23 的 pydantic 校验
   会因非法值直接抛错，这是测试中发现的坑）；
2. `scripts/train_leapbotce_libero_2653853.sh`：代理连通性检测失败且未显式设
   WANDB_MODE 时自动 `WANDB_MODE=offline`（曲线落盘 run 目录，事后
   `wandb sync <run目录>/wandb/latest-run` 补传）；
3. AIFlow 配置注释记录该坑。注意 `NCCL_TIMEOUT` 环境变量 torch 并不识别
   （实际生效的是 torch 默认 600s），此前是无效导出。
验证：leapbot-va 环境实测"offline 直启"与"死代理探活降级"两条路径均干净
初始化/退出（rc=0）；训练本体在失败任务中已证明正常（ZeRO-1 8 卡 bs16 起来
并跑了数百步），本次修复只动 wandb 遥测路径，不碰训练。
