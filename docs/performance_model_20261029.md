# 2026-10-29 性能模型配方与部署方法

本轮测试测量真实硬件上的计算、通信时延和吞吐量。原论文 a6 的训练后 delta 在当前测试床未找到，按照用户授权，以相同 a6 结构装配性能模型。首次从共享盘读取 12 GB LIBERO checkpoint 的过程因 I/O 延迟终止；最终使用云机 NVMe 上的公开 RoboTwin 完整 MoT 权重，并以 LIBERO checkpoint 的小型 I/O 参数覆盖 action encoder、action head、video head 和本体编码器，使输入输出形状保持 a6 LIBERO 配方。新 codec、边侧 projector 和 rank-16 LoRA 未训练；这些结果支持所执行结构的系统性能分析，不能替代 a6 任务成功率或 retention 结果。测试不驱动机械臂。后续提供 a6 后，可直接替换 delta 并复测同一 profiling 流程。

## 模型结构与初始化

| 组件 | 初始化和固定结构 |
|---|---|
| 云端 Wan 视频专家与端侧 ActionDiT | 本地 `robotwin_uncond_3cam_384.pt` 的完整 MoT 专家；action encoder/head、video head 与 proprio 用 `libero_uncond_2cam224.pt` 对应参数覆盖 |
| T5、VAE、SigLIP | 测试床已有本地预训练权重 |
| 视频专家 LoRA | q/k/v/o，30 层 × 4 = 120 个 wrapper，rank 16，alpha 16；A 随机、B 为零，初始更新恰为零 |
| Planning codec | a6 同构，7×8 = 56 tokens，width 3072，bottleneck 1024，tied K=V，关闭 K/V decoder；随机初始化，30 层 scale 初值为 1 |
| 边侧视觉前缀 | select/topk，每视图 5 tokens，两视图共 10 tokens；projector 未训练 |
| 推理形状 | batch 1；两张 224×224 RGB 相机图；8 维本体；输出 32×7 动作块；10 次 Euler 去噪；bf16 |
| Codec 载荷 | 56×3072×2 = 344,064 bytes；不含 JSON 帧头和 HTTP 开销 |
| Raw 载荷 | 2×30×98×3072×2 = 36,126,720 bytes；是同一模型的无压缩接口对照 |

初始化 seed 为 0。完整 MoT 在 CPU 上加载后才注入 LoRA，避免 wrapper 改变参数键名后再次加载完整 checkpoint 导致基座权重遗漏。通过 `mot_pretrained_path` 设置基座并打开 `skip_dit_load_from_pretrain`，省略先加载 vanilla Wan/ActionDiT 再覆盖的重复过程。`redirect_common_files=true` 选用现成 converted T5/VAE 权重。所有大模型导出先写云机本地 NVMe，正式链路测量窗口内不复制到共享盘。

独立导出脚本为 `deploy_cloudedge/performance_model.py`。它导出标准 LeapBot-CE delta（供云服务加载）和 CPU 端侧 artifact（供 Jetson 直接加载），每份附 JSON provenance。端侧 artifact 保留 action expert、SigLIP、projector、本体编码器、processor/normalizer 以及 codec per-layer scales；移除 Wan、T5、VAE、codec 和训练 registry 的云侧引用，避免 32 GB Jetson 先构建完整模型再裁剪的内存峰值。artifact 是由本项目本地生成的 Python/Torch 对象，仅对该可信来源使用 `torch.load(..., weights_only=False)`。

```bash
python deploy_cloudedge/performance_model.py \
  --base /home/aihub/daojie/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  --io-overrides /tmp/leapbotce_20261029/libero_io.pt \
  --out-prepared-base /tmp/leapbotce_20261029/artifacts/performance_base.pt \
  --siglip /home/aihub/daojie/FastWAM/checkpoints/siglip-base/model.safetensors \
  --stats /data/Database/fastwam_data/checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  --model-base-path /home/aihub/daojie/FastWAM/checkpoints \
  --out-delta /tmp/leapbotce_20261029/artifacts/performance_a6shape_delta_seed0.pt \
  --out-edge-artifact /tmp/leapbotce_20261029/artifacts/performance_a6shape_edge_seed0.pt \
  --out-cloud-artifact /tmp/leapbotce_20261029/artifacts/performance_full_seed0.pt \
  --initialization-note robotwin_experts_libero_io_untrained_codec_projector_zero_LoRA \
  --device cpu --threads 8 --seed 0
```

## 测试床与环境

| 项目 | 云端 aihub | 边端 Jetson |
|---|---|---|
| 架构 | x86_64 | aarch64 |
| GPU | RTX 4090 D，48 GB（每卡驱动报告 49,140 MiB） | Jetson AGX Orin，32 GB 统一内存，compute capability 8.7 |
| 系统/驱动 | 原始记录见 `hardware_aihub.json`；初始驱动 580.178.04 | L4T R36.4.7，CUDA 12.6，MODE_50W，CPU governor schedutil |
| 环境维护 | 挂载盘 `envs/leapbotce/x86_64` 覆盖层，复用已有本地解释器 | 挂载盘 `envs/leapbotce/aarch64` 覆盖层，复用 JetPack 配套本地解释器 |
| 推理部署 | Wan/T5/VAE/codec 云服务 | 仅端侧 artifact，避免完整模型峰值 |
| 链路 | 校园 Wi-Fi，明确 IPv4 地址连接 | 校园 Wi-Fi；客户端也在 Jetson 上运行 |

初始 GPU 快照存在已有显存驻留进程，计算时延需结合正式运行期间 GPU 利用率快照解释。环境和源代码维护在共享挂载盘；为减少 CIFS 导入延迟，可将同一代码与覆盖层复制到 `/tmp` 执行，副本路径和版本记录在实验 manifest。架构相关 torch 运行时分别来自现有 x86 与 JetPack 环境，不能跨机器直接混用。详见 `jetson_preflight_20261029.md`。

## 计量口径

- 云端 CUDA events：VAE、视频 prefill、codec；端侧 CUDA events：SigLIP/前缀与 10 步动作去噪。event 同步结束后读取，单位 ms。
- 客户端 wall-clock：完整请求/响应、通信与端到端流水线；同一客户端使用 monotonic clock，不依赖跨主机时钟差。CUDA event 仅覆盖 GPU 阶段，不与包含同一阶段的 HTTP 墙钟读数重复相加。
- 当前客户端采用 cloud→client→edge 中转协议，包含云响应和端侧 `/payload` 两个 HTTP 载荷腿；客户端在 Jetson 时第二腿为端机本地服务。上传/下载墙钟不等同于纯单向网络传播，旧 `z_transfer_ms` 与跨主机计数器相减指标不再填值。
- 稳态统计排除配置明确声明的 warm-up，保留所有正式样本，报告 mean/p50/p95、样本数、异常和 timeout。
- 完成吞吐率为测量窗口内已完成 replan 数除以窗口时长。串行客户端吞吐与并发服务吞吐分别标注；`1000/mean(stage_ms)` 是该阶段的近似服务能力，不能替代完成吞吐。
- `serial` 是单条闭环流；`pipeline` 仅将下一次云请求与当前端侧调用重叠，云端最多预取一次，端侧串行。它测量这种调度下的完成吞吐，不代表多客户端饱和容量。32 条生成动作或声明执行 10 条动作的折算速率均不称为机械臂实测控制频率。
- 模型输入和网络载荷随 raw/codec 模式改变时，分别记录 bytes；传输时延包含实际协议开销，不能用标称 1 Gbps 算术下界代替。
- 网络 `iperf3` 各方向 30 s × 3 轮；正式前后组与模型运行同一时段。环境安装或共享盘大文件读写重叠的网络记录明确标注，不作为干净链路基准。

原始硬件、配置、运行日志与后续测量 JSON 统一存于 `deploy_cloudedge/results/testbed_20261029/`。最终结果报告应从这些文件生成；本文件记录测量前已确定的方法，不预填任何性能数字。

## 测试结束后的模型归档

本轮delta、仅端侧artifact及完整policy已持久保存到aihub本地NVMe的`/home/aihub/daojie/LeapBotCE_artifacts/testbed_20261029/`；旧`/tmp/leapbotce_20261029/artifacts`为兼容symlink。共享测量目录的`artifact_archive_manifest.json`保存大小、SHA256及完整初始化来源。可重新生成的12GB prepared base已删除以回收磁盘，原始MoT基座、LIBERO I/O覆盖和完整policy仍在。服务及临时探针已停止。
