# LeapBot-CE 真机部署：云/端两阶段 FastAPI 服务

> 2026-10-29 测量口径更新：跨主机不能相减 `perf_counter`。
> 新版 `bench_client.py` 使用客户端同一时钟记录完整 HTTP 事务与实测完成吞吐，
> `z_transfer_ms` 等旧指标已弃用。旧 H800 JSON 保留作历史记录。
> 将客户端运行在 Jetson，cloud URL 指向 aihub，edge URL 使用 localhost，
> 使载荷仅经过一次物理云→端链路。部署依赖包括模型传递依赖；六个 HTTP/张量包并不足够。
> 本次使用已有 FastWAM 基座和同 a6 结构、未经训练的新模块测量性能，详见 docs 实测报告。

LeapBot-CE 的部署形态拆成两个独立服务，中间只过规划载荷 z：

```
端（机器人侧）                                  云
┌────────────────────────────────┐   LAN/HTTP   ┌──────────────────────────────┐
│ SigLIP fresh grounding          │              │ Wan2.2 5B video expert（冻结）│
│ 1.02B action expert             │  z 0.34 MB   │ planning-token codec          │
│ proprio encoder                 │ <──────────  │ T5 text encoder（每任务一次）  │
│ 控制循环（不阻塞在云上）          │   每次重规划  │ VAE + 一次 prefill            │
└────────────────────────────────┘              └──────────────────────────────┘
        ↑ 陈旧帧+本体（上联）                        ↑ 延迟观测（ age = 网络时延）
```

- **cloud_server.py**：收两相机原始帧 + 8 维本体状态 → VAE 编码 + Wan 一次
  prefill + KV codec → 返回 **z（56 tokens × 3072，bf16，0.34 MB）**。
  子步时延（VAE / prefill / codec）用 CUDA event 计时随帧头返回。
- **edge_server.py**：`/payload` 收到 z 后立刻用 checkpoint 自带的 per-layer
  scale 重建 30 层 K/V（零传输成本）；`/step` 用当前新鲜视图 + 当前本体去噪
  出 32×7 动作块。子步时延（SigLIP / 去噪）同样回传。
- **protocol.py**：`LBC1` 帧 = 4B magic + 4B JSON 头长 + JSON 头 + 原始张量
  字节（bf16 直传，无 base64 膨胀）。
- **bench_client.py**：在 Jetson 上发起云请求、本机安装 payload 与本机 step，
  用客户端同一时钟记录完整 HTTP 事务、序列化、端到端 mean/p50/p95，以及
  `成功完成数 / 正式测量窗口秒数`。`--raw` 对照完整 K/V，
  `--stage cloud|payload|edge|full` 分测阶段，`--schedule pipeline` 最多提前一个
  cloud 请求并保持 payload/step 串行；返回动作的 replan_id 与有限值均检查。
  `cloud_uplink_ms`、`edge_uplink_ms`、`z_transfer_ms` 已弃用为 null，不能继续用于跨机测量。
- **transport_probe_server.py / link_probe_client.py**：无模型的独立 HTTP 载荷实验，
  小请求下载指定字节 / 大请求小确认，单独量通信事务开销；使用 synthetic body，
  不能混称为真实模型 z 的端到端时延。
- **bw_sweep.py**：带宽 mini-study 扫描——对同一训练好的 codec 做
  **零样本载荷变体**（请求头 `pool_grid` 重定向自适应池化目标网格 +
  `payload_dtype=int8` 逐 token 对称量化，scale 内联帧头、端侧反量化），
  网格 × dtype 各 n 次 replan，输出载荷字节 / z 相对误差 / codec 与
  KV 重建开销（`results/bw_sweep.json`）。其旧 `z_transfer_ms` 口径已弃用，勿用于论文网络时延。retention（成功率）不在此脚本
  范围内，需完整评测协议。
- **boot 自检**：cloud_server 启动时先在整模型上跑"拆分路径 vs 参考路径"
  逐位对齐（`encode_cloud`+`infer_action_edge` vs 编解码+重建+分阶段去噪，
  max|diff| 阈值 1e-2），通过后才裁掉边侧权重 —— 保证线上协议与评测代码
  数学等价。

## 模型配方（与论文 a6 主线一致）

- hydra task = `libero_all_leapbotce_codec_lora`（LoRA r16 + KV codec，
  pool_grid [7,8] → 56 tokens，use_kv_decoder=false，2 视图 select 模式）。
- `mot_pretrained_path` = `libero_uncond_2cam224.pt`（训练所用冻结基座，
  delta ckpt 只存可训参数，缺此行 = video expert 回退 vanilla Wan）。
- checkpoint：`codec_a6_lora_fs_15ep_seed42_h200/checkpoints/weights/step_032544.pt`
  （论文 a6 主结果的 15ep 终点）。

## 调试卡（本 pod，H800 × 2，LAN）

```bash
bash run_h800_debug.sh              # 云 GPU 3 :8600，端 GPU 5 :8601，50 次重规划
# 结果: results/h800_lan.json + logs/cloud_3.log, logs/edge_5.log
```

- 两个服务把非本侧权重裁掉。历史 H800 读数不作为 4090D/Orin 的内存或
  时延结论；本轮报告使用实际进程监控，Jetson 的统一内存与独显显存分别解释。
- 任何跨主机请求时长都由客户端单一 `perf_counter` 计时。服务端 CUDA events
  仅作计算归因；`X-Recv-Done - X-Recv-Start` 只在同一服务内解释。
  不相减不同主机的 monotonic 时钟；NTP 同步也不能将其变为可比较的时间基准。
- **口径注意**：旧调试卡与集群其它负载共享 GPU。模型计算和 HTTP 墙钟均受
  当时服务/主机负载影响；旧传输字段使用的计量口径已弃用，不能作为本轮 LAN
  或 Wi-Fi 基准。

## 4090D / Jetson AGX Orin 测试床（真机，校园网 WiFi）

```bash
# 云机（4090D）
python cloud_server.py --port 8600 --ckpt <同上> --stats <同上>
# 端机（Jetson AGX Orin，随臂；JetPack + arm64 torch 轮子）
python edge_server.py  --port 8601 --ckpt <同上> --stats <同上>
# 客户端在 Jetson 上运行
python bench_client.py --cloud http://<云机IP>:8600 --edge http://<端机IP>:8601 \
    --out results/testbed.json
```

部署依赖包含 `fastapi uvicorn requests pillow numpy torch hydra-core omegaconf timm
transformers safetensors einops` 等模型传递依赖；使用环境中实际 import、GPU
算子及模型 self-test 进行预检。ARM/x86_64 的 CUDA/PyTorch 二进制不能混用。
本轮维护源在共享 `envs/leapbotce/{aarch64,x86_64}`，复用已有本地 Python/CUDA
运行时，详见 [Jetson 预检](../docs/jetson_preflight_20261029.md)。

#### 本轮实际启动（环境和权重已按本轮记录准备）

```bash
# aihub 的本地运行缓存；先按 performance_model.py 导出文件
CUDA_VISIBLE_DEVICES=0 python cloud_server.py --port 8600 \
  --cloud-artifact /tmp/leapbotce_20261029/artifacts/performance_full_seed0.pt
# Jetson 只加载端侧文件，避免完整云模型启动峰值
python edge_server.py --port 8601 \
  --edge-artifact /tmp/leapbotce_20261029/edge_artifact.pt
# 客户端务必放在 Jetson；使用已维护环境的 python wrapper
python bench_client.py --cloud http://10.7.140.45:8600 \
  --edge http://127.0.0.1:8601 --n 50 --warmup 5 \
  --primary primary_224.png --wrist wrist_224.png --out codec_serial.json
# 对照：加 --raw；重叠执行：加 --schedule pipeline
```

使用真实 a6 delta 的原服务启动方式继续支持。硬件路径可以通过重复
`--config-override` 指定：`model.redirect_common_files=true`（现有 converted
T5/VAE），`model.edge_vision.checkpoint_path=<siglip>`，以及
`model.skip_dit_load_from_pretrain=true`（仅在完整 MoT 基座必定加载时启用）。
`--cloud-artifact/--edge-artifact` 只接受本地生成、可信的 pickle 文件。
完整配方与来源见 [性能模型记录](../docs/performance_model_20261029.md)。

### 链路带宽实测规程（写论文用，给实施者）

链路 = 实验室校园网 WiFi（两台电脑同网段通信）。**论文报告实测有效带宽，
不引用标称协议速率**——WiFi 实测带宽波动大，标称值不可信。规程：

1. **iperf3 双向 × 多轮**：云机 `iperf3 -s`，端机
   `iperf3 -c <云机IP> -t 30`（端→云上联方向，观测帧方向）与
   `iperf3 -c <云机IP> -t 30 -R`（云→端，z 载荷方向）各测 3 轮，
   **演示/评测开始前与结束后各一组**，记录每轮 Mbits/s 与组内波动。
   可加 `-u -b 100M` 一轮记 UDP 丢包率——WiFi 丢包/重传直接打在 p95 腿上。
2. **HTTP 交叉验证**：用 `link_probe_client.py --operation download` 在 Jetson
   对 aihub 的独立 probe server 测 344064/36126720B 事务，并保存每个请求。
   `application_body_mbit_per_s` 是 HTTP 应用 goodput，与 iperf3 TCP 吞吐分开。
   模型客户端的 cloud_request 包含观测上行、计算和载荷下行，不能用其倒推
   纯下行带宽；旧 `z_transfer_ms` 也不能这样使用。
3. **入库**：iperf3 原始输出存 `results/testbed_iperf.txt`（或 json）、
   harness 读数在 `results/testbed.json`，随 `git add -f results/` 入库。
4. **时段一致性**：校园网白天/晚高峰差异大，带宽测量与真机演示、
   时延重测尽量同一时段完成；论文如报告带宽，注明测量时刻与轮数。
5. **论文写法**：链路注明 "campus Wi-Fi, measured X–Y Mbit/s
   (iperf3, n 轮, 测试前后)"。HTTP事务时延、TCP有效吞吐、模型端到端时延分别
   报告，使用真实请求记录，不能用载荷字节除以标称带宽替代。

## 评测调用约定（真机控制循环）

```python
# 任务开始（一次）：context 走 T5，按任务缓存
ctx = POST cloud/encode_prompt {prompt}
POST edge/session {context}

# 每次重规划的串行调用顺序（本轮不驱动机械臂）：
z  = POST cloud/encode_cloud   # 上一帧 + 本体（真实部署中 age = 网络时延）
POST edge/payload              # z 到达即重建 K/V
a  = POST edge/step            # 当前视图 + 当前本体 → 32×7 动作块，执行 10 步
```

`bench_client.py --schedule pipeline` 在边侧调用期间提前一个云请求，payload与
step依次执行并检查cache id。吞吐与每次观测dispatch到动作返回的时延分别统计；
提前请求带来的排队会增加后者。本脚本不实现机械臂伺服循环。

`edge/payload` 与 `edge/step` 拆开是为把 z 传输腿从端计算里干净剥离（测量
仪器语义）；生产单往返可用合并端点 `edge/infer_action_edge`（z+视图一次
POST，少一次 RTT）。

## 结果与复现

- [本轮时延/吞吐报告](../docs/testbed_latency_throughput_20261029.md)
- [网络原始记录与排除说明](../docs/testbed_network_20261029.md)
- 原始 JSON、环境/硬件快照、命令和来源：`results/testbed_20261029/`。
- 聚合：`python analyze_testbed.py results/testbed_20261029`，按三轮技术重复报告
  吞吐范围和请求的描述性分位数；保留失败和被排除的 pilot 记录。
- 旧 `results/h800_lan*.json` 是 H800 localhost 历史结果，不能替代 Orin/4090D 实测。
- `edge/payload` 与 `edge/step` 分开用于分解 HTTP/安装/计算成本，不应将 payload
  请求减空 POST RTT 的残差称为纯网络传输。
