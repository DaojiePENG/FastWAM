# LeapBot-CE 真机部署：云/端两阶段 FastAPI 服务

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
- **bench_client.py**：扮演机器人控制循环，逐次重规划记录
  云上行 / 云计算（VAE+prefill+codec）/ **z 传输（0.34 MB，去 RTT 基线）** /
  KV 重建 / 端上行 / 端计算（SigLIP+去噪）/ 全程往返，聚合 mean/p50/p95。
  `--raw` 切到无 codec 反事实接口：云侧 `/encode_cloud_raw` 返回完整 30 层
  K/V cache（36.1 MB bf16），端侧 `/payload_raw` 收缓存直接装配（无 scale
  重建）；模型计算段与 codec 路径相同，用于单独量化传输腿差异。
- **bw_sweep.py**：带宽 mini-study 扫描——对同一训练好的 codec 做
  **零样本载荷变体**（请求头 `pool_grid` 重定向自适应池化目标网格 +
  `payload_dtype=int8` 逐 token 对称量化，scale 内联帧头、端侧反量化），
  网格 × dtype 各 n 次 replan，输出载荷字节 / z 相对误差 / codec 与
  KV 重建开销（`results/bw_sweep.json`）。retention（成功率）不在此脚本
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

- 两个服务把非本侧权重裁掉：云进程 ~21 GB（Wan+T5+VAE），端进程 ~4 GB
  （action expert+SigLIP+proprio），这本身就是 4090D(48G)/5080(16G) 的
  显存画像预演。
- 同机时网络腿共享 CLOCK_MONOTONIC，z 传输 = 客户端墙钟 − 空 POST RTT 基线；
  跨机部署同一客户端直接量真实 LAN（墙钟戳同时写在帧头，跨机用 NTP 对齐）。
- **口径注意**：调试卡与集群其它负载共享 GPU（util 常态 100%），模型段时延
  是"并发挤占下"的上界读数；z 传输腿（网络）不受影响，可直接当 LAN 参考。

## 4090D / 5080 测试床（后续真机）

```bash
# 云机（4090D）
python cloud_server.py --port 8600 --ckpt <同上> --stats <同上>
# 端机（5080）
python edge_server.py  --port 8601 --ckpt <同上> --stats <同上>
# 任一机（或机器人工控机）
python bench_client.py --cloud http://<云机IP>:8600 --edge http://<端机IP>:8601 \
    --out results/testbed.json
```

依赖只有 `fastapi uvicorn requests pillow numpy torch`（eval env 已装齐）；
不依赖 LIBERO/RoboTwin 仿真器，端机无需仿真栈。

## 评测调用约定（真机控制循环）

```python
# 任务开始（一次）：context 走 T5，按任务缓存
ctx = POST cloud/encode_prompt {prompt}
POST edge/session {context}

# 每次重规划（端侧控制循环，永不阻塞在云上）：
z  = POST cloud/encode_cloud   # 上一帧 + 本体（真实部署中 age = 网络时延）
POST edge/payload              # z 到达即重建 K/V（异步、latest-wins）
a  = POST edge/step            # 当前视图 + 当前本体 → 32×7 动作块，执行 10 步
```

`edge/payload` 与 `edge/step` 拆开是为把 z 传输腿从端计算里干净剥离（测量
仪器语义）；生产单往返可用合并端点 `edge/infer_action_edge`（z+视图一次
POST，少一次 RTT）。

## 已知口径/后续

- Table 7（论文）当前填的是本 pod H800 × 2 + localhost（LAN 语义）读数
  （含 `--raw` 无 codec 对照行）；4090D/Orin 真机部署后用同一脚本重测
  覆盖即可，z 传输腿换真实 LAN IP。
- SigLIP/prefill 在并发 GPU 上的读数偏大；若要干净上界可在空闲卡复测。
- 动作后处理（gripper 二值化/反演）属 LIBERO 评测语义，不在这层服务里；
  真机控制循环按需在端侧加。
