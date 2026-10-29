# Jetson 真机部署预检（2026-10-29）

当前端侧硬件是 **Jetson AGX Orin Developer Kit 32 GB**（Linux 报告 32,187,990,016 B 物理内存），并非部署说明中的 64 GB 假定配置。SSH 使用 `jetson`（10.7.136.8）正常；系统为 aarch64、L4T R36.4.7、CUDA 12.6，功耗模式 `MODE_50W`，CPU governor 为 `schedutil`。本轮未改变功耗或系统配置。

已有本地 `robot` 环境提供 Python 3.10.19、torch 2.8.0、torchvision 0.23.0；GPU 名称 Orin、compute capability 8.7，bf16 矩阵乘已在真实 GPU 上通过。所有新增包和配置维护于挂载盘 `/data_pdj/daojie/envs/leapbotce/aarch64/`，采用本地既有解释器加共享 `site-packages` 覆盖层。启动方法：

```bash
bash /data_pdj/daojie/envs/leapbotce/aarch64/python.sh \
  /data_pdj/daojie/codes/LeapBot_ws/FastWAM/deploy_cloudedge/edge_server.py --help
```

`server_common`、`edge_server` 和 torch/torchvision/numpy/Pillow/FastAPI/uvicorn/requests/Hydra/OmegaConf/einops/rich/timm/transformers 全部导入通过；`pip check` 返回 `No broken requirements found.`。原环境 OpenCV 4.13 的元数据要求 NumPy 2，覆盖层固定 OpenCV 4.11.0.86 与已有 NumPy 1.26.1 匹配。版本锁和启动说明位于环境目录 `requirements-overlay-installed.txt` 和 `README.md`。此环境依赖 Jetson 本地已有的 JetPack torch 运行时；不能直接在 x86_64 主机运行 ARM 覆盖层。

共享盘 CIFS 映射文件权限为 0644，Jetson 侧不支持创建 POSIX 符号链接。因此不使用共享盘完整 conda clone；通过 `bash` 调用 wrapper。端侧首次完整模型装配可能超过统一内存预算，应先导出仅含端侧模块的模型产物，再启动推理。

网络使用校园 Wi-Fi（HKUSTGZ，5 GHz，观测信号 −49 dBm）；这里的标称 Wi-Fi bitrate 仅用于硬件记录，不作为实测有效带宽。到 aihub 的可用地址为 10.7.140.45；10.7.0.131 初次 ICMP 3/3 无响应。Jetson 无法通过系统 DNS 解析 `aihub`，服务 URL 使用明确 IP。初始 RTT 含 1.8 s 尾值，正式测量需保留 p95 和每次原始记录。NTP 已同步。

硬件、依赖与 GPU 检查原始记录在 `deploy_cloudedge/results/testbed_20261029/hardware_jetson.json` 和 `dependencies_jetson.json`。初始连接测试的 iperf3 数据以 `iperf_setup_overlap_*.json` 保存，因为与环境准备重叠，不用于正式前后组对比。

正式计时执行使用 Jetson 本地 `/tmp/leapbotce_20261029` 临时缓存（源代码与 ARM 包均从共享盘打包传入），避免逐文件 CIFS 导入流量。共享盘仍维护包版本锁和启动脚本；`start_edge.sh` 调用缓存代码，`python_cache.sh` 可运行缓存客户端。缓存环境 `server_common`/`edge_server` 和主要依赖导入已通过，用时 11.44 s；`runtime_cache_jetson.json` 记录实际模块路径。完整有效依赖冻结版为共享环境目录 `requirements-effective-jetson.txt`（121 项），覆盖层锁为 `requirements-overlay-installed.txt`（45 项）。模型产物也应先复制至 Jetson `/tmp` 再启动。
