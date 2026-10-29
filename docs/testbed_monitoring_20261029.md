# 2026-10-29 真机性能实验资源监控

监控在两主机本地以 1 Hz 只读采样，正式矩阵结束后停止并归档。按 15 组正式操作的 `started_utc` 加客户端 `completed_perf-started_perf` 映射实际计量窗，排除初始化和 warm-up；取窗内样本，不把间歇低利用率删除。两机 NTP 已同步，边界归属仍只有约秒级精度。原始记录在 `results/testbed_20261029/resource_monitor_{jetson,aihub}/`；统计输出 `monitor_summary.json`；复算脚本 `analyze_resource_monitors.py`。

Jetson 共 1171 个正式窗采样：统一内存占用 8315–8868 MB（tegrastats 标注 MB，系统总量 30697 MB），SWAP 始终 0/15348 MB。CPU 温度 49.156–52.000 °C，`tj` 传感器 49.156–52.093 °C。GR3D 利用率均值 40.27%，p95 59%，范围 0–99%；其中 codec edge-only 各轮均值约 55.3–56.0%，codec pipeline 约 54.8–55.7%，raw 串行约 19.4–20.5%（其他时间主要等待网络）。CPU 当前活动核心最高频率在样本中为 729–1497 MHz。部署保持预检时的 `MODE_50W`，未改变电源模式；监控没有逐秒重读 NVPModel。

正式 Jetson 监控输出仅含 `GR3D_FREQ 百分比`，没有 GPU MHz；同时没有 `gpu@` 温度 token，因此不能以其他传感器代替 GPU 温度。没有采集 throttle reason 或限频计数器；上述温度、频率与利用率记录不能单独证明“未发生降频”。记录表明本轮没有交换分区占用，没有观测到统一内存紧张。

aihub GPU0 正式窗有 1178 个采样：总已用显存 38202–38297 MiB，GPU 利用率均值 1.63%、p95 15%、范围 0–34%，温度 38–48 °C，功耗 18.34–96.90 W（均值 64.05 W）。GPU0 保留的既有 ComfyUI 进程占 14684 MiB，本次 FastWAM 云服务进程占 23000 MiB；这些来自每 10 s 进程显存快照，不能把 GPU0 全部显存归为本模型。GPU1 利用率始终 0%，已用显存固定 17254 MiB，主要为同一既有进程；温度 30–31 °C，功耗均值 21.11 W。云端采样的低平均利用率包含等待端侧与网络的时间，不等同云端算子单次效率。

采样无解析或查询错误，两个监控 wrapper 均按本次 PID 安全停止。由于采样间隔为 1 s，短时云推理可能落在采样之间；CUDA-event 模型段时延仍以 benchmark 原始记录为准。Jetson 的 VDD_GPU_SOC、VDD_CPU_CV、VIN_SYS_5V0 为各硬件电源轨读数，不能任意相加为墙上功耗。
