# 2026-10-29 校园 Wi-Fi 真机链路实测

正式前测使用 Jetson AGX Orin（10.7.136.8）与 aihub（10.7.140.45）之间的真实校园 Wi-Fi，iperf3 单 TCP 流，每方向 3 轮，每轮 30 s，接收端吞吐量口径。保留的有效试验发生于北京时间 18:39:33–18:44:05，轮次文件编号不表示严格时间顺序；每轮真实开始时间写在 JSON 及汇总中。

| 方向 | 3 轮接收吞吐量（Mbit/s） | 均值（Mbit/s） | 样本标准差（Mbit/s） |
|---|---|---:|---:|
| Jetson → aihub，上行观测 | 13.486 / 12.684 / 7.932 | 11.367 | 3.002 |
| aihub → Jetson，下行计划载荷 | 85.501 / 88.145 / 87.169 | 86.938 | 1.337 |

链路有明显方向差异；应使用对应方向带宽解释实际协议腿。iperf3 是持续大流传输基准，HTTP 请求的真实载荷吞吐还包含请求头、串行化、RTT 和协议开销，应以独立 HTTP 实测补充。这里的接收端 Mbps 不可直接当作模型重规划频率或动作吞吐率。

原始数据与摘要在 `deploy_cloudedge/results/testbed_20261029/iperf_before_{forward,reverse}_{1,2,3}.json`、`iperf_before_summary.json`。第一次安装期间的 6 轮以 `iperf_setup_overlap_*.json` 独立保留；最初正式组的上行第 1 轮与旧 SMB 权重读取重叠，第 2 轮退出时间不确定，两者均以 `iperf_setup_overlap2_forward_*.json` 排除并补测。最初下行 3 次瞬时控制连接错误未得到任何有效吞吐值，以 `iperf_control_failed_reverse_*.json` 保留；随后使用独立 subprocess、轮间 1 s 间隔重跑成功。有效样本选择依据测量有效性与外部流量记录，未按吞吐高低筛选。

端侧为 50 W 模式；本轮未更改功耗设置。ARM 依赖安装、源码缓存传输与正式组错开；云端新模型 CPU 导出读取本地 NVMe，HTTP 测试等待前测完成后才开始。`iperf_before_manifest.json` 记录协调与排除依据。

正式后测在 15 组模型实验及监控停止后进行，仍为每方向 3×30 s，6 轮均首试成功，零失败；测量与权重归档错开。后测端→云均值 **13.441 Mbit/s**、范围 **12.265–14.606**；云→端均值 **93.653 Mbit/s**、范围 **91.684–95.971**。前后合计每方向 n=6，端→云范围 **7.932–14.606 Mbit/s**、均值 **12.404**；云→端范围 **85.501–95.971 Mbit/s**、均值 **90.296**。后测有一定改善，论文应报告前后真实范围与样本数，不能用单一常数代表所有请求时刻。

原始后测为 `iperf_after_{forward,reverse}_{1,2,3}.json`；分组和合并统计见 `iperf_after_summary.json`、`iperf_before_after_summary.json`。本次 :18609 iperf3 服务完成后按保存的 PID 3440582 和命令内容核验后发送 SIGTERM，清理记录在 `iperf_server_cleanup.json`。
