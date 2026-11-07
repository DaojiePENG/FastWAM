# 出差期间无人值守手册（2026-11-05 建）

> 场景：用户 11-05 晚起出差约两天，不一定能随时反馈或提交 AIFlow 单。
> 本文件只讲"要提交什么、跑完会落什么数、我（Claude）盯什么"。

## 0. ⛔ 作废：等待式提单在平台上必死（2026-11-06 01:24 实测）

11-05 19:46 提交的两张等待单（22984907 / 22984995）在 **11-06 01:23:54 / 01:23:58
被平台杀掉**，容器日志原文：

```
任务因gpu 利用率空闲时间大于4501 秒退出
任务gpu 利用率不足
exit code: 250
```

即平台有 **GPU 空闲看门狗**（阈值 ≈4501 s ≈75 min 累计空闲）。`eval_a6_trainablations_wait.sh`
的轮询循环不碰 GPU，从启动起就在给空闲计时器充值，5.6 h 后到达上限被杀。
**结论：任何"占卡等 ckpt"的提法在本平台不可行**，下面 §1 的双单方案作废（文件保留，
不再使用）。替代口径见 §5：ckpt 落盘后再提普通单。

## 1. ~~要提交两张单（11-05 用户拍板：并行双单）~~（已作废，见 §0）

| | 单 A | 单 B |
|---|---|---|
| 任务名（平台填） | `leapbot_ce_eval_libero_a6_trainablations_edgeonly` | `leapbot_ce_eval_libero_a6_trainablations_singleage` |
| 分发到 | `AIFlow_config_eval_libero_a6_trainablations_edgeonly.py` | `..._trainablations_singleage.py` |
| 工作目录 userPath | `/dahuafs/userdata/2653853/code/FastWAM`（**必须 FastWAM 树**） | 同左 |
| 卡型 / 卡数 | H100 / **4**（PACK=1，每 worker ~25 GB） | H100 / **4** |
| 等待的 ckpt | edgeonly `step_032544.pt`（训练 11-06 ≈08:00） | singleage `step_032544.pt`（11-06 ≈13:25） |
| 评测耗时 | 12–13 h → **11-06 ≈20:30** | 12–13 h → **11-07 ≈02:00** |
| 提交时机 | **现在就可以提**（最长等 36 h，落盘即开跑） | 同左 |

为什么不提单张 8 卡：PACK=1 时 8 worker × 25–30 GB ≈ 200–240 GB 峰值会在部分
H100 节点顶到容器内存墙、静默 SIGABRT（10-19 事故）。两张 4 卡并行同样是 8 卡，
既避开该坑，又比"单张 4 卡串行合并单"（11-07 ≈09:00 收工）早约 6.5 h。

串行合并单 `leapbot_ce_eval_libero_a6_trainablations` 仍保留可用（只提一张时用），
只是收工更晚。

**代价**：等待期间两张卡组各自空转（B 单最多 ~18 h）。若哪天能按点提普通单
（`..._a6_edgeonly.py` / `..._a6_singleage.py`），结果完全等价、零浪费。

## 2. 跑完会落什么数

| 结果根（TAG） | 行名（`tab:ablations` 训练块） | 交付 |
|---|---|---|
| `baseline-evaluations/leapbotce/a6_edgeonly_step32544/delay_{0,10,20,40}/` | `w/o cloud cache (training)` | 四窗成功率 + retention |
| `baseline-evaluations/leapbotce/a6_singleage_step32544/delay_{0,10,20,40}/` | `single-branch random-age` | 四窗成功率 + retention |

判定完成的口径：每个 delay 下 `find delay_N -name 'gpu*_task*_results.json' | wc -l` = 40。

## 3. 我（Claude）在出差期间会做什么

| 频率 | 动作 |
|---|---|
| 每小时 | 跑原六项盯盘：nocloudcache（已冻结）、两路训练、三 seed、三条 RoboTwin sweep、异常即停判定 |
| 实时 | **starwam rs8 整行四窗 50/50 齐的那一刻**，替换 `tab:robotwin` 的 † 行（D0 85.5 / D10 / D20 / D40）并把脚注从 "Ten-task subset … pending" 改为全量口径，编译校验后 commit（不 push） |
| 实时 | 两个评测单每落满一窗就聚合四套件成功率，写 docs 与 ops log；四窗全齐后回填 `tab:ablations` 训练块 |
| 约束 | 不杀任何进程；未获批不删资产；论文仓不 push；论文措辞不改，只填数字与口径 |

## 4. 替代口径（2026-11-06 起）：ckpt 落盘后提普通单

| | edge-only 行 | single-branch random-age 行 |
|---|---|---|
| ckpt 落盘 | **已完成**：`step_032544.pt` 2.05 GB，2026-11-06 08:07:03（训练 08:07:15 正常退出） | 预计 2026-11-06 ≈13:55（08:37 时 28760/32544，eta 05:18） |
| 平台任务名 | `leapbot_ce_eval_libero_a6_edgeonly` | `leapbot_ce_eval_libero_a6_singleage` |
| 分发到 | `AIFlow_config_eval_libero_a6_edgeonly.py` | `AIFlow_config_eval_libero_a6_singleage.py` |
| 卡型 / 卡数 | H100 或 H200 / **4**（PACK=1，每 worker ~25 GB） | 同左 |
| 额外 override | 配置内已烤入 `model.cloudedge.use_cloud_cache=false` | 无 |
| 实测时长 | 15–16 h（同族先例：nocloudcache 3.9 h/delay × 4） | 15–20 h（推理含云端 encode，应不短于 edge-only） |
| 结果根（TAG） | `.../leapbotce/a6_edgeonly_step32544/` | `.../leapbotce/a6_singleage_step32544/` |

`check_aiflow_dispatch.py` 已验：这两个任务名分别路由到上面的普通单配置（45/45 PASS），
且两者都带 ckpt 存在性校验，落盘前提会被拒。

## 5. 出差期间的已知待决项（不阻塞，回来再定）

1. native D60/D80 探针（`robotwin_v2_sweep_b6_ext`）是否重跑 —— 现判不可用，§8.4 因此不更新。
2. nocloudcache 22865923 已自行结束（11-05 18:44 已不在运行列表），D20 4/40、D40 0/40 冻结不补。
3. 论文仓 9 笔 commit 待 push。
