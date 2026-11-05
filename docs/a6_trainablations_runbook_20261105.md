# 出差期间无人值守手册（2026-11-05 建）

> 场景：用户 11-05 晚起出差约两天，不一定能随时反馈或提交 AIFlow 单。
> 本文件只讲"要提交什么、跑完会落什么数、我（Claude）盯什么"。

## 1. 只剩一张单要提交

| 项 | 值 |
|---|---|
| 任务名（平台填） | `leapbot_ce_eval_libero_a6_trainablations` |
| 工作目录 userPath | `/dahuafs/userdata/2653853/code/FastWAM`（**必须是 FastWAM 树**，上轮事故就错在这里） |
| 配置 | 平台自动分发到 `AIFlow_config_eval_libero_a6_trainablations.py`（已自测 43/43 PASS） |
| 卡型 / 卡数 | H100 / **4**（PACK=1，每 worker ~25 GB） |
| 预计时长 | 等 edgeonly 到 11-06 ≈08:00、singleage 到 11-06 ≈13:25，之后两次四窗评测各 12–13 h → **整体 ~11-07 03:00 收工** |
| 提交时机 | **现在就可以提**。脚本自带等待逻辑（`scripts/eval_a6_trainablations_wait.sh`，最长等 36 h），ckpt 没落盘就空等，落盘即开跑 |

它的作用是替代两张单：`AIFlow_config_eval_libero_a6_edgeonly.py` 与
`..._singleage.py` 在 ckpt 不存在时会在 import 期直接报错，出差期间没人能给
它们踩点提交，所以合并成一个"等着跑"的作业。

**代价**：等待期间那 4 张卡空转（最多 ~18 h）。若您能保证 11-06 08:00 和
13:25 左右各提交一次普通单，就不需要这张合并单 —— 二者结果完全等价（TAG、
delays、trials、PACK、override 逐项相同）。

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

## 4. 出差期间的已知待决项（不阻塞，回来再定）

1. native D60/D80 探针（`robotwin_v2_sweep_b6_ext`）是否重跑 —— 现判不可用，§8.4 因此不更新。
2. nocloudcache 22865923 已自行结束（11-05 18:44 已不在运行列表），D20 4/40、D40 0/40 冻结不补。
3. 论文仓 9 笔 commit 待 push。
