# RoboTwin 2.0 全量 50 任务基准 —— LeapBot-CE vs AHA-WAM vs StarWAM

> 状态文档(实验设计 + 推进结果,新日志条目在最上)。创建:2026-10-26。
> 背景:10-26 用户指示 —— 对 AHA-WAM (robotwin)、StarWAM (mot, official)、
> LeapBot-CE (ours, B6 codec_lora) 三方法做 RoboTwin 2.0 **全量 50 任务**
> 基准对比;v2 暂时只用这三个方法。

## 1. 目标与范围

| 维度 | 取值 |
|---|---|
| 方法(3) | LeapBot-CE(ours,B6 codec_lora step_158206)/ AHA-WAM(官方 robotwin ckpt)/ StarWAM(官方 MoT ckpt) |
| 任务(50) | RoboTwin 2.0 全量 `envs/*` 任务(§3 清单;两棵 RoboTwin 树清单逐名一致已核) |
| 时延窗(4) | **D{0,10,20,40}**(10-15 长期准则,不跑 D5) |
| 每格 | 50 episodes(demo_clean;与既有 10 任务 sweep 同协议) |
| 总格数 | 3 方法 × 50 任务 × 4 窗 = **600 格** |

## 2. 统一评测协议

- sim 配置:`task_config/demo_clean.yml`(embodiment aloha-agilex,
  eval_instruction=seen,50 eps);三方法同配置。
- 时延协议:各方法 adapter 内实现,同语义(每 replan 采样 delay ~
  U{0..D},frame-delay 注入);ours = eval 脚本 delay 窗,AHA =
  xpolicy sweep,StarWAM = STARWAM_MAX_DELAY_STEPS/SEED=1234。
- 结果判读:每格 `<root>/delay_<D>/<task>/_result_clean.txt`(或各方法
  等价 result 文件)尾行成功率;成功判据与既有 sweep 相同。
- **结果根三方法独立**(防 .done 撞车,10-24 教训):
  - ours: `/mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce/robotwin_v2_sweep_b6/`
  - AHA: `/mnt/c2-worldmodel/2653853/baseline-evaluations/ahawam/robotwin_v2_sweep/`
  - StarWAM: `/mnt/c2-worldmodel/2653853/baseline-evaluations/starwam/robotwin_v2_sweep/`
- **clean vs random(10-26 用户确认)**:官方 RoboTwin 2.0 提供两套 task_config
  —— `demo_clean.yml`(instruction=seen、干净背景、无光照/高度随机)与
  `demo_randomized.yml`(instruction=unseen、随机背景+杂乱+光照,
  clean_background_rate=0.02)。**v2 基准用 demo_clean**(与既有全部基线行
  —— AHA 10-05 sweep、X-WAM、Motus、B 线 —— 同协议,跨表可比);
  demo_randomized 暂不跑,若加做每方法格数翻倍(200→400),另行排期。
- 任务清单实锤(10-26 二次核验):`code/RoboTwin/envs/`、
  `FastWAM/third_party/RoboTwin/envs/` 两树各 50 个任务文件,
  与 §3 冻结清单逐名 diff 零差异(即官方 benchmark 任务全集)。

## 3. 50 任务清单(envs/ 注册表,两树一致)

前 10 个 = 既有 dtrain20 评测子集(有先例数据),后 40 个 = v2 新增。

```
adjust_bottle        beat_block_hammer    blocks_ranking_rgb   blocks_ranking_size
click_alarmclock     click_bell           dump_bin_bigbin      grab_roller
handover_block       handover_mic         hanging_mug          lift_pot
move_can_pot         move_pillbottle_pad  move_playingcard_away move_stapler_pad
open_laptop          open_microwave       pick_diverse_bottles pick_dual_bottles
place_a2b_left       place_a2b_right      place_bread_basket   place_bread_skillet
place_burger_fries   place_can_basket     place_cans_plasticbox place_container_plate
place_dual_shoes     place_empty_cup      place_fan            place_mouse_pad
place_object_basket  place_object_scale   place_object_stand   place_phone_stand
place_shoe           press_stapler        put_bottles_dustbin  put_object_cabinet
rotate_qrcode        scan_object          shake_bottle         shake_bottle_horizontally
stack_blocks_three   stack_blocks_two     stack_bowls_three    stack_bowls_two
stamp_seal           turn_switch
```

## 4. 方法资产卡

### 4.1 LeapBot-CE(ours)— 管线最熟,改动最小

| 项 | 值 |
|---|---|
| 代码树 | `/dahuafs/userdata/2653853/code/FastWAM`(eval_leapbotce_robotwin_sweep_2653853.sh + lane) |
| ckpt | `.../robotwin_all_codec_lora_dtrain20_seed42_gpu_n3/checkpoints/weights/step_158206.pt`(10-26 07:16 落盘,训练收工) |
| env | fastwam-robotwin-2653853 |
| 旋钮 | ROBOTWIN_EVAL_TASKS(50 名)/ LEAPBOTCE_SWEEP_ROOT(v2 根)/ ROBOTWIN_EVAL_DELAYS —— **三旋钮全部已存在** |
| 平台入口 | 任务名含 `b6` 变体 + v2 旋钮(AIFlow_config_eval_robotwin.py 需小改 + 四步自测) |
| 预估 | 200 格,H100/H200×8 PACK=2 → 16 lanes,avg ~2-2.5h/格 → **~1.5 天** |
| 未知风险 | 40 新任务的 instruction 模板与 text 编码(smoke 验证);`_eval_step_limit.yml` 各任务步上限 |

### 4.2 AHA-WAM — 全链路已验证(10-05 跑过全量 50 格)

| 项 | 值 |
|---|---|
| 代码树 | `/dahuafs/userdata/2653853/code/RoboTwin`(XPolicyLab;POLICY=AHA_WAM) |
| ckpt | `/mnt/c2-worldmodel/2653853/baselines/aha-wam/robotwin_ahawam.pt`(14.7 GB) |
| env | policy: pod_shared_envs/aha-wam-2653853;sim: robotwin-2653853 |
| 入口 | `bash run_sweep_cluster_xpolicy_2653853.sh AHA_WAM`;`ROBOTWIN_EVAL_TASKS` 可覆写任务表 ✓ |
| 结果根 | `${SWEEP_BASE}/ahawam/robotwin_sweep` → v2 需 SWEEP_BASE/根覆写(小改) |
| 先例 | 50 格(10 任务×5 窗)4×H800 8 泳道 ~27h → 200 格 16 泳道 **~2.3 天** |
| 未知风险 | 新任务 cell 时长长尾;~34GB/格 |

### 4.3 StarWAM(mot, official)

| 项 | 值 |
|---|---|
| 代码树 | `/dahuafs/userdata/2653853/code/StarWAM`(examples/robotwin,dual-server adapter)+ sim 在 code/RoboTwin(XPolicyLab) |
| ckpt | `/mnt/c2-worldmodel/2653853/baselines/starwam/starwam-robotwin/mot/starwam_wan225b_robotwin_mot.pt`(+ action_stats.json);recipe starwam_robotwin_mot_wan22_5b.yaml;qpos 14-d |
| env | infer: starwam-libero-2653853;eval: robotwin-2653853 |
| 入口 | `baseline_delay/starwam_robotwin_v2_chains_2653853.sh`(链式启动器:4 窗 × P 链 round-robin 分 50 任务)→ `starwam_robotwin_sweep_lane_2653853.sh <GPU> <D> <task...>`(任务走 argv ✓) |
| 结果根 | lane ROOT 已改 `STARWAM_SWEEP_ROOT` env 覆写(10-26),chains 缺省即 v2 独立根 ✓ |
| 平台入口 | **code/StarWAM/AIFlow_config.py 已建**(10-26,首个平台入口):任务名 `leapbot_ce_eval_robotwin_starwam_v2`(全量)/`…_smoke`(单格 place_shoe×D0)/`…_d2040`、`…_d010`(拆单);分发自测 `scripts_check/check_aiflow_dispatch_v2.py` **5/5 PASS** |
| GPU 编号 | 平台 CUDA_VISIBLE_DEVICES → 逻辑槽位(与 FastWAM cluster-sweep、AHA xpolicy sweep 同约定);lane 显卡用户态接线已改 nvidia_userland_wire.sh 节点自适应(平台驱动 ≠ 535.183.06 也能渲染) |
| smoke 先例 | 10-11 beat_block_hammer D0 MoT 5/5 通过 |
| text cache | recipe 占位 /path/to/... 为 pod 本地懒编码产物;10-26 起 lane 缺省 `STARWAM_TEXT_CACHE_DIR` 指共享盘(2016 条既有条目已并入),server 启动器加可选覆写(XPolicyLab 6547afd)→ 新任务首格懒编码一次后全节点复用 |
| 未知风险 | server+sim 同卡显存(平台卡型);cache 落盘权限由共享盘路径消除 |

## 5. 结果表

每格 = `<root>/delay_<D>/<task>/_result_clean.txt` 尾行成功率;macro = 50 任务平均。
**三方法全部收齐(ours 10-28 / AHA 10-29 / StarWAM 10-30)。**

| 任务 | ours D0/D10/D20/D40 | AHA D0/D10/D20/D40 | StarWAM D0/D10/D20/D40 |
|---|---|---|---|
| adjust_bottle | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 0.94 / 0.84 / 0.56 | 1.00 / 1.00 / 0.96 / 0.82 |
| beat_block_hammer | 0.98 / 0.96 / 0.92 / 0.90 | 0.98 / 0.46 / 0.22 / 0.10 | 1.00 / 0.92 / 0.86 / 0.62 |
| blocks_ranking_rgb | 1.00 / 1.00 / 1.00 / 0.98 | 1.00 / 0.90 / 0.50 / 0.02 | 1.00 / 1.00 / 1.00 / 0.86 |
| blocks_ranking_size | 0.92 / 0.84 / 0.72 / 0.80 | 1.00 / 0.86 / 0.44 / 0.00 | 0.92 / 0.94 / 0.98 / 0.74 |
| click_alarmclock | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 0.98 / 1.00 / 1.00 |
| click_bell | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 0.96 / 0.98 | 1.00 / 1.00 / 1.00 / 1.00 |
| dump_bin_bigbin | 0.96 / 0.92 / 0.94 / 0.98 | 0.98 / 0.72 / 0.58 / 0.32 | 0.90 / 0.92 / 0.94 / 0.62 |
| grab_roller | 1.00 / 1.00 / 0.98 / 0.98 | 1.00 / 1.00 / 0.98 / 0.94 | 1.00 / 1.00 / 1.00 / 1.00 |
| handover_block | 0.68 / 0.72 / 0.58 / 0.24 | 0.90 / 0.44 / 0.30 / 0.02 | 0.78 / 0.78 / 0.60 / 0.42 |
| handover_mic | 0.98 / 1.00 / 1.00 / 0.94 | 1.00 / 0.62 / 0.22 / 0.02 | 1.00 / 1.00 / 1.00 / 0.88 |
| hanging_mug | 0.68 / 0.56 / 0.54 / 0.46 | 0.86 / 0.26 / 0.16 / 0.00 | 0.46 / 0.34 / 0.26 / 0.12 |
| lift_pot | 0.96 / 0.92 / 1.00 / 0.92 | 1.00 / 0.98 / 0.96 / 0.90 | 1.00 / 1.00 / 1.00 / 0.98 |
| move_can_pot | 0.62 / 0.54 / 0.44 / 0.46 | 0.88 / 0.88 / 0.80 / 0.28 | 0.80 / 0.80 / 0.84 / 0.60 |
| move_pillbottle_pad | 0.98 / 0.96 / 0.96 / 0.96 | 1.00 / 0.92 / 0.58 / 0.16 | 0.96 / 1.00 / 0.98 / 0.84 |
| move_playingcard_away | 1.00 / 0.96 / 0.96 / 0.96 | 1.00 / 0.90 / 0.36 / 0.18 | 1.00 / 1.00 / 1.00 / 1.00 |
| move_stapler_pad | 0.72 / 0.60 / 0.58 / 0.50 | 0.90 / 0.46 / 0.18 / 0.02 | 0.88 / 0.92 / 0.86 / 0.58 |
| open_laptop | 0.88 / 0.80 / 0.84 / 0.76 | 0.98 / 0.88 / 0.62 / 0.66 | 0.96 / 0.94 / 0.98 / 0.94 |
| open_microwave | 0.58 / 0.40 / 0.46 / 0.40 | 0.70 / 0.58 / 0.42 / 0.32 | 0.12 / 0.16 / 0.12 / 0.20 |
| pick_diverse_bottles | 0.82 / 0.80 / 0.68 / 0.70 | 0.86 / 0.86 / 0.76 / 0.32 | 0.78 / 0.74 / 0.68 / 0.32 |
| pick_dual_bottles | 0.90 / 0.88 / 0.96 / 0.90 | 0.94 / 0.94 / 0.82 / 0.70 | 0.84 / 0.88 / 0.82 / 0.56 |
| place_a2b_left | 1.00 / 0.96 / 0.94 / 0.92 | 0.98 / 0.92 / 0.58 / 0.26 | 0.98 / 0.98 / 0.98 / 0.88 |
| place_a2b_right | 0.90 / 0.88 / 0.88 / 0.88 | 0.88 / 0.88 / 0.58 / 0.16 | 0.92 / 0.92 / 1.00 / 0.74 |
| place_bread_basket | 0.88 / 0.86 / 0.84 / 0.90 | 0.96 / 0.86 / 0.64 / 0.42 | 0.96 / 0.96 / 0.92 / 0.80 |
| place_bread_skillet | 0.74 / 0.88 / 0.84 / 0.74 | 0.92 / 0.80 / 0.54 / 0.30 | 0.96 / 0.92 / 0.94 / 0.88 |
| place_burger_fries | 0.94 / 0.94 / 0.92 / 0.94 | 0.90 / 1.00 / 0.52 / 0.02 | 0.96 / 0.96 / 0.92 / 0.48 |
| place_can_basket | 0.62 / 0.84 / 0.82 / 0.72 | 0.78 / 0.78 / 0.38 / 0.06 | 0.60 / 0.38 / 0.50 / 0.44 |
| place_cans_plasticbox | 0.76 / 0.84 / 0.84 / 0.86 | 1.00 / 0.80 / 0.44 / 0.06 | 0.98 / 0.98 / 0.96 / 0.74 |
| place_container_plate | 0.98 / 1.00 / 0.92 / 1.00 | 1.00 / 0.94 / 0.66 / 0.14 | 0.98 / 0.96 / 0.92 / 0.84 |
| place_dual_shoes | 0.82 / 0.76 / 0.74 / 0.72 | 0.94 / 0.76 / 0.50 / 0.16 | 0.88 / 0.88 / 0.84 / 0.68 |
| place_empty_cup | 1.00 / 1.00 / 1.00 / 0.98 | 1.00 / 0.98 / 0.66 / 0.16 | 1.00 / 1.00 / 0.98 / 0.88 |
| place_fan | 0.96 / 0.94 / 0.94 / 0.82 | 0.96 / 0.78 / 0.42 / 0.10 | 0.98 / 0.92 / 0.92 / 0.74 |
| place_mouse_pad | 0.74 / 0.64 / 0.66 / 0.62 | 0.92 / 0.66 / 0.30 / 0.10 | 0.80 / 0.68 / 0.70 / 0.60 |
| place_object_basket | 0.80 / 0.86 / 0.90 / 0.84 | 0.88 / 0.74 / 0.52 / 0.10 | 0.92 / 0.82 / 0.88 / 0.84 |
| place_object_scale | 0.92 / 0.84 / 0.86 / 0.82 | 0.94 / 0.78 / 0.38 / 0.06 | 0.96 / 0.96 / 0.90 / 0.86 |
| place_object_stand | 0.96 / 0.96 / 0.94 / 0.90 | 0.96 / 0.82 / 0.56 / 0.24 | 0.96 / 0.92 / 0.86 / 0.78 |
| place_phone_stand | 0.88 / 0.90 / 0.88 / 0.86 | 0.94 / 0.60 / 0.14 / 0.04 | 0.90 / 0.84 / 0.88 / 0.86 |
| place_shoe | 0.88 / 0.88 / 0.92 / 0.84 | 0.90 / 0.88 / 0.82 / 0.40 | 0.92 / 0.90 / 0.86 / 0.84 |
| press_stapler | 0.90 / 0.90 / 0.88 / 0.88 | 0.98 / 0.96 / 0.98 / 0.98 | 0.96 / 0.92 / 0.94 / 0.94 |
| put_bottles_dustbin | 0.94 / 0.94 / 0.84 / 0.44 | 0.90 / 0.32 / 0.04 / 0.00 | 0.88 / 0.94 / 0.80 / 0.36 |
| put_object_cabinet | 0.82 / 0.84 / 0.86 / 0.88 | 0.92 / 0.82 / 0.52 / 0.24 | 0.94 / 0.92 / 0.92 / 0.78 |
| rotate_qrcode | 0.90 / 0.94 / 0.94 / 0.76 | 0.98 / 0.74 / 0.24 / 0.02 | 0.86 / 0.96 / 0.82 / 0.52 |
| scan_object | 0.70 / 0.64 / 0.74 / 0.60 | 0.90 / 0.76 / 0.60 / 0.46 | 0.90 / 0.90 / 0.80 / 0.70 |
| shake_bottle | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 0.96 / 0.96 | 1.00 / 1.00 / 0.98 / 0.96 |
| shake_bottle_horizontally | 1.00 / 1.00 / 1.00 / 1.00 | 1.00 / 0.98 / 1.00 / 0.94 | 1.00 / 1.00 / 0.98 / 1.00 |
| stack_blocks_three | 0.96 / 0.86 / 0.92 / 0.86 | 0.74 / 0.42 / 0.10 / 0.02 | 0.90 / 0.72 / 0.66 / 0.38 |
| stack_blocks_two | 0.98 / 0.98 / 1.00 / 0.96 | 1.00 / 0.92 / 0.64 / 0.08 | 1.00 / 1.00 / 1.00 / 0.90 |
| stack_bowls_three | 0.66 / 0.80 / 0.84 / 0.74 | 0.82 / 0.64 / 0.22 / 0.00 | 0.84 / 0.82 / 0.68 / 0.30 |
| stack_bowls_two | 0.96 / 0.98 / 0.98 / 0.90 | 0.98 / 0.86 / 0.48 / 0.12 | 0.94 / 0.94 / 0.90 / 0.80 |
| stamp_seal | 0.74 / 0.74 / 0.74 / 0.70 | 1.00 / 0.62 / 0.38 / 0.10 | 0.80 / 0.92 / 0.88 / 0.70 |
| turn_switch | 0.48 / 0.46 / 0.44 / 0.52 | 0.68 / 0.72 / 0.64 / 0.60 | 0.68 / 0.60 / 0.60 / 0.66 |
| **macro(50 任务)** | **87.0 / 85.8 / 85.2 / 80.9** | **93.5 / 78.7 / 54.3 / 29.6** | **89.5 / 87.9 / 85.6 / 72.0** |

## 6. 推进计划

1. **本地 smoke(各 1 格新任务 × D0 × 50 eps,调 Cards 5/6,一卡一格)**:
   ours = stack_bowls_two(10-26 14:13 GPU 6 起评,在飞);AHA = click_bell
   (排 ours 后);StarWAM = place_shoe(平台 smoke 单提即可,shelf smoke
   可省 —— 平台链路 dual-server + 渲染接线与 shelf 不同,须平台验)。
2. **config 小改 + 四步自测**(10-19 铁律):✅ 10-26 全部完成 —— ours
   b6v2(7/7)、AHA v2 旋钮(2/2)、StarWAM 新入口(5/5)。
3. **平台提单(用户)**:每方法一单,8 卡;ours smoke ① 绿即可提 b6v2
   全量;AHA smoke ② 绿后提;StarWAM 提 smoke 单验平台链路,过即提全量。
   错峰建议见交付的提单指引表。
4. 收数判读(异常即停)→ §5 表格回填 → 本文档 + daojie md 双归档。

## 7. 状态日志(新条目在最上)
- **2026-11-09 08:5x ours 主根 D0 去污染落盘：macro 92.9 → 87.0（Ret 87.0% → 93.0%），论文主表/附录表/曲线图同步改**。§5 表 ours 列 D0 有 **20 格**是 11-08 定位的旧 lane 拷贝 bug 残留（`_result_clean.txt` 被兄弟 lane 覆盖成 0.98），本格 `delay_0/<task>/<task>.log` 末行的进程内计数才是真值。以 50 格 log 末行重算：**D0 2174/2500 = 86.96%**、D10 2146、D20 2129、D40 2022 全部与本格 log 一致（**只有 D0 变**，D10/D20/D40 与现表分毫不差），Ret = 2022/2174 = **93.01%**。已同步：本文档 §5 逐任务表 ours D0 列 20 格 + macro 行、`LeapBotCEPaper` 的 `Figures/data/robotwin_delay_summary.json`（含 provenance 更正说明与重算 sha256）、`robotwin_v2_pertask.csv`、重生成 `sec/X_pertask_robotwin.tex` 与 `fig_delay_curves.*`、主表 `tab:robotwin` 的 `\method` 行（92.9→87.0，Ret 87.0%→93.0%）。**验证依据**：`robotwin_v2_sweep_b6_d0recheck` 独立重跑 18/20 格，同 18 格宏平均 82.11(旧) vs 82.56(新跑)、逐格 mean|Δ| 4.67pp、无系统偏差 → 旧跑 log 值可信，重跑仅作验证不替换。剩余 2 格（`lift_pot`、`place_bread_basket`）未重跑：`lift_pot` 首格 curobo warp JIT 缓存 PCH 竞争（`catastrophic error: error while deleting ... .pch` → `CUDA kernel build failed with error code 6`）后进程内 planner 坏死，`eval_policy.py:266` 的 `while succ_seed < test_num` 无重试上限 ⇒ 空转 19h（5.4 万次 `target_pose cannot be None`、0 集），用户 11-09 已停单，两格取主线根 log 值。连带影响：中段 D10/D20 的 StarWAM 领先解释需改写（ours D0 现 **低于** StarWAM 89.5，不再是"分母小"）。
- **2026-11-02 13:05 StarWAM chunk8 结果入论文 Table 2(paper commit 24ca1fd+PAPER_TODO b645c2d,未 push)**。用户指示"更新到 Table 2,论文中不用做过多说明,本地文档备注"。tab:robotwin(编译编号实锤 = Table 2)新增行 `StarWAM-MoT (8-step chunk)†`:D0 88.6 / D10 -- / D20 -- / D40 26.0 / Ret 29.3%(未舍入 130/443);表尾一句脚注 "Ten-task subset, forced 8-step execution chunk; D∈{10,20} pending",正文未动。⚠️ 口径:该行 10 任务受限 × 人为 chunk 变体,与表内其它行(50 任务全量 × native cadence)不同口径 —— PAPER_TODO 已留口径提醒与 D{10,20} 回填待办(starwam rs8 六窗单收齐后)。latexmk 编译过(BUILD_OK)。任务表位置:tab:audit=Table 1,tab:robotwin=Table 2(PAPER_TODO 里旧指称 "Table 3" 已过时)。
- **2026-11-02 12:45 chunk8 探针双侧扩全六窗(starwam 718a072 / ours 1468683,checker 28/28)**。用户指示:chunk8 剂量曲线补全 D{0,10,20,40,60,80}。两变体 DELAYS default 均改六窗,既有 rs8 根 D{0,40} .done 幂等跳过 —— starwam 侧新增 40 格(立即可提,8×H100 预估 ~40h);ours 侧**必须等运行中单 22807684(D{0,40})收齐后重提同名**,否则在跑格无 .done 会被新单重复调度撞车。chunk8 链路首格实证已过(rs8 D40 格 replans/ep 8.9/8.6 vs chunk24 的 3.0,×3 精确符合;delay U{0..40} mean≈14 正常)。handover 复跑单 22809780 资源排队中。
- **2026-11-02 12:10 handover_block 矛盾定性反转 + 复跑变体就绪(commit fa7f8f6,checker 28/28)**。v1 根(robotwin_sweep_b6,10-22 批)读数实锤:handover_block D20=68/D40=22 —— 塌方在 v1/v2 **两批独立复现**,主根可信,**ext 轮 98/98 是孤例**;嫌疑收窄到 ext 轮特有因素(curobo 四架构重编译后 kernel 差异,双臂任务 handover_block/blocks_ranking_size 恰为 ext 轮偏高格),但 24→98 幅度未定,异常即停。复跑变体 `leapbot_ce_eval_robotwin_b6_handover`(AIFlow_config_eval_robotwin_b6_handover.py):矛盾本体 handover_block + 锚任务 beat_block_hammer(三方连续 80-90,校准当前代码态可比性)× D{20,40} × 显式 step_158206 × 独立根 `robotwin_sweep_b6_handover_recheck`,4 格 ~2-2.5h,判读矩阵三分支写入 docstring(≈24=当轮异常/≈98+锚正常=真实 V 形/锚也漂=curobo 嫌疑升级需重测 ext ours 侧)。裸 b6 快捷键指 v1 旧根被 .done 全 skip(单 22807730 秒退,未复跑任何东西),故走独立变体+独立根。剔除争议格后 ours 曲线 D40=95.1→D60=98.0 仍平,分离重现结论存续,§8.4 数值定性前带星号。同日:ours rs8 单 22807684(8×H100,lanes/GPU=2 正常)与 dtrain40 ep9 快评单 22808438(4×H100,delays:0 干净)运行中。
- **2026-11-02 11:30 D60/D80 + rs8 探针收数完成(§8.4 预测验证:分离重现) + 一处跨批次矛盾待定性**。10 任务受限 macro:StarWAM chunk24 = 96.0/95.4/93.4/79.6/**53.6/41.5**(D0-D80,末两窗 ext 根),ours chunk24 = 98.8/94.4/91.4/88.2/**98.0/99.0**(末两窗 b6_ext 根)—— 分离从 D40 +8.9pp 扩大到 D60 +44.4pp、D80 +57.5pp,§8.4"StarWAM 加速塌陷、ours 平缓衰减"预测**证实**。delay 注入双侧实证:ours ext delay_log U{0..60} mean=22.5/max=60、U{0..80} mean=29.4/max=80,50 eps/格,ckpt=step_158206 与主根冻结同一文件;StarWAM jsonl delay_mean 0-36.8(均值 25.1)。rs8(StarWAM 侧):chunk8 D0=88.6 → D40=**26.0**(vs chunk24 D40=79.6)—— StarWAM 曲线陡,曝光惩罚主导;D0 88.6 vs 96.0 → 短 chunk 无观测新鲜度增益。**ours 侧 rs8(b6_ext_rs8)尚未提单**。⚠️ 跨批次矛盾(异常即停留档):ours 主根 D40=88.2 被 handover_block 单任务拖累(D0 98→D20 58→D40 24 连续恶化),而 ext 轮同 ckpt 同 seed 下 handover_block D60/D80=98/98;两批各自内部自洽、互相矛盾,主根 handover_block@D{20,40} 两格待复跑定性——定性前,10 任务受限 D40=88.2 作为 rs8 的 ours 对照基线须带星号使用(剔除该格则为 95.8)。§8.4 数值待矛盾定性后更新。
- **2026-10-31 23:00 chunk 剂量-响应探针 rs8 基建就绪(FastWAM commit 0229447 + XPolicyLab 2413c8f + StarWAM 39565d2),待用户提单**。动机:与 D60/D80 探针互补的另一半 —— 固定 D、chunk 执行步数 24→8,delay 采样 ×3(~19→~56 replans/集),画 retention vs replan 频率剂量曲线,验证 §8.4"保护只延后不消除"。**双方法对称**(只改基线=审稿必打):ours `b6_ext_rs8` + StarWAM `starwam_ext_rs8`,10 任务同表 × D{0,40} 受限版,独立 rs8 结果根;chunk=24 对照取既有 v2 per-task 数据不重跑。实现链:基座 `AI_FLOW_REPLAN_STEPS`→`LEAPBOTCE_REPLAN_STEPS`→lane `EVALUATION.replan_steps`;StarWAM `AI_FLOW_REPLAN_STEPS`→`STARWAM_REPLAN_STEPS`→adapter env 覆盖(SERVER_HOST/PORT 同款模式);delay 协议零改动。判读矩阵(双向可证伪,写入变体 docstring):ours 平+StarWAM 陡=机制叙事最强证据;双平=保护只延后;StarWAM 反稳=混杂(观测更新增益>曝光惩罚)主导须改叙事。混杂披露:短 chunk 观测新鲜度增益(D0 可能升)+采样惩罚反向,retention 比值部分归一。自测 26/26 PASS;lane 脚本(baseline_delay/,非 git 管辖)bash -n 双过。⚠️ StarWAM rs8 链路新执行策略,首格盯防(沿用 10-26 先例不单独 smoke)。
- **2026-10-31 09:55 探针两单平台调度事故 + curobo 多架构重编译修复(commit 待填)**。
  两探针单(22727253 b6_ext 02:21 起跑 / 22727274 starwam_ext 09:14 起跑)均被平台
  调度到**非 H100-80G 节点**(PACK=1 → lanes/GPU=1,推算 VRAM<74GB),curobo 的
  5 个扩展 .so(09-16 于 sm_90 调试 pod 构建,单架构)缺目标节点内核 → 全 cell
  `RuntimeError: CUDA error: no kernel image is available` 死循环,零产出,b6_ext
  侧 log 膨胀至 ~917MB×8(约 82 卡时损失,待批后清理)。22727253 用户已停;
  22727274 cell log 自 09:46 冻结(90s 零增长,疑似已停,待平台确认)。
  **修复**:重编译 `code/RoboTwin/envs/curobo`(双消费 env fastwam-robotwin /
  robotwin editable 同指此树,torch 同 2.7.1+cu126,单次构建通用):
  `TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"` build_ext --inplace,5 .so 体积 2-3.4×
  (多架构特征);本 pod(H800 sm_90)MotionGen 完整链路回归(init+warmup+
  plan_single success=True,5 内核全过)。旧 .so 备份 `envs/curobo/so_backup_20261031/`,
  属主已修。fwv2 树无自有 curobo、共用此树,新 .so 含 sm_90 对 H 系节点安全。
  sm_80/86/89/90 覆盖平台现实卡池(A100-40G/A800-40G、A6000、L40S/4090、H 系);
  更老架构(V100/T4)不在池内,未编 PTX 前向兼容。
  **教训固化**:探针/全量任务起跑首屏判 `lanes/GPU=2 → 16 lanes` 才是 H100-80G
  正常节点,`=1` 即错节点,立即停单重提;重提时在平台备注/卡型栏指定 H100。
- **2026-10-31 01:50 D60/D80 时延探针基建就绪(commit a81174f),待用户提单**。动机:StarWAM-MoT v2 保留率 80.4% 与 ours 87.0% 接近,机制上是 chunk 保护(~16.6 replans/ep vs AHA ~450,27× 曝光差)在 D≤40 段未破;D60/D80(≫ours chunk 24 步)推出保护段,检验分离是否重现(LIBERO D40 分离 67.8pp)。冻结口径:10 任务 v1 子集 × D{60,80} × ours(B6 v2 同 ckpt)+StarWAM-MoT = 40 格;独立结果根 `leapbotce/robotwin_v2_sweep_b6_ext` / `starwam/robotwin_v2_sweep_ext`(防污染主根);对照窗 D{0,10,20,40} 从既有 per-task 数据取同 10 任务算受限 macro,不重跑。变体 `eval_robotwin_b6_ext` / `eval_robotwin_starwam_ext`(后者 env 预设+V2_SWEEP_ROOT patch),自测 24/24 PASS。两单各 8×H100 并行;ours 约 1.5-2h,StarWAM 约 12-15h。同 commit:a7 改 16×H100 重提(eff128,lr 2.8e-4,steps 43393,save_every 2170)。

- **2026-10-30 19:05 StarWAM v2 全量收齐入表(§5 填列)**。单 22560488(10-27 提),200/200 jsonl 全 50 行可解析、FAILED=0,末格 D40/stamp_seal 19:05 收尾。macro **89.5 / 87.9 / 85.6 / 72.0**,retention 80.4%——三基线中最平缓(ours 87.1% / AHA 31.7%),但 D0 绝对值最低(89.5 vs ours 92.9)。v2 三方法收数全部完成。

- **2026-10-29 10:39 AHA v2 全量收齐入表(§5 填列)**。200/200 `_result.txt` 全可解析,macro 93.5 / 78.7 / 54.3 / 29.6;四方法中退化最陡的基线,与 ours 的平缓 retention 曲线对照成立(机制叙事用)。StarWAM 180/200 收数中,收齐后补最后一列。

- **2026-10-28 19:15 ours v2 全量收齐入表(§5 填列)**。b6v2 单 22529754 平台报完成,复核 200/200(`_result_clean.txt` 全可解析、FAILED=0),macro **D0 92.9 / D10 85.8 / D20 85.2 / D40 80.9**,单调退化平滑无跳变格;最低格 = handover_block D40 0.24 / open_microwave D40 0.40(难任务量级合理)。v1 老根(22518318)由本单完整取代。同窗收数:AHA 179/200(`_result.txt` 深层路径 `delay_<D>/<task>/eval_result/<task>/AHA_WAM/…/<ts>/`)、StarWAM 134/200,收齐后补 §5 两列。

- **2026-10-26 18:50 平台提单落地 + StarWAM env 平台化收尾**。
  ① 用户已提并排队:**22529754 leapbot_ce_eval_robotwin_b6v2**(gpu:8)、
  **22529775 leapbot_ce_eval_robotwin_ahawam_v2**(gpu:8),资源紧张排队中。
  ② smoke 取舍:ours(96%)/AHA(100%)双绿且均有平台先例 → 无需再提;
  StarWAM 跳过独立 smoke、直接提正式版(队列紧张不双排队;StarWAM 格
  故障为快败型,首格盯防即可)。③ StarWAM 平台化最后一处:lane/env
  裸名(robotwin-2653853、starwam-libero-2653853)→ 共享盘 .envs prefix
  路径(AHA 平台教训 0e3fcf1e:平台 pod 无本机 env 注册,裸名失效;
  prefix 激活 shelf 行为不变,双 env import 级验证过)。STARWAM 正式
  任务名 `leapbot_ce_eval_robotwin_starwam_v2`,userPath=/dahuafs/userdata/2653853/code/StarWAM,
  零参数,×8 卡。

- **2026-10-26 17:55 v2 smoke ② GREEN(AHA 100%),lane ROOT 落根缺陷修复**。
  ① AHA click_bell × D0 × 50 eps:**50/50 = 100%**(31 min,管线健康),
  **P1 提单 `leapbot_ce_eval_robotwin_ahawam_v2` 解锁**。② 实锤一处
  通道缺陷:xpolicy **lane 驱动 ROOT 硬编码**旧根,cluster 驱动的
  SWEEP_BASE/SUFFIX 只影响自身记账 → smoke 产物先落错根(旧根
  delay_0/click_bell,与既有 10 任务不撞、数据真实),已 mv 搬正至
  ahawam/robotwin_v2_smoke;修复 = cluster 驱动 export
  XPOLICY_SWEEP_ROOT + lane 驱动 ROOT 可覆写(RoboTwin 9af26de,
  bash -n 双过,checker 2/2,旧根恢复 10 任务原样)—— **v2 全量提交
  的落根链路由此才真正打通**(未修则 AIFlow 全量会写进旧根)。
  StarWAM lane 同款坑已在通道建设时先行修复。

- **2026-10-26 17:30 v2 smoke ① GREEN 收官,smoke ②(AHA)启动**。
  ① ours stack_bowls_two × D0:lane 16:54 exit 0,50 eps 终值
  **96%**(_result_clean 0.96,.done 落;两段各 24/25)—— codec_lora
  ckpt(step_158206)+ 新任务管线(instruction/文本编码/评测)平台级
  首验通过。**P0 提单 `leapbot_ce_eval_robotwin_b6v2` 正式解锁**。
  ② AHA smoke(click_bell × D0 × 50 eps)16:56 起,GPU 6(调试卡,
  空闲 ~73GB ≥45GB 达标),xpolicy 驱动预检全绿(render OK/单 lane),
  root = ahawam/robotwin_v2_smoke(独立后缀防撞正式根)。
  同窗:B6 22518318 D0 7/10 收(已收 7 格 99.1%)、D10 6/10(97.3%)、
  D20 3/10;详见 daojie md §6 17:20 条。

- **2026-10-26 15:30 三方法平台提单通道全部打通**。①ours b6v2(FastWAM
  097065e,checker 7/7);②AHA v2 旋钮(RoboTwin b4ed4f6,checker 2/2);
  ③**StarWAM 通道收口**(code/StarWAM 非独立 git 仓,文件落盘+chown):
  - lane 驱动:`STARWAM_SWEEP_ROOT` env 覆写 + 显卡用户态接线改
    nvidia_userland_wire.sh 节点自适应(本 pod 干跑 = 精确 535.183.06 包,
    与原硬编码行为一致);
  - 新建 `baseline_delay/starwam_robotwin_v2_chains_2653853.sh`:4 窗 ×
    P=max(1,NGPU/4) 链,50 任务 round-robin 分链,平台 CVD 逻辑槽位;
    dry-run 三拓扑验证(8 卡=8 链×25 格,4 卡=4 链×50 格,smoke=1 链);
  - 新建 `code/StarWAM/AIFlow_config.py`(首个 StarWAM 平台入口)+
    `scripts_check/check_aiflow_dispatch_v2.py` **5/5 PASS**(全量/smoke/
    d2040/d010/哈希尾剥离);shlex 无空格值不加引号 → 自测断言按落盘形态。
  - clean vs random 口径:demo_clean(seen)为 v2 协议,与全部既有基线行
    同协议;demo_randomized 暂不做(用户 10-26 问询确认)。
- **2026-10-26 15:20 在飞状态**:B6 平台 22518318(四窗 40 格)D0 5/10
  done、D10 3/10,四旋钮修复在产线验证中;3 格日志内有 RoboTwin 任务级
  episode 异常(click_alarmclock/grab_roller 的 get_grasp_pose None),
  为"该 episode 判负、评测继续"的标准路径 —— click_alarmclock D0 仍以
  1.0 收格,非加载类故障。rung-9(s112k)后半收齐:handover 0.32 /
  put 0.70 / scan 0.76 / shake 1.00 / stack 0.98,十格均值 86.8%
  (85-90 平台带内;handover 回落至震荡带下沿)。v2 smoke ①(ours
  stack_bowls_two,GPU 6)在评。

- **2026-10-26 14:15 v2 smoke ①(ours)启动**:GPU 6 14:13 起评
  stack_bowls_two × D0 × 50 eps(root robotwin_v2_sweep_b6,ckpt
  step_158206,LEAPBOTCE_EVAL_TASK=robotwin_all_leapbotce_codec_lora
  四旋钮齐);过 = 40 新任务管线首验通过,再排 AHA(click_bell)/
  StarWAM(place_shoe)smoke。同窗:B6 平台重提(22518318)金丝雀
  全绿在跑(上轮第四旋钮事故已修复,详见 daojie md §6 14:00 条)。

- **2026-10-26 12:45 文档创建**。①50 任务清单实锤(envs/ 注册表,
  FastWAM/standalone 两树逐名一致);②三方法管线盘点完成(§4 资产卡),
  改动点:ours config v2 变体、AHA 根覆写、StarWAM ROOT 覆写,均小改 +
  四步自测;③B6 平台全量(10 任务四窗,H100×8,22515304)12:20 起跑
  金丝雀全绿(ckpt=step_158206 ✓ 四窗 ✓ 16 lanes ✓ render OK ✓)——
  与 v2 无冲突(10 任务在 robotwin_sweep_b6 根,v2 用独立根);④smoke
  排程:GPU 6 今日 14:00 起 ours → AHA,明日 StarWAM。

## 8. chunk 保护机制分析(论文 RoboTwin 段写作素材,10-31 冻结)

> 用途:解释 Table 3 三方法 RoboTwin 差异 + D60/D80 探针动机。写论文时
> 引用本节数字,勿凭记忆重算。

### 8.1 定义

frame-delay 协议下每次 **replan 才采样一次**时延 k~U{1..D};chunk 化策略
多步动作一次执行,replan 频次低 → 每集抽到大时延的次数少 → 退化被节拍
**结构性缓冲**。时延曝光 ∝ replans/集,与成功率高低无关,是评测构造量。

### 8.2 数值(本评测口径)

| 基准 | 方法 | replan 节奏 | ≈replans/集 | 相对 AHA 曝光 |
|---|---|---|---|---|
| LIBERO | ours + 四基线 | 每 10 步(32-action chunk 执行前 10 步) | 集长/10 | 全员相同 |
| RoboTwin(集≈450 步) | AHA-WAM | 每 1 步 | ~450 | 1×(无保护) |
| 同上 | ours | 每 24 步 | ~19 | ~1/24 |
| 同上 | StarWAM-MoT | ~每 27 步(实测 ~16.6 replans/ep) | ~16.6 | ~1/27 |

来源:节奏 = 论文协议(LIBERO 10 步、ours RoboTwin 24 步)+ StarWAM v2
运行实测(~16.6 replans/ep);集长 ~450 由 AHA per-step ~450 replans/ep 反推。

### 8.3 论证三角(优势归因)

1. **同节奏对照(LIBERO)**:全员每 10 步 replan、曝光完全相同,基线塌到
   D40 ret 2.8-13.5% vs ours 77.7% → LIBERO 差距与保护无关,纯训练机制。
2. **等保护对照(RoboTwin vs StarWAM)**:每集 ~19 vs ~16.6 次抽样(差 15%),
   D40 +8.9pp(≈12σ)、retention 93.0% vs 80.4%(11-09 D0 去污染后)、per-task ret 中位 0.97 vs
   0.84（CSV 整数口径；原记 0.94 vs 0.83）、头对头 D40 30/49 —— **等曝光下的净优势**。
3. **无保护对照(RoboTwin vs AHA)**:+55.4pp(Fig 1(b)/curves caption 已用)。
   AHA 每 1 步 replan 是唯一无保护方法。

三点互证把优势归因到 paired-age 训练机制,而非评测构造。

### 8.4 边界与预测(诚实披露,写作注意)

- **不能**把 RoboTwin 优势全归因于保护:ours/StarWAM 曝光几乎相同。
- **不能**回避 cadence 差异:AHA 对比必须披露 native cadence 保留
  (论文协议已声明 "systems with other native cadences retain them")。
- 中段(D10/D20)StarWAM 略先(D10 +2.1≈3σ 真、D20 +0.4 噪声):两法都在各自
  chunk 保护段内。**11-09 更正**:ours D0 去污染后为 87.0，**低于** StarWAM 89.5，
  原"其 D0 低(89.5 vs 92.9)分母小"的解释作废 —— 中段小领先现在发生在 ours 起点
  更低的情形下，retention 对比同步变为 93.0% vs 80.4%；论文中段措辞需按此重写。
- chunk 保护**只延后不消除**:每集仍有 ~17-19 次大时延抽样,D→保护段外
  (D≫节拍)失效。D60/D80 探针(§7 10-31 条)把 D 推出保护段
  (60/80 ≫ 24 步节拍),预测 StarWAM 加速塌陷、ours 平缓衰减(stale-condition
  训练泛化),分离应重现(LIBERO D40 已示 67.8pp 分离)。探针若证实,
  RoboTwin 段落按此收尾;若不证实,中段叙事维持"等保护下小幅领先"。
- **2026-11-09 探针结案（本条取代上一条的分支判断）**：native D60/D80 clean 重跑
  （`robotwin_v2_sweep_b6_ext2`，10 任务子集、每格 50 集）ours = **81.20 / 76.40**，
  对照同子集 StarWAM chunk24 = **53.8 / 41.6** → 分离 **+27.4 / +34.8 pp**，
  "StarWAM 加速塌陷、ours 平缓衰减"的预测**证实**。11-02 那批 D60/D80=98/98 的
  ours 读数是污染格值（本格 log 实为 10%/8%），跨批次矛盾不存在。全量 50 任务侧：
  ours rs8 D60/D80 = 79.8 / 75.1，StarWAM rs8 = 10.4 / 8.4（均 50/50 定稿）。
