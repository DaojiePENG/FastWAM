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
| 入口 | `baseline_delay/starwam_robotwin_sweep_lane_2653853.sh <GPU> <D> <task...>`(任务走 argv ✓) |
| 结果根 | ROOT 硬编码 `starwam/robotwin_sweep` → **需加 env 覆写(小改)** |
| smoke 先例 | 10-11 beat_block_hammer D0 MoT 5/5 通过 |
| 未知风险 | 新任务 text cache(precompute_text_cache.sh);server+sim 同卡显存 |

## 5. 结果表(待填)

每方法一张 50 任务 × 4 窗成功率表 + 全任务均值;收数脚本沿用
`_result_clean.txt` 尾行读取。**模板(收工后填):**

| 任务 | ours D0/D10/D20/D40 | AHA D0/D10/D20/D40 | StarWAM D0/D10/D20/D40 |
|---|---|---|---|
| (50 行) | | | |

## 6. 推进计划

1. **本地 smoke(各 1 格新任务 × D0 × 50 eps,调 Cards 5/6,一卡一格)**:
   ours = stack_bowls_two;AHA = click_bell;StarWAM = place_shoe。
   调度:GPU 6 今日 ~14:00(rung-10 前半收工后)依次 ours → AHA;
   StarWAM 明日 GPU 6。
2. **config 小改 + 四步自测**(10-19 铁律):ours b6v2 变体、AHA v2 旋钮、
   StarWAM ROOT 覆写;`check_aiflow_dispatch.py` 扩 CASES。
3. **平台提单(用户)**:每方法一单,8 卡;建议 ours → AHA → StarWAM
   错峰(空闲卡常态 ~32,可两单并行)。
4. 收数判读(异常即停)→ §5 表格回填 → 本文档 + daojie md 双归档。

## 7. 状态日志(新条目在最上)

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
