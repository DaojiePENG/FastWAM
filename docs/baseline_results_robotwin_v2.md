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
**ours 已收齐(2026-10-28,单 22529754,b6v2);AHA / StarWAM 收数中,列待填。**

| 任务 | ours D0/D10/D20/D40 | AHA D0/D10/D20/D40 | StarWAM D0/D10/D20/D40 |
|---|---|---|---|
| adjust_bottle | 1.00 / 1.00 / 1.00 / 1.00 | 待填 | 待填 |
| beat_block_hammer | 0.98 / 0.96 / 0.92 / 0.90 | 待填 | 待填 |
| blocks_ranking_rgb | 0.98 / 1.00 / 1.00 / 0.98 | 待填 | 待填 |
| blocks_ranking_size | 0.98 / 0.84 / 0.72 / 0.80 | 待填 | 待填 |
| click_alarmclock | 1.00 / 1.00 / 1.00 / 1.00 | 待填 | 待填 |
| click_bell | 1.00 / 1.00 / 1.00 / 1.00 | 待填 | 待填 |
| dump_bin_bigbin | 0.98 / 0.92 / 0.94 / 0.98 | 待填 | 待填 |
| grab_roller | 1.00 / 1.00 / 0.98 / 0.98 | 待填 | 待填 |
| handover_block | 0.98 / 0.72 / 0.58 / 0.24 | 待填 | 待填 |
| handover_mic | 0.98 / 1.00 / 1.00 / 0.94 | 待填 | 待填 |
| hanging_mug | 0.98 / 0.56 / 0.54 / 0.46 | 待填 | 待填 |
| lift_pot | 0.98 / 0.92 / 1.00 / 0.92 | 待填 | 待填 |
| move_can_pot | 0.98 / 0.54 / 0.44 / 0.46 | 待填 | 待填 |
| move_pillbottle_pad | 0.98 / 0.96 / 0.96 / 0.96 | 待填 | 待填 |
| move_playingcard_away | 0.98 / 0.96 / 0.96 / 0.96 | 待填 | 待填 |
| move_stapler_pad | 0.98 / 0.60 / 0.58 / 0.50 | 待填 | 待填 |
| open_laptop | 0.98 / 0.80 / 0.84 / 0.76 | 待填 | 待填 |
| open_microwave | 0.98 / 0.40 / 0.46 / 0.40 | 待填 | 待填 |
| pick_diverse_bottles | 0.98 / 0.80 / 0.68 / 0.70 | 待填 | 待填 |
| pick_dual_bottles | 0.90 / 0.88 / 0.96 / 0.90 | 待填 | 待填 |
| place_a2b_left | 1.00 / 0.96 / 0.94 / 0.92 | 待填 | 待填 |
| place_a2b_right | 0.98 / 0.88 / 0.88 / 0.88 | 待填 | 待填 |
| place_bread_basket | 0.98 / 0.86 / 0.84 / 0.90 | 待填 | 待填 |
| place_bread_skillet | 0.74 / 0.88 / 0.84 / 0.74 | 待填 | 待填 |
| place_burger_fries | 0.98 / 0.94 / 0.92 / 0.94 | 待填 | 待填 |
| place_can_basket | 0.98 / 0.84 / 0.82 / 0.72 | 待填 | 待填 |
| place_cans_plasticbox | 0.98 / 0.84 / 0.84 / 0.86 | 待填 | 待填 |
| place_container_plate | 0.98 / 1.00 / 0.92 / 1.00 | 待填 | 待填 |
| place_dual_shoes | 0.82 / 0.76 / 0.74 / 0.72 | 待填 | 待填 |
| place_empty_cup | 0.98 / 1.00 / 1.00 / 0.98 | 待填 | 待填 |
| place_fan | 0.98 / 0.94 / 0.94 / 0.82 | 待填 | 待填 |
| place_mouse_pad | 0.98 / 0.64 / 0.66 / 0.62 | 待填 | 待填 |
| place_object_basket | 0.80 / 0.86 / 0.90 / 0.84 | 待填 | 待填 |
| place_object_scale | 0.92 / 0.84 / 0.86 / 0.82 | 待填 | 待填 |
| place_object_stand | 0.96 / 0.96 / 0.94 / 0.90 | 待填 | 待填 |
| place_phone_stand | 0.88 / 0.90 / 0.88 / 0.86 | 待填 | 待填 |
| place_shoe | 0.88 / 0.88 / 0.92 / 0.84 | 待填 | 待填 |
| press_stapler | 0.90 / 0.90 / 0.88 / 0.88 | 待填 | 待填 |
| put_bottles_dustbin | 0.94 / 0.94 / 0.84 / 0.44 | 待填 | 待填 |
| put_object_cabinet | 0.82 / 0.84 / 0.86 / 0.88 | 待填 | 待填 |
| rotate_qrcode | 0.90 / 0.94 / 0.94 / 0.76 | 待填 | 待填 |
| scan_object | 0.70 / 0.64 / 0.74 / 0.60 | 待填 | 待填 |
| shake_bottle | 1.00 / 1.00 / 1.00 / 1.00 | 待填 | 待填 |
| shake_bottle_horizontally | 1.00 / 1.00 / 1.00 / 1.00 | 待填 | 待填 |
| stack_blocks_three | 0.96 / 0.86 / 0.92 / 0.86 | 待填 | 待填 |
| stack_blocks_two | 0.98 / 0.98 / 1.00 / 0.96 | 待填 | 待填 |
| stack_bowls_three | 0.66 / 0.80 / 0.84 / 0.74 | 待填 | 待填 |
| stack_bowls_two | 0.96 / 0.98 / 0.98 / 0.90 | 待填 | 待填 |
| stamp_seal | 0.74 / 0.74 / 0.74 / 0.70 | 待填 | 待填 |
| turn_switch | 0.48 / 0.46 / 0.44 / 0.52 | 待填 | 待填 |
| **macro(50 任务)** | **92.9 / 85.8 / 85.2 / 80.9** | 待填 | 待填 |

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
