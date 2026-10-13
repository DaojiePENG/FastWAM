# LeapBotCE 实验监测与 AIFlow 提交记录

> 用途：跟进实验状态 + 提交 AIFlow 申请的速查。新条目在最上。
> 技术论证见 `leapbotce_dualstream_gap_analysis.md`，配方细节见根目录 `LEAPBOTCE.md`。

## 2026-10-13 头号根因定位：冻结专家一直是【原始 Wan】而非 FastWAM LIBERO release + `mot_pretrained_path` 开关

**用户疑点成立（代码实锤）**：官方 FastWAM 的 97.4% 来自视频专家在 LIBERO 上联训的
release ckpt（`/mnt/c2-worldmodel/2638088/fastwam/fastwam_release/libero_uncond_2cam224.pt`，
`payload = {mot: 两专家, proprio_encoder, step, torch_dtype}`）；而我们 A0/a1-a3 的模型配置
`model_id: Wan-AI/Wan2.2-TI2V-5B` 直出原始 Wan + `ActionDiT_linear_interp` 初始化，
**冻结的视频专家从未见过 LIBERO 分布**。24 点 gap（97.4→73.2）的头号嫌疑从"冻结/解冻"
升级为"冻结在错误的起点"。冻结点本身在
`src/fastwam/models/wan22/leapbotce_edge_train_factory.py:20`（`model.dit = nn.ModuleDict(可训模块)`，
trainer 的 dit-only 模式因此只解冻 action_expert/projector/adapter/codec，5B 视频专家留在
`requires_grad_(False)`）；上游对照：官方 trainer `_apply_dit_only_train_mode` 训的是
`model.dit = self.mot` = 两专家一起 —— 官方 97.4 本来就是联训产物。

**修复（已落地，`mot_pretrained_path` 开关）**：

- `leapbotce_factory.py`：新增 `mot_pretrained_path` 参数，构建后调 `load_checkpoint` 走
  v1 full-MoT 分支（`payload["mot"]` 严格载入两专家 + proprio_encoder；edge 侧
  projector/codec 不碰，保持自身初始化）。
- `configs/model/leapbotce.yaml`：`mot_pretrained_path: null` 默认（null = 原行为）。
- `scripts/eval_leapbotce_libero_sweep_2653853.sh`：任务推导白名单**补上**
  `model.mot_pretrained_path` 转发 —— 不补的话 a0_release 评测会拿原始 Wan 重建冻结基座
  + release 训出的 delta，静默错配（delta ckpt 只存微调部分，见 `save_checkpoint`）。
- 验证：release payload 键结构与我们的 MoT 完全匹配（1649 键全在 `mixtures.{video,action}`，
  proprio 8→4096）；GPU 6 全模型构建冒烟 6 张量（video block 0/29、action、proprio w/b）
  与 release 精确相等，edge projector 原样。

**新训练变体（待提交）**：`leapbot_ce_libero_a0_release`（AIFlow_config_libero_a0_release.py）
= A0 底座逐项对齐（task=libero_all_leapbotce、d=20/seed42、eff128、24951 步、stale -1）
**唯一差异** `model.mot_pretrained_path=<release>`；4×H200 ~2.1 天。读数：
`a0_release − A0(73.2)` = 冻结专家域内预训练的净贡献；接近 97.4 则确认 init 是主因，
解冻（第二梯）降级为可选。RoboTwin 同理换 `robotwin_uncond_3cam_384.pt`（指南 §11）。

**同起点（原始 Wan）codec vs A0 修正读数（10-13 下午补）**：之前"codec 占 ~10-15 点"
是错位对比（a1 spatial 59 对 A0 四套件均值 73.2 —— object 套件 95-97% 拉高了均值）。
同套件 spatial 对比：**A0 56.0/57.0/52.5（n=400/格，D=0/10/20）vs a1 codec
59.0/61.0/63.0（n=100/格）vs a2 decoder 51/54/58 —— codec(tied) 与 A0 无统计差甚至
略优，a2 略差但 CI(±10) 重叠**。含义：105× 压缩在同起点下近乎零成本，24 点 gap
与压缩无关、全部来自 init/训练配方 → codec 作为 claim 安全；主 claim 线补
`leapbot_ce_libero_a1_release`（AIFlow_config_libero_a1_release.py = a1 消融线
+ release init，其余逐项同 a1），a0_release 角色改为对照（分解 压缩成本 vs
训练配方成本）。decoder 路线（a2）是否加码（如 rank 256→512、pool_grid 加 token）
等两个信号再定：① 标准规模 a1/a2 评测（判决 tied vs decoder），② a1_release
落点 —— 在原始 Wan 的 garbage-in 区调 codec 设计没有意义。

## 一、现在该提交什么（更新于 2026-10-13 下午）

| 任务名（整段照抄） | 配置文件 | 卡数 | 预计耗时 | 状态 |
|---|---|---|---|---|
| `leapbot_ce_libero_a1_release` | AIFlow_config_libero_a1_release.py | 8×H200 | ~1.1 天 | ⬜ 待提交（**主 claim 线：codec(tied) + release**） |
| `leapbot_ce_libero_a0_release` | AIFlow_config_libero_a0_release.py | 8×H200 | ~1.1 天 | ⬜ 待提交（对照：无 codec + release） |
| `leapbot_ce_libero_a2_release` | AIFlow_config_libero_a2_release.py | 8×H200 | ~1.1 天 | ⬜ 待提交（decoder 因子 + release） |
| `leapbot_ce_libero_a3_release` | AIFlow_config_libero_a3_release.py | 8×H200 | ~1.1 天 | ⬜ 待提交（grid 挑选因子 + release） |

- **决策（10-13 下午，用户拍板）**：原始 Wan 线的标准评测全部取消——同起点下
  codec≈A0 已由 diag 数字确认，tied-vs-decoder 改在 release 域内判决。
  32 卡空闲 → 四条 release 线 4×8 卡并行，~1.1 天（A0 实测 25.7h）全部落地，
  一次拿全 {无压缩, codec, +decoder, +grid} × release init 的完整消融矩阵。
- 四线 eff batch 128 不变（8×bs16×accum1）；只有 4 卡时改回 GPU_NUM=4/ACCUM=2。
- 训练完评测：各线 terminal ckpt 用 AIFlow_config_eval_libero 路线换 RUN_DIR/TAG
  （a{0,1,2,3}_release_step24951），sweep 脚本会从 config.yaml 自动转发
  mot_pretrained_path 重建冻结基座。

## 二、提交前核对（30 秒）

1. **任务名别改字**：分发器按名字关键字最长匹配。名字只有 `libero` 没有变体关键字
   （a1_codec / a2_decoder / a3_grid）的会落回基础配置 = 200k 续训线（带 `--resume`），
   误提交 = 重复续训。
2. **节点选 H200**：H100-80G 跑不了 bs16（bs8 实测峰值 64.2GB）。只有 H100 时先说，
   我出 bs8/accum4 等价版。
3. 想快：申请 8 卡 + 该配置里 `AI_FLOW_ACCUM = 1` → ~1.1 天（eff batch 128 不变，
   动力学与 4 卡版一致；`AI_FLOW_GPU_NUM` 同步改 8）。
4. 平台 pod 镜像 torch 是残包：训练配置已全部显式钉死 bs/lr/steps/卡数，别放开自动表。

## 三、消融矩阵与结果回收（LIBERO only）

| 项 | 是什么 | 结果回收 |
|---|---|---|
| A0 | frozen-cloud 基线（h200 定标那版，D=0 65-73） | 已有（tag dtrain20_seed42_h200） |
| A1 | codec 主配方（云 56 token 无解码器） | run `codec_a1_seed42_h200` |
| A2 | A1 + 边缘侧 K/V 低秩解码器 | run `codec_a2_decoder_seed42_h200` |
| A3 | A1 + 边缘挑选改 grid（均匀间隔） | run `codec_a3_grid_seed42_h200` |
| A4 | 边缘贡献（**不训练**，A1 ckpt 评测时关掉边缘） | TAG `codec_a1_seed42_h200_noedge` |

评测 codec 线：`AIFlow_config_eval_libero.py` 常改区 `AI_FLOW_RUN_DIR` →
`.../runs/libero_all_leapbotce_codec/<run_name>`，`AI_FLOW_TAG` → 对应 run 名；
A4 额外 `AI_FLOW_EVAL_EXTRA = "model.cloudedge.edge_prefix_enabled=false"`。

- 消融定夺规则：A2 若明显优于 A1 → RoboTwin 最终方案带解码器；A3 若优于 A1 → 挑选换 grid。
- RoboTwin 正式跑时 pool_grid 须 override `[6,10]`（token 网格 12×10），配置文件注释已写。

## 四、记录（新在上）

### 2026-10-13 终点 ckpt 诊断（a1/a2）+ 预算勘误 + 标准评测变体
- 训练收尾确认：a1/a2 正常目录 0 traceback，最后动作 = 写 step_024951
  weights+state（a2 10-13 02:00、a1 06:59）；a3 ETA ~16:00。
- 终点诊断（GPU 7 双 worker，libero_spatial × D{0,10,20} × 10 trials）：

  | 线 | D=0 | D=10 | D=20 | ret@20 |
  |---|---|---|---|---|
  | a1 tied | 59.0 | 61.0 | 63.0 | 1.068 |
  | a2 解码器 | 51.0 | 54.0 | 58.0 | 1.137 |

- **反转**：33% ckpt 时 a2 领先 2-3 点，终点上 a1 点估计反超 5-8 点
  （CI 重叠，n=100 ±10）。解码器在此预算下未兑现（嫌疑：随机初始化
  瓶颈欠收敛 vs 纯噪声）；判决 = 标准规模评测（500 eps/格，CI ±4.4），
  主线选择（tied vs decoder）等该数字。两条 retention 都 >1（平/升）。
- **预算勘误（重要，防止再犯）**：eff batch 128 不是科学常数 = √LR 折算
  自官方 eff16 配方（`LEAPBOTCE.md:88,101,181,208`）。P0 的"200k 步"
  是 @eff16 = **3.2M 样本**；消融线 24951 步 @eff128 = 3.19M 样本 =
  **同一预算**。即 codec 三线已是 P0 全预算，"正式线"无需再加跑
  200k@eff128（那是 8× 预算）。若学习曲线未饱和要加预算：2× = 6.4M 样本
  ≈ 8 卡 eff128 49902 步 ~2.2 天，或 16 卡 eff256（lr 4e-4）~1.1 天
  （dynamics 未验证）。
- 标准规模消融评测改走分发器变体：新增 `AIFlow_config_eval_libero_a{1,2,3}.py`
  （薄委托钉死 RUN_DIR/CKPT/TAG/SUITES/DELAYS/TRIALS；任务名
  `leapbot_ce_eval_libero_a{1,2,3}`，4×H200，可并行提交零串味；
  本地 import 冒烟全过）。a1/a2 即提；a3 等 ~16:00 ckpt 落盘后提。
  今晚标准数字出 → 定主线 → 胜者跑四套件 × 5 窗全量 sweep（填 Table 2）。
- RoboTwin codec 前置不变：pool_grid=[6,10] 适配 + 单格冒烟，今天备明天提。

### 2026-10-12 三线 step_008317（33%）调试机诊断评测 + summarize 脚本修复
- 用户要求对 a1/a2/a3 的 step_008317.pt（33% 进度）做初步测评。调试机 GPU 7
  （4/5/6 被 RoboTwin fork 占用；7 有 7.8GB 常驻可共存）。配置：libero_spatial、
  D∈{0,10,20}、10 trials（=100 eps/格）、PACK=1、单格实测 ~35-38 min。
- 结果（success rate / retention vs D=0，Wilson95 ±10 点）：

  | 线 | D=0 | D=10 | D=20 | ret@20 |
  |---|---|---|---|---|
  | a1 tied | 44.0 | 45.0 | 37.0 | 0.841 |
  | a2 解码器主线 | 46.0 | 48.0 | 42.0 | 0.913 |
  | a3 grid | 50.0 | 46.0 | 54.0 | 1.080 |

- 读法：三线 retention 都平（84–108%），codec 接口的延迟鲁棒性在诊断规模立住；
  a2 全窗口 ≥ a1 且差距随 D 单调拉大（+2/+3/+5），方向符合解码器机制预期；
  a3 nominal 最高但三线 CI 大量重叠，33% 训练 + 100 eps/格下消融不可分，
  消融表数字等 24951 终点 ckpt + 标准规模。
- **教训**：`LIBERO_EVAL_DELAYS` 必须空格分隔（脚本 `read -ra` 按 IFS 切分）；
  逗号版会先创建 `delay_0,10,20` 残留目录再炸 hydra，且该残留会让
  `summarize_leapbotce_delays.py` 在评测全部完成后 ValueError 挡汇总。
  已修脚本（过滤非整数 delay 目录，commit `95680ee`）+ 删 a2 根目录残留。
- 结果：`/mnt/c2-worldmodel/2653853/baseline-evaluations/leapbotce/
  {a1_codec,a2_decoder,a3_grid}_step8317_diag/delay_sweep_summary.json`。
  TAG 绑 ckpt：后续 step_16634 / 24951 复评须换新 TAG。

### 2026-10-11（晚 2）主线翻转为"开 K/V 解码器版"（LeapBotCEPaper f1987c4）
- 用户拍板：**最终版本讲开解码器的（A2 recipe = a2_decoder run）**，tied K=V
  （A1 = a1_codec run）降为消融行，专门量化解码器在零传输成本下的贡献。
  在训三线不用动：a2_decoder = full 主线，a1_codec = tied 消融，a3_grid = grid 消融。
- 论述修正两处代码事实（读 `cloud_kv_adapter.py` 核实）：
  ① 解码器是**纯线性**低秩瓶颈 `dec_up(dec_down(z))`，无 GELU（旧 KV adapter
  才带 GELU + 零初始化；新解码器默认初始化、非零初始化）；② 逐层 scale
  初始化为 1。eq:edgekv 已按线性形式重写。
- 接口参数账 9M → **12M**（codec 6.3M 云侧 + 解码器 3.1M 边缘 + 前缀投影
  2.4M）；摘要/引言/图注/实验/补充材料/结论全部同步，真机节边缘阶段行
  加了 K/V decode。main.pdf 11 页 / suppl 4 页，0 未定义引用。

### 2026-10-11（晚）论文重构至 codec 主线 + 真机时延节（LeapBotCEPaper 9f4a75a）
- **题目**改为 "LeapBot-CE: Planning-Token Codec for Delay-Robust Cloud-Edge
  Manipulation with Frozen World Models"。主线 = CloudPlanningCodec（z 载荷
  56×3072 ≈ **0.34MB bf16**，对 36MB 全 cache ~105×；`~52×` 是 LEAPBOTCE.md 里
  fp32 z 对 bf16 cache 的混精度口径，论文里统一按同精度算）+ edge select 前缀
  （2 views × 5 token = 10 token，EdgePrefixProjector 768→3072 带逐层 scale）。
- **方法章重写**：新增 "Cloud Planning Codec" 小节（eq:codec 空间池化→深度
  softmax 混合→瓶颈 MLP；eq:edgekv 边缘广播 + 逐层 scale，可选 rank-256 低秩
  K/V 解码器）；原 "Optional Cloud KV Adapter" 小节删除并入 codec（eq:adapter
  已删，全仓无悬空引用）；Edge Grounding 小节从 pooled 单 4096-d token 改为
  select 前缀（eq:edgeseq 序列 = [plan 56 | edge 10 | action 32]）。
- **实验章**：110k sweep 数字显式标注为 uncompressed-interface（A0）配置；
  消融表重构为两块（接口块 A0→A4 + 训练块）；带宽段重写（codec 载荷数学 +
  对比 CEVLA 1.8MB）；**新增 Sec 4.6 真机部署节**：Franka + 云 4090D-48GB +
  边缘 RTX 5080（暂定），加 RTT {0,400,1000}ms 沿 CEVLA 协议；表 5（成功率）
  表 6（时延分解：云 VAE+prefill / codec / 上行 0.34MB / 边缘 SigLIP+prefix /
  10 步去噪，mean/p50/p95）全占位 + 预期论述（0.34MB ≈ 2.7ms@1Gbps /
  28ms@100Mbps vs 36MB 的 288ms/2.9s → codec 是 WLAN 级链路可部署前提；云侧
  预期 10^2 ms 量级被 10 步 chunk 摊销；边缘 10 ms 量级永不等云）。
- 摘要/引言/相关工作/结论同步改口径；补充材料 codec 节替换原 adapter 节，
  两处跨文档 \cref 悬空引用改为字面节号（独立编译本来就解析不到主文 label）。
- 编译验证：main.pdf 11 页、X_suppl.pdf 4 页、0 未定义引用（8 页限制是终稿
  事，todo 红字清除后再压）。PAPER_TODO.md 已同步（表 3 新计划、表 5/6 数据
  准备指引、fig2 草图标过时）。

### 2026-10-11（下午）codec 消融评测链路调通（本地卡 4,5 冒烟 a1/a2/a3 全过）
- 背景：三条消融训练在跑（~0.2 step/s，首个 ckpt save_every=8317 预计当晚
  落盘），趁等待把评测代码和 AIFlow 启动脚本先调通。三个 run 都还没有
  ckpt → 用**合成 ckpt** 冒烟：200k 真实 action_expert/proprio_encoder +
  随机初始化 codec/select 投影（load_checkpoint 守卫允许，冒烟只验链路
  不验质量，成功率 0% 是预期正确）。
- **sweep 脚本改动**（`42a04c9`）：原写死 `task=libero_all_leapbotce`
  （frozen-cloud），codec ckpt 会在加载期 ValueError。现从
  RUN_DIR/config.yaml **自动推导** task（kv_codec.enabled → codec task）
  并把 kv_codec/edge_vision 字段钉成 eval 覆盖（pool_grid/use_kv_decoder/
  selection/mode/tokens_per_view/num_views + edge_prefix_enabled=false 时
  透传）；LIBERO_EVAL_TASK 可强制覆盖。AIFlow eval 配置加
  AI_FLOW_EVAL_TASK 透传 + 三条消融线 RUN_DIR 注释（只需换 RUN_DIR +
  新 TAG，task 不用手设）。
- **冒烟结果**（GPU 4,5 空闲卡，D=0 × libero_spatial × 2 trials，
  PACK=1 legacy 模式，独立 TAG smoke_codec_*_1011）：
  a1 codec 10/10 ✓ / a2 解码器 10/10 ✓ / a3 grid 10/10 ✓（各 ~7-13 min，
  manager config 证实 select/grid/use_kv_decoder 覆盖全部生效；
  replans=800/20ep 符合 chunk=10 节奏 schema）。fake run dir 留在
  `runs/libero_all_leapbotce_codec/_smoke_codec_a{1,2,3}/`（各 2GB）可复用。
- **正式评测提交口令**（真实 ckpt 落盘后）：AIFlow_config_eval_libero.py
  常改区 RUN_DIR→对应 codec run、TAG→新名（如 codec_a1_seed42_h200），
  其余不动。

### 2026-10-11（14:38）a3 重提后秒挂（节点抽奖）：.envs 死 shebang 已全量修复
- **现象**：a3_grid 重提后落在节点 10.194.160.30，2 分钟内死：
  `train_zero1.sh: line 111: .envs/leapbot-va/bin/accelerate: cannot execute:
  required file not found`。a1/a2 落在的节点正常（已在步进，ETA ~31-38h，
  **不用动它们**）。
- **根因**：envs 从 miniconda3 迁到 code/.envs 时**没有重写 console 脚本的
  shebang**——accelerate 等 1245 个脚本仍指向
  `#!/home/myuser/miniconda3/envs/<env>/bin/python3.10`，而该路径已随迁移
  清除。有 pod 本地残留旧 miniconda3 的节点能侥幸解析 → 节点抽奖；
  a3 抽到的节点没有 → 秒挂。18 个 env 全有雷（leapbot-va 内 85/144 脚本）。
- **修复**：`.envs/*/bin/` 全量重写 shebang →
  `#!/dahuafs/userdata/2653853/code/.envs/<env>/bin/python`（绝对共享盘路径，
  与节点无关）；18 个 env 清零，`accelerate` 本机实测可执行。对已运行任务
  无影响（进程已起，解释器在 exec 时已解析）。
- **下一步**：**只重提 a3_grid**，a1/a2 继续跑。

### 2026-10-11（下午）消融三线首提全挂 → edge 前缀 K/V 维度 bug 当日修复
- **现象**：a1_codec / a2_decoder / a3_grid 三个任务全部"训练程序错误"，
  各自启动后 1-2 分钟死于第一个训练 step。
- **排查路径**：平台日志在 `FastWAM/jinnTrainResult/<任务名>-<hash>/train.log`
  （**用户指定：以后排错优先看这里**）；三份 log 同一错误
  `RuntimeError: Expected size 3072 but got size 4096` @
  `mot.py:563 forward_action_with_video_cache_tensor` 的
  `torch.cat([video_cache_k, edge_prefix_k, k_action], dim=1)`。
  （run 目录内 wandb 佐证：a1/a2 .wandb 仅 7 字节=硬杀未刷盘；a3 优雅退出
  留下完整 traceback，与本诊断一致。）
- **根因**：`EdgeVisionEncoder` select 模式的 `edge_dim` 静默回退
  `text_dim`（umt5-xxl = 4096），而边缘前缀 K/V 要与 video cache K/V
  （`num_heads*head_dim` = 3072，kv_adapter/kv_codec 都正确用了这个）在
  注意力里拼接；TI2V-5B 两者不等。**pooled 模式（A0/frozen-cloud）输出
  的是 context token，本来就活在 text_dim，所以老配方从未触发**。
  三个消融全走 select 模式 → 全挂，A3 的 grid/topk 差异无关。
- **修复**（`0abdfde`）：`LeapBotCE.__init__` 构造 EdgeVisionEncoder 时
  显式传 `edge_dim = num_heads * head_dim`；select 模式缺 `edge_dim` 改为
  构造期 ValueError（不再静默回退）；补回归测试。26/26 过；另用真 SigLIP
  在 CPU 验证形状链：修后前缀 [B,10,3072] 与 cache 对齐，旧默认正是
  [B,10,4096]。
- **下一步**：三线**原配置重提即可**（代码修复对 pod 生效取决于仓库挂载
  /dahuafs 路径，无需改配置）；a2/a3 的 EXTRA_OVERRIDES 核对无误。

### 2026-10-11（傍晚）RoboTwin 我们泳道停测决策落实（用户确认）
- 用户决策：**LeapBot-ce RoboTwin 评测泳道停止，不再投入**（"意义不大"）。
- 核查：该 sweep 其实**已自行跑完**——50/50 格 `.done` 齐全，最后一格
  delay_40/stack_blocks_two 今日 12:44 收尾（lane 日志 `_cluster_logs/`）；
  当前无进程、无 tmux/screen，无需杀任何东西，也不烧卡了。若曾有经
  AIFlow 平台的提交，平台侧任务也已自然结束。
- 处置：**数据保留不删**（`leapbotce/robotwin_sweep/`）；生成器 FOOTNOTES
  加「已停止」披露 bullet，报告已重生成；行保留备查，**不入结论/论文**。
- 判定依据（D=0 逐任务实测 `_result_clean.txt`）：总 7.4% = shake_bottle
  一项 74% 贡献全部 37 次成功，**其余 9 任务全 0.0**（同格 Fast-WAM 99.6）；
  42.1 replans/ep 剂量正常 → 失败是模型侧（RoboTwin CE 配方/ckpt 未就位，
  与 LEAPBOTCE.md「RoboTwin CE 未实现」一致），非协议/审计问题。
- 连带影响：RoboTwin 最终方案（A2 解码器 / A3 grid 的取舍原依赖消融回收）
  的落地目标需重新定夺——LIBERO 消融三线照常等回收，但结果先只用于
  LIBERO 结论；RoboTwin 侧除非配方就位并重训，否则不重开泳道。
- **后续计划（用户 2026-10-11 确认）**：中间过程不再折腾（不为 RoboTwin
  单独排查/重训），等 LIBERO 消融（A1/A2/A3）回收、最终配方定夺后，
  **直接用最终最优方案重开 RoboTwin 泳道做正式测评一次**。前提注意：
  若最终方案是 CE codec 系（A1/A2/A3），RoboTwin CE 推理路径必须先真正
  跑通（pool_grid [6,10] / token 网格 12×10；本次 D=0=7.4% 已证实当前
  路径不可用）；若最终方案是 A0 frozen-cloud，评测路径与 Fast-WAM 泳道
  同构，成本最低。

### 2026-10-11（午后）两份 baseline 报告 + 论文 Table 1 加「replans/ep (D=40)」列（延迟剂量）
- 用户要求：本地总结表和论文表各加一列各模型预估 replan 次数并纳入分析。已落地
  （列口径 = audit 实测的每回合延迟抽样次数，即策略消费一次延迟观测并提交动作记一次）。
- **LIBERO 值**（D=40）：FastWAM 39.9 / StarWAM-MoT 39.9 / SharedDiT 39.8
  （10 步一抽、400 步上限、几乎全失败 → 跑满）；OpenWAM **581**（`policy.act`
  **每控制步**抽一次 —— 协议披露修正：它不是 10 步一 replan）；LingBot 46.3
  （原生 chunk ~16 步）；LeapBotCE 25.9/26.1（同节奏，成功率高 → 成功回合提前
  收尾；**剂量差异是成功率的结果不是原因，解读别反**）。OpenWAM 剂量最高却最
  鲁棒 → 快环架构补偿，剂量是必要披露维度但非唯一变量。
- **RoboTwin 值**：Fast-WAM 16.5（5.4-30.9/任务）vs AHA-WAM 450.7（147-795）
  = 27×；X-WAM 12.9（2 任务）；我们泳道 42.1。
- 生成器改动（两份都改，直接改 md 会被下次刷新覆盖）：LIBERO 加
  `draws_per_ep`（replans → 直方图 → OpenWAM videos delay_samples 三级回退，
  按 max_delay_steps 过滤）+ 机制表「抽样节奏」列 + 「延迟剂量披露」段；
  修 `parse_leapbotce_sweep` suite/全量 replans 口径 bug（虚高 4×，127→26）；
  RoboTwin 加三类 audit 解析（flat/samples/aha）。归因小节已并入生成器
  FOOTNOTES（重生成不再丢）。
- 论文 `sec/4_experiments.tex`：tab:audit 加 draws/ep 列 + caption 定义；
  Mechanism 段加剂量分析（chunk 系统同剂量 → 差异在模型侧；OpenWAM 581 次/ep
  最鲁棒 → 快环效应）；Confounds 段加跨基准节奏披露（RoboTwin Fast-WAM 24 步
  replan = LIBERO 泳道 2.4× 低暴露；AHA 逐步）。pdflatex 编译通过。
- ⚠️ 顺带发现：重生成后**我们自己的 RoboTwin 泳道行浮出**（之前 10-05 生成时无
  数据），D=0 仅 7.4%（shake_bottle 74，其余任务全 0），42 replans/ep ——
  该泳道质量有问题，RoboTwin 正式方案前需排查（疑似 ckpt/配方未就位，与
  LEAPBOTCE.md「RoboTwin CE 未实现」一致）。

### 2026-10-11（下午）RoboTwin Fast-WAM 曲线反常性归因完成（延迟审计复核）
- 用户问：Fast-WAM RoboTwin 只掉 18 点，LIBERO 同款官方 ckpt 掉 95 点，是否反常。
  **结论：两组数字都真，差异来自延迟暴露剂量不可比**（详见
  `baseline_results_robotwin.md` 新增归因小节）。
- 核心证据链：① RoboTwin 官方默认 replan=24 步 vs LIBERO 泳道 replan=10，
  实测 stale 规划 16.5 vs ~40 次/ep，暴露剂量差 2.4×；② 同环境 AHA 几乎逐步
  replan（147-795 次/ep）同 mean≈19 帧延迟掉 -49 → 环境本身延迟敏感，
  Fast-WAM 的 -18 是暴露被 chunk 压低的结果；③ per-task replans/ep 与掉点
  排序两方法都几乎一致；④ 3/10 准静态任务（click/shake/grab）任意暴露都
  不掉，托住均值，LIBERO 无此类任务。
- 待用户定夺：是否补 RoboTwin Fast-WAM `replan_steps=10` 对照格（D=40 一格
  ~500 eps）直接量化协议敏感性；论文表述按归因小节口径写。

### 2026-10-11（中午）codec 消融三线配置就绪；200k 线今天收尾
- **在跑**：200k 续训 11:52 实测 step 198,150/200,000（eta ~23min，speed 1.33 step/s，
  lr 已退火到 1.0e-06），预计 ~12:15 落盘 step_200000。10-10 17:07 那次提交
  （torch 残包秒挂）之后 17:22 重提成功，跑了 ~18.5h。
- **已备好**：消融三线 AIFlow 变体配置（a1/a2/a3，各 4×H200 ≈2.1 天，eff128 与 A0 逐项
  对齐）；已验证 dispatcher 分发 / 命令生成 / hydra compose / launcher dry-run。
- **耗时校准**：A0 h200 基线 8×H200 全程 25.7h（≈3.7 s/step @eff128，wandb 时间戳）。
- 评测脚本加 `LIBERO_EVAL_EXTRA` 透传（A4 用）；eval 配置加 `RUN_DIR`/`EVAL_EXTRA` 透传。
- codec 实现（CloudPlanningCodec + 边缘 select 前缀）今日已落地：25/25 测试 +
  真 SigLIP 冒烟（timm SigLIP 无 CLS → mean-patch 参考，NaN 隐患已修）。

### 2026-10-10（晚）110k 评测全量成功；200k 续训带病上线后恢复
- 110k tag 全量评测（d0/5/10/20/40 × 4 套件 × 50 trials，seed 1234）success（19:49）。
- 17:07 提交的 200k 续训秒挂（pod torch 残包，`torch.cuda` AttributeError）；
  配置加防御后 17:22 重提成功。
- 同日平台首提 8×H200 训练也死于代理不通弄死 rank0（历史教训，已双保险：
  启动脚本自动降 offline + trainer 降级，wandb 曲线事后本地 sync）。
