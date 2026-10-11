#!/usr/bin/env python3
"""Regenerate FastWAM/docs/baseline_results_robotwin.md (RoboTwin 2.0 benchmark).

Run any time (safe to re-run while evaluations are in progress):
    python /dahuafs/userdata/2653853/code/FastWAM/docs/generate_baseline_report_robotwin.py

Kept strictly separate from the LIBERO report (baseline_results_libero.md /
generate_baseline_report_libero.py). Scans each method's RoboTwin delay-sweep
run root, reads per-(task, delay) cells, and renders the markdown report:
  * 主表: 每个方法一行, 列 = 帧延迟, 单元格 = 成功率(%); 加粗 = 该延迟下全部
    单元完成 (n = 50 trials × 任务数); `(n=k)` = 部分完成
  * 附录: 每个任务 × 延迟的明细表
"""
from __future__ import annotations

import glob
import json
import os
import re
from datetime import datetime

EVAL_ROOT = "/mnt/c2-worldmodel/2653853/baseline-evaluations"
DOCS_DIR = "/dahuafs/userdata/2653853/code/FastWAM/docs"
OUT_PATH = os.path.join(DOCS_DIR, "baseline_results_robotwin.md")

DELAY_KEYS = [0, 5, 10, 20, 40]
TRIALS_PER_TASK = 50

# 10-task subset spanning single-arm precision, dynamic, dual-arm and
# multi-object task families (protocol decision for the RoboTwin campaign).
TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "click_alarmclock",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "put_object_cabinet",
    "scan_object",
    "shake_bottle",
    "stack_blocks_two",
]

# display name -> (run root under EVAL_ROOT, audit schema for the
# replans/ep column). Add future methods as separate entries; each root must
# contain delay_<D>/<task>/... cells as produced by the lane driver.
# audit schema:
#   "flat"     one jsonl row per replan (episode/replan_idx keys)
#              — Fast-WAM lane, LeapBot-ce lane, DreamZero
#   "samples"  one jsonl row per episode carrying delay_samples[]
#              — X-WAM / Motus lanes
#   "aha"      one jsonl row per trial with result.delay_metrics.sample_count
#              — AHA-WAM lane
METHOD_ROOTS = [
    ("LeapBot-ce (ours, dual-stream)", f"{EVAL_ROOT}/leapbotce/robotwin_sweep", "flat"),
    ("Motus (stage-2 ckpt)", f"{EVAL_ROOT}/motus/robotwin_sweep", "samples"),
    ("X-WAM (robotwin_sft)", f"{EVAL_ROOT}/x-wam/robotwin_sweep", "samples"),
    ("Fast-WAM (official release)", f"{EVAL_ROOT}/fastwam/robotwin_sweep", "flat"),
    ("DreamZero (AgiBot)", f"{EVAL_ROOT}/dreamzero/robotwin_sweep", "flat"),
    ("AHA-WAM (robotwin)", f"{EVAL_ROOT}/ahawam/robotwin_sweep", "aha"),
]

FOOTNOTES = """
---

### 协议与披露

- **基准**: RoboTwin 2.0 (SAPIEN, aloha-agilex 双臂), `demo_clean` 任务配置,
  seen 指令, 50 trials/任务 × 10 任务, 每延迟 n=500。所有方法行的仿真本体均由
  task_config 的 `embodiment: [aloha-agilex]` 决定 (env_cfg_type=arx_x5 仅影响
  policy 侧动作维度, 且 aloha-agilex 与 dual_x5 维度一致: arm [6,6] / ee [1,1],
  状态/动作均 14 维)。
- **帧延迟协议** (CloudEdgeVLA 风格): 单流全观测老化; 每次 replan 从
  seed=1234 的 per-episode RNG 采样, replan 时观测取 hist[-1-d]。延迟分布
  存在实现差异 (LIBERO 侧同样如此, 见 LIBERO 报告): X-WAM/Motus 为
  d ~ U{0..D} (允许 0 = 新鲜观测), Fast-WAM 官方式实现为
  d ~ U{1..min(D,len-1)} (至少 1 步老化; 首个 replan 前仅 1 帧历史时退化
  为 0)。Fast-WAM 每 24 控制步 replan (官方 RoboTwin 默认 replan_steps=24),
  Motus 每 8 步 (chunk=8), X-WAM 全 chunk (Ta=32) 执行, AHA-WAM 几乎逐步
  replan (audit 实测 147-795 次/ep)。Fast-WAM 泳道
  `skip_get_obs_within_replan=false`, 即老化窗口需要逐步渲染观测流。
- **延迟剂量披露 (replans/ep 列)**: 延迟观测的毒害剂量 ≈ 抽样次数 × 实际老化
  深度; 各泳道抽样节奏不同 (见上), `replans/ep (D=40)` 列把节奏差异变成
  audit 实测数字 (每回合延迟抽样次数的均值)。同基准内: Fast-WAM 16.5 vs
  AHA-WAM ~450 (27×); 跨基准: LIBERO 泳道 chunk=10 系统同为 10 步一抽
  (上限 40 次/ep, 实测 26-40 随成功率变化), Fast-WAM 在 RoboTwin 的官方
  24 步 replan 只有其节奏的四成 —— 跨环境对比延迟鲁棒性时必须把该列
  一并披露 (见下方归因小节)。
- **expert_check**: 采用 RoboTwin 评审标准做法 —— 脚本专家都失败的种子被跳过
  (种子消耗与策略无关, 跨延迟一致)。
- **Motus 权重披露**: 官方仅公开 Stage-2 预训练底座 (HuggingFace,
  global_steps 420k), 论文的 RoboTwin finetune 权重未发布; 因此该行评估的是
  预训练底座, 与论文报告的 87% (finetune) **不可直接对比**。为加载该 ckpt 对
  推理几何做了适配: action chunk 16→8 (匹配训练时 action-only 8-token 布局),
  registers 4→0, 丢弃几何不匹配的确定性 sin-cos pos buffer, state encoder
  权重借自 ckpt 的 action encoder。代码适配见
  `code/Motus/inference/robotwin/Motus/models/motus.py` (load_checkpoint /
  MOTUS_NUM_REGISTERS)。
- **Motus sweep 已暂停 (2026-09-23, 用户确认)**: 离线探针证实 released 底座
  在 RoboTwin 上输出**退化动作** — 动作块恒 ≈0.01±0.01, 对 state (±0.3 rad)
  与视觉 (真实 vs 灰图) 扰动几乎无响应 (max|Δ|≈0.05, 与噪声同量级); episode
  视频中机器人几乎不动, 150+ episodes 全 0%。根因: 发布物自相矛盾 — ckpt
  几何为 8 action-only token (无 state encoder/registers), 而官方推理代码
  与 README 按 48-token+state+registers 构建 (官方示例 strict 加载失败)。
  README 的 87.02% 依赖未公开 finetune 权重。表中 Motus 行仅含已完成
  adjust_bottle 三格 (0.0), 其余为 —; 详见评估根目录下探针与审计日志。
- **X-WAM 权重与协议适配**: 使用官方已发布的 `robotwin_sft` finetune 权重
  (EE-space, action_type=ee, 3 视角 head/left/right, 16-d proprio, delta 动作
  在 anchor 状态上积分还原绝对 EE 位姿, replan_steps=None 执行整块预测)。
  坐标系: 训练帧 = 环境帧绕 z 轴旋转 -90° (`pos_train=(y_env,-x_env,z_env)`,
  `R_train=Rz(-90°)@R_env`, delta 按 `R'=dR@R` 世界系左乘积分), 已用
  config.yaml 分位角 8 位有效数字逐项验证; 专家状态对照探针证实 grip 通道
  (pre-grasp 观测下 Σdgrip=-0.978 满闭, anchor 扫描极性正确)。
  帧延迟协议移植自 Motus adapter 并保持语义一致 (离线探针
  `code/baseline_delay/probe_xwam_delay_2653853.py` 逐项验证: RNG 流回放一致、
  全观测老化、逐 episode 审计 jsonl 落盘)。每 episode 审计:
  `<run_root>/delay_<D>/<task>_delay_metrics.jsonl`。
- **X-WAM sweep 已暂停 (2026-09-29, 用户确认跳过)**: 帧修复后闭环 0/140+
  (beat_block_hammer D=0/10/20/40 全 0, adjust_bottle D=40 完整格 2/50=4%,
  官方声称 20 任务均值 89.8%), 差距 ~86 个百分点。逐项排查已关闭适配器侧
  嫌疑: 坐标系变换 8 位有效数字逐项核对、grip 头在专家 pre-grasp 观测下
  正确满闭 (Σdgrip=-0.978)、anchor 极性扫描正常、执行保真 0.0001、条件
  图像正常。残余失败为**轨迹级**: 模型首块转移高度偏低 (~0.92 vs 专家
  上绕 0.99), 开指掠过瓶子将其撞走, 此后无法重建 pre-grasp。官方数字
  出自作者内部 stack (未随仓库发布; 仓库附带的 XPolicyLab 适配有帧 bug,
  未被作者端到端跑通过), 复现其声称水平所需成本超出本研究预算 → 按用户
  决策跳过。保留数据: `x-wam/robotwin_sweep/delay_40/` 两完整格 +
  `_archive_framebug_20260928/` 帧修复前全部作废产物。
- **LeapBot-ce (ours) RoboTwin sweep 已停止 (2026-10-11, 用户确认不再投入)**:
  sweep 事实上已于当日 12:44 跑完 (50/50 格全有 .done 标记), 无进程残留;
  但结果**不达结论门槛, 行保留仅备查, 不入任何结论/论文**。判定依据:
  D=0 仅 7.4%, 其中 shake_bottle 一项 (74%) 贡献了全部 37 次成功,
  其余 9 任务全 0.0 (同格 Fast-WAM 99.6); 且 42.1 replans/ep 剂量正常
  —— 失败不是协议问题而是模型侧:
  RoboTwin 的 CloudEdge 配方/ckpt 未就位 (与 LEAPBOTCE.md「RoboTwin CE
  未实现」一致, pool_grid [6,10] override 也从未生效过)。继续烧卡无意义,
  数据保留于 `leapbotce/robotwin_sweep/` 供配方就位后对照。后续计划
  (用户 2026-10-11 确认): 等 LIBERO 消融定夺最终配方后, 用最终最优方案
  重开 RoboTwin 泳道正式测评一次, 中间不再单独排查/重训。

### Fast-WAM 曲线为何远好于 LIBERO 泳道 (2026-10-11 延迟审计归因)

Fast-WAM RoboTwin 只掉 18 点而 LIBERO 同款官方 ckpt 掉 95 点, 复核结论:
**两组数字都真 (实现核查 + audit jsonl 复核), 差异来自"延迟暴露剂量"不可比**,
不是 RoboTwin 实现偏松。按证据强度排列:

1. **replan 间隔差 2.4× (协议事实)**。延迟只在 replan 那一刻注入, 两次 replan
   之间开环执行。RoboTwin 用官方默认 `replan_steps=24` → 实测 5.4-30.9 次
   stale 规划/ep (10 任务均值 16.5); LIBERO 泳道 `replan=10` → D=40 实测
   ~40 次/ep。单位时间暴露剂量 RoboTwin 是 LIBERO 的 ~40%。
2. **同环境内部对照 (最硬的证据)**。AHA-WAM 同一批任务、同协议、实际延迟均值
   同为 ~19 帧, 但几乎逐步 replan (audit: 147-795 次/ep, 均值 ~450 ≈ Fast-WAM
   的 27×) → 掉 97.6→48.6 (-49)。环境本身延迟敏感, Fast-WAM 的 -18 是
   暴露次数被 24 步 chunk 压低后的结果。
3. **per-task replans/ep 与掉点排序两方法都几乎一致** (延迟按"每次 replan
   抽一次签"计费, 抽签次数决定剂量):
   - Fast-WAM: click_alarmclock 5.4 次/ep → 掉 0, grab_roller 8.3 → 0,
     shake_bottle 7.9 → -4; handover_block 30.9 → -52, stack_blocks 29.9 → -20,
     put_object_cabinet 26.6 → -26。
   - AHA: click_alarmclock 147 → 0, shake_bottle 186 → 0, grab_roller 258 → -6;
     handover 795 → -88, stack_blocks 791 → -92, put_object_cabinet 674 → -68。
4. **任务构成**: click_alarmclock / shake_bottle / grab_roller 目标准静态,
   任意暴露量级都几乎不掉 (AHA 147-258 次暴露仍 100/100/94), 3/10 任务托住
   均值; LIBERO 四套件全是精密短回合操作, 没有这类"免死"任务。
5. **ckpt 变体差异 (次要, 未核查)**: LIBERO 行是 `libero_uncond_2cam224`
   无条件变体, RoboTwin 行是官方 release 原生配方, 两者本身稳健性可能不同。

**论文处理**: 跨环境对比延迟鲁棒性必须披露 replan interval (10 vs 24) 或把
曲线按每回合 replan 次数归一; 表述避免"Fast-WAM 对延迟鲁棒", 改为"低 replan
频次 + 准静态任务占比高使 RoboTwin 泳道暴露剂量低"。可选补强: RoboTwin
Fast-WAM `replan_steps=10` 对照 (D=40 一格 ~500 eps) 直接量化协议敏感性,
需要时按现有 sweep 脚本加一个 override 即可。
"""


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def collect_draws_per_ep(run_root: str, audit: str, delay: int = 40):
    """Measured delay draws per episode from a lane's audit jsonl at `delay`.

    延迟剂量 = 每回合策略消费延迟观测（并据其提交动作）的次数。不同泳道的
    audit 落盘格式不同（见 METHOD_ROOTS 注释），逐类解析后返回每回合计数的
    列表（空列表 = 无 audit 数据，主表显示 —）。
    """
    per_ep = []
    if audit == "flat":
        for tf in glob.glob(os.path.join(run_root, f"delay_{delay}", "*",
                                         "*delay_log*.jsonl")):
            eps: dict = {}
            for ln in open(tf, errors="ignore"):
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                if "episode" in r and "replan_idx" in r:
                    eps[r["episode"]] = eps.get(r["episode"], 0) + 1
            per_ep += list(eps.values())
    elif audit == "samples":
        for tf in glob.glob(os.path.join(run_root, f"delay_{delay}",
                                         "*_delay_metrics.jsonl")):
            for ln in open(tf, errors="ignore"):
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                samples = r.get("delay_samples")
                if isinstance(samples, list) and samples:
                    per_ep.append(len(samples))
    elif audit == "aha":
        for tf in glob.glob(os.path.join(run_root, f"delay_{delay}", "*",
                                         "delay_metrics_*.jsonl")):
            for ln in open(tf, errors="ignore"):
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                dm = (r.get("result") or {}).get("delay_metrics") or {}
                if dm.get("sample_count"):
                    per_ep.append(int(dm["sample_count"]))
    return per_ep


def fmt_draws(per_ep):
    if not per_ep:
        return "—"
    return f"{sum(per_ep) / len(per_ep):.1f}"


def parse_cell(run_root: str, task: str, delay: int):
    """Return dict(rate, n, done) for one (task, delay) cell; None if absent."""
    delay_dir = os.path.join(run_root, f"delay_{delay}", task)
    if not os.path.isdir(delay_dir):
        return None
    # completed attempts: _result.txt / _result_clean.txt / _result_random.txt
    # (RoboTwin's eval loop suffixes the result file with the task_config)
    result_files = sorted(glob.glob(os.path.join(delay_dir, "**", "_result*.txt"),
                                    recursive=True))
    if result_files:
        with open(result_files[-1]) as fh:
            lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
        if lines:
            try:
                return {"rate": float(lines[-1]), "n": TRIALS_PER_TASK, "done": True}
            except ValueError:
                pass
    # in-progress / dead attempts: newest "Success rate: s/n" from cell logs
    best = None
    for log in sorted(glob.glob(os.path.join(delay_dir, f"{task}*.log"))):
        txt = strip_ansi(open(log, errors="ignore").read())
        for m in re.finditer(r"Success rate: (\d+)/(\d+)", txt):
            best = (int(m.group(1)), int(m.group(2)))
    if best and best[1] > 0:
        return {"rate": best[0] / best[1], "n": best[1], "done": False}
    return {"rate": None, "n": 0, "done": False}


def collect():
    """methods -> delay -> {"successes": s, "total": n, "cells": {task: cell}};
    附带 draws(D=40) = 每回合延迟抽样次数列表（audit 实测）。"""
    table = {}
    for name, run_root, audit in METHOD_ROOTS:
        rows = {}
        for delay in DELAY_KEYS:
            cells = {}
            for task in TASKS:
                cell = parse_cell(run_root, task, delay)
                if cell is not None:
                    cells[task] = cell
            s = sum(int(round(c["rate"] * TRIALS_PER_TASK)) for c in cells.values()
                    if c["rate"] is not None)
            n = sum(c["n"] for c in cells.values())
            rows[delay] = {"successes": s, "total": n, "cells": cells}
        rows["draws40"] = collect_draws_per_ep(run_root, audit, 40)
        table[name] = rows
    return table


def fmt_cell(rows, delay):
    cell = rows[delay]
    if cell["total"] == 0:
        return "—"
    rate = 100.0 * cell["successes"] / cell["total"]
    complete = cell["total"] >= TRIALS_PER_TASK * len(TASKS)
    txt = f"{rate:.1f}" if complete else f"{rate:.1f} (n={cell['total']})"
    return f"**{txt}**" if complete else txt


def main():
    table = collect()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# RoboTwin 2.0 帧延迟基线 (CloudEdgeVLA 协议)",
        "",
        f"*更新: {now} · 与 LIBERO 报告独立: 见 `baseline_results_libero.md`*",
        "",
        "## 主表 (10 任务 × 50 trials, 成功率 %)",
        "",
        "| 方法 | D=0 | D=5 | D=10 | D=20 | D=40 | replans/ep (D=40) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, rows in table.items():
        cells = [fmt_cell(rows, d) for d in DELAY_KEYS]
        cells.append(fmt_draws(rows.get("draws40")))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "`replans/ep (D=40)` = 各泳道 audit jsonl 实测的每回合延迟抽样次数（均值；"
        "不同方法抽样节奏不同——Fast-WAM 24 步一 replan、Motus chunk=8、X-WAM 全块执行、"
        "AHA-WAM 逐步 replan——该列把节奏差异变成显式数字，见下方披露）。",
        "",
        "### 每任务明细",
        "",
    ]
    for name, rows in table.items():
        lines += [f"**{name}**", "", "| 任务 | D=0 | D=5 | D=10 | D=20 | D=40 |", "|---|---|---|---|---|---|"]
        for task in TASKS:
            row = [task]
            for d in DELAY_KEYS:
                c = rows[d]["cells"].get(task)
                if c is None or c["rate"] is None:
                    row.append("—")
                elif c["done"]:
                    row.append(f"**{100.0 * c['rate']:.1f}**")
                else:
                    row.append(f"{100.0 * c['rate']:.1f} (n={c['n']})")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    lines.append(FOOTNOTES)
    with open(OUT_PATH, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
