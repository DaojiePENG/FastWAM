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

# display name -> run root under EVAL_ROOT. Add future methods (X-WAM,
# DreamZero, AHA-WAM, ...) as separate entries; each root must contain
# delay_<D>/<task>/... cells as produced by the lane driver.
METHOD_ROOTS = [
    ("Motus (stage-2 ckpt)", f"{EVAL_ROOT}/motus/robotwin_sweep"),
    ("X-WAM (robotwin_sft)", f"{EVAL_ROOT}/x-wam/robotwin_sweep"),
    ("Fast-WAM (official release)", f"{EVAL_ROOT}/fastwam/robotwin_sweep"),
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
  seed=1234 的 RNG 采样 d ~ U{0..D}; Motus replan 每 8 个控制步 (chunk=8)。
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
"""


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


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
    """methods -> delay -> {"successes": s, "total": n, "cells": {task: cell}}"""
    table = {}
    for name, run_root in METHOD_ROOTS:
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
        "| 方法 | D=0 | D=5 | D=10 | D=20 | D=40 |",
        "|---|---|---|---|---|---|",
    ]
    for name, rows in table.items():
        cells = [fmt_cell(rows, d) for d in DELAY_KEYS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "### 每任务明细", ""]
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
