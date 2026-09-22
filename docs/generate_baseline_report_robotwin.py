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
]

FOOTNOTES = """
---

### 协议与披露

- **基准**: RoboTwin 2.0 (SAPIEN, aloha/agilex 双臂), `demo_clean` 任务配置,
  seen 指令, 50 trials/任务 × 10 任务, 每延迟 n=500。
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
"""


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_cell(run_root: str, task: str, delay: int):
    """Return dict(rate, n, done) for one (task, delay) cell; None if absent."""
    delay_dir = os.path.join(run_root, f"delay_{delay}", task)
    if not os.path.isdir(delay_dir):
        return None
    # completed attempts: _result.txt (last line = success rate as a float)
    result_files = sorted(glob.glob(os.path.join(delay_dir, "**", "_result.txt"),
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
