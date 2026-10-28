#!/usr/bin/env python3
"""RoboTwin v2 三方法(b6v2 ours / AHA / StarWAM)四窗结果聚合 → §5 表格行。

数据口径(与 docs/baseline_results_robotwin_v2.md §5 一致):
- ours  (leapbotce/robotwin_v2_sweep_b6): delay_<D>/<task>/_result_clean.txt 尾行 float
- AHA   (ahawam/robotwin_v2_sweep):       delay_<D>/<task>/**/_result.txt 尾行 float
          (深层路径 eval_result/<task>/AHA_WAM/.../<ts>/,取 mtime 最新一份)
- StarWAM (starwam/robotwin_v2_sweep):    delay_<D>/<task>_delay_metrics.jsonl
          逐集 "success" 布尔求均值(完整=50 行)

默认打印每方法完整度 n/200 与四窗 macro;--md 追加 50 行 markdown
(缺格打 "—"),可直接粘进 §5。ours macro 应复现 92.9/85.8/85.2/80.9
(2026-10-28 收官锚点),不符即停止输出并 exit 1(异常即停)。
"""

import glob
import json
import os
import re
import statistics
import sys

BASE = "/mnt/c2-worldmodel/2653853/baseline-evaluations"
ROOTS = {
    "ours": f"{BASE}/leapbotce/robotwin_v2_sweep_b6",
    "AHA": f"{BASE}/ahawam/robotwin_v2_sweep",
    "StarWAM": f"{BASE}/starwam/robotwin_v2_sweep",
}
DELAYS = (0, 10, 20, 40)
# ours 收官锚点(50 任务 macro,百分比),聚合口径回归用
OURS_ANCHOR = (92.9, 85.8, 85.2, 80.9)


def parse_float_tail(path: str):
    tail = open(path).read().strip().splitlines()
    nums = re.findall(r"[01]\.\d+\s*$", tail[-1]) if tail else []
    return float(nums[0]) if nums else None


def read_cell(method: str, delay: int, task: str):
    root = ROOTS[method]
    if method == "ours":
        f = f"{root}/delay_{delay}/{task}/_result_clean.txt"
        return parse_float_tail(f) if os.path.isfile(f) else None
    if method == "AHA":
        hits = glob.glob(f"{root}/delay_{delay}/{task}/**/_result.txt", recursive=True)
        if not hits:
            return None
        return parse_float_tail(max(hits, key=os.path.getmtime))
    # StarWAM
    f = f"{root}/delay_{delay}/{task}_delay_metrics.jsonl"
    if not os.path.isfile(f):
        return None
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if not rows or any("success" not in r for r in rows):
        return None
    return statistics.mean(bool(r["success"]) for r in rows)


def load(method: str, tasks):
    """-> {delay: {task: rate}}, 完整格才有键。"""
    table = {}
    for d in DELAYS:
        table[d] = {t: read_cell(method, d, t) for t in tasks}
    return table


def macros(table):
    return {d: statistics.mean(v for v in table[d].values() if v is not None)
            for d in DELAYS}


def main() -> int:
    md = "--md" in sys.argv
    # 任务清单 = ours delay_0 实际目录(ours 已 200/200 收官,即 50 任务权威表)
    tasks = sorted(os.path.basename(p)
                   for p in glob.glob(f"{ROOTS['ours']}/delay_0/*")
                   if os.path.isdir(p))
    assert len(tasks) == 50, f"ours delay_0 任务数 {len(tasks)} != 50"

    data, counts = {}, {}
    for m in ROOTS:
        data[m] = load(m, tasks)
        counts[m] = sum(v is not None for d in DELAYS for v in data[m][d].values())

    mac = {m: macros(data[m]) for m in ROOTS}
    anchor = tuple(round(mac["ours"][d] * 100, 1) for d in DELAYS)
    if anchor != OURS_ANCHOR:
        print(f"ABORT: ours macro {anchor} != 锚点 {OURS_ANCHOR}(口径漂移,异常即停)")
        return 1

    for m in ROOTS:
        print(f"{m}: {counts[m]}/200 格;macro " +
              " / ".join(f"D{d}={mac[m][d]*100:.1f}" for d in DELAYS))

    if md:
        print()
        for t in tasks:
            cells = []
            for m in ROOTS:
                cells.append(" / ".join(
                    "—" if data[m][d][t] is None else f"{data[m][d][t]:.2f}"
                    for d in DELAYS))
            print(f"| {t} | {cells[0]} | {cells[1]} | {cells[2]} |")
        row = " | ".join(
            f"**{' / '.join(f'{mac[m][d]*100:.1f}' for d in DELAYS)}**" for m in ROOTS)
        print(f"| **macro(50 任务)** | {row} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
