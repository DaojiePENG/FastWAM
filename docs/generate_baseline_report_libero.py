#!/usr/bin/env python3
"""Regenerate FastWAM/docs/baseline_results_libero.md (LIBERO benchmark) from the raw result JSONs.

Run any time (it is safe to re-run while evaluations are in progress):
    python /dahuafs/userdata/2653853/code/FastWAM/docs/generate_baseline_report.py

It scans /mnt/c2-worldmodel/2653853/baseline-evaluations/<baseline>/... for the
delay-sweep summary files each harness writes, unifies the per-delay numbers,
and renders the markdown report:
  * 主表: CloudEdgeVLA 风格 —— 每个方法一行, 列 = 帧延迟, 单元格 = 成功率(%)
  * 对比曲线图 baseline_delay_curve_libero.png
  * 附录: 每个运行的明细表
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime

EVAL_ROOT = "/mnt/c2-worldmodel/2653853/baseline-evaluations"
CODE_ROOT = "/dahuafs/userdata/2653853/code"
DOCS_DIR = os.path.join(CODE_ROOT, "FastWAM", "docs")
OUT_PATH = os.path.join(DOCS_DIR, "baseline_results_libero.md")
FIG_PATH = os.path.join(DOCS_DIR, "baseline_delay_curve_libero.png")

DELAY_KEYS = [0, 5, 10, 20, 40]

# task suites, in report order. Run roots embed the suite either as
# libero_<suite> (FastWAM) or libero-<suite>-... (StarWAM/OpenWAM style).
SUITES = ["spatial", "object", "goal", "10"]
_SUITE_TOKENS = {s: (f"libero_{s}", f"libero-{s}") for s in SUITES}


def suite_of(path):
    """Extract the task suite from a run path; default spatial (legacy roots)."""
    p = str(path)
    if "libero_plus" in p or "libero-plus" in p:
        # Kairos's LIBERO-Plus category protocol is not comparable with the
        # libero_spatial main table († footnote) — appendix-only.
        return "kairos-plus"
    for s in SUITES:
        if any(tok in p for tok in _SUITE_TOKENS[s]):
            return s
    return "spatial"


# display name -> (internal keys to merge, footnote tag)
METHODS = [
    ("FastWAM", ["FastWAM"]),
    ("OpenWAM (dual-system)", ["OpenWAM (dual-system)"]),
    ("LingBot-VA", ["LingBot-VA"]),
    ("StarWAM-MoT", ["StarWAM-MoT"]),
    ("StarWAM-SharedDiT", ["StarWAM-SharedDiT"]),
    ("Kairos †", ["Kairos"]),
    ("LeapBot-DualLane-V2 (在研)", ["LeapBot-DualLane-V2"]),
    ("LeapBotCE (d_train=20)", ["LeapBotCE (d_train=20)"]),
    ("LeapBotCE (d_train=20, 110k)", ["LeapBotCE (d_train=20, 110k)"]),
    ("LeapBotCE (d_train=20, 200k)", ["LeapBotCE (d_train=20, 200k)"]),
    ("LeapBotCE (d_train=20, h200 25k)", ["LeapBotCE (d_train=20, h200 25k)"]),
    ("LeapBotCE (d_train=40)", ["LeapBotCE (d_train=40)"]),
]


# --------------------------------------------------------------------------
# parsers: each returns {delay: row-dict}
# row-dict: successes,total,mean_delay,replans,hist
# --------------------------------------------------------------------------
def _row(successes, total, mean_delay=None, replans=None, hist=None):
    return {
        "successes": successes,
        "total": total,
        "mean_delay": mean_delay,
        "replans": replans,
        "hist": hist or {},
    }


def parse_sweep_summary(path):
    """FastWAM (windows) / LingBot (rows) delay_sweep_summary.json."""
    data = json.load(open(path))
    items = data.get("windows") or data.get("rows") or []
    out = {}
    for w in items:
        d = w.get("delay_max", w.get("max_delay_steps"))
        if d is None:
            continue
        out[d] = _row(
            w.get("successes", 0),
            w.get("total", 0),
            w.get("actual_delay_mean"),
            w.get("replans"),
            w.get("delay_histogram"),
        )
    return out


def parse_task_result_jsons(run_root):
    """Aggregate per-task result JSONs (FastWAM gpu*_task*_results.json,
    LingBot libero_spatial_N.json) across a run root's delay* subdirs.
    Used when the run's root-level sweep summary hasn't been written yet."""
    out = {}
    for sub in sorted(glob.glob(os.path.join(run_root, "delay*"))):
        if not os.path.isdir(sub):
            continue
        try:
            d = int(os.path.basename(sub).replace("delay", "").replace("_", ""))
        except ValueError:
            continue
        succ = tot = 0
        weighted = []
        hist = defaultdict(int)
        for p in glob.glob(os.path.join(sub, "**", "*.json"), recursive=True):
            if os.path.basename(p) in ("delay_summary.json", "delay_sweep_summary.json"):
                continue
            try:
                j = json.load(open(p))
            except Exception:
                continue
            if not isinstance(j, dict) or "total_episodes" not in j \
                    or ("max_delay_steps" not in j and "delay_max" not in j):
                continue
            succ += j.get("successes", 0)
            tot += j.get("total_episodes", 0)
            if j.get("delay_mean") is not None:
                weighted.append((j["delay_mean"], max(j.get("total_episodes", 1), 1)))
            for k, v in (j.get("delay_histogram") or {}).items():
                try:
                    hist[int(float(k))] += v
                except (ValueError, TypeError):
                    pass
        if tot:
            md = (sum(m * w for m, w in weighted) / sum(w for _, w in weighted)
                  if weighted else None)
            out[d] = _row(succ, tot, md, None, dict(hist))
    return out


def parse_delay_summary_file(path):
    """summarize_delay_results.py output (successes/total at one delay)."""
    data = json.load(open(path))
    return _row(
        data.get("successes", 0),
        data.get("total", 0),
        data.get("actual_delay_mean"),
        data.get("replans"),
        data.get("delay_histogram"),
    )


def _count_starwam_results(path):
    """Count trials recursively in a StarWAM incremental results.json
    (task_results maps task_id -> chunk dicts keyed by trial; each leaf has
    'success'). Returns (successes, total, mean_delay)."""
    try:
        data = json.load(open(path))
    except Exception:
        return None
    tr = data.get("task_results")
    if not isinstance(tr, dict) or not tr:
        return None
    succ = tot = 0
    delays = []
    for v in tr.values():
        if not isinstance(v, dict):
            continue
        # preferred: explicit per-task aggregates (successes/trials fields)
        t = v.get("trials")
        s = v.get("successes")
        if isinstance(t, int) and t > 0:
            tot += t
            succ += s if isinstance(s, int) else 0
        eps = v.get("episodes")
        if isinstance(eps, list):
            for e in eps:
                if isinstance(e, dict) and "success" in e:
                    dm = e.get("actual_delay_mean")
                    if isinstance(dm, (int, float)):
                        delays.append(dm)
    if tot == 0:
        return None
    mean_d = sum(delays) / len(delays) if delays else None
    return _row(succ, tot, mean_d)


def parse_starwam_delay_dir(sub):
    """Prefer delay_summary.json; fall back to (or take max of) the incremental
    results.json that rollout.py writes at task boundaries, so in-flight delays
    report live counts instead of stale summaries from older run dirs."""
    rows = []
    p = os.path.join(sub, "delay_summary.json")
    if os.path.exists(p):
        try:
            rows.append(parse_delay_summary_file(p))
        except Exception:
            pass
    rp = os.path.join(sub, "results.json")
    if os.path.exists(rp):
        r = _count_starwam_results(rp)
        if r is not None:
            rows.append(r)
    if not rows:
        return None
    return max(rows, key=lambda r: r["total"])


def parse_openwam_combined(paths):
    """OpenWAM: delay_summary.json with tasks[] (each entry one delay)."""
    out = {}
    for path in paths:
        try:
            data = json.load(open(path))
        except Exception:
            continue
        agg = defaultdict(lambda: [0, 0, {}])
        for t in data.get("tasks", []):
            d = t.get("max_delay_steps")
            if d is None:
                continue
            a = agg[d]
            a[0] += t.get("successes", 0)
            a[1] += t.get("total", 0)
            for k, v in (t.get("delay_histogram") or {}).items():
                a[2][k] = a[2].get(k, 0) + v
        for d, (s, tot, hist) in agg.items():
            n = sum(hist.values())
            mean = sum(int(k) * v for k, v in hist.items()) / n if n else None
            out[d] = _row(s, tot, mean, None, hist)
    return out


def parse_leapbot_results(path):
    """LeapBot-DualLane-V2 results.json for one delay dir."""
    parts = path.split("/")
    delay_dir = next((p for p in parts if p.startswith("delay_")), None)
    if delay_dir is None:
        return {}
    d = int(delay_dir.split("_")[1])
    try:
        data = json.load(open(path))
    except Exception:
        return {}
    return {
        d: _row(
            data.get("total_successes", 0),
            data.get("total_episodes", 0),
            data.get("delay_mean_steps"),
            None,
            data.get("delay_histogram"),
        )
    }


def parse_leapbotce_sweep(path):
    """LeapBotCE delay_sweep_summary.json (scripts/summarize_leapbotce_delays.py).

    One sweep root covers all four LIBERO suites, so split the aggregated
    windows[] per task_suite. Returns {suite_key: {delay: row}} where
    suite_key matches the report's SUITES tokens ("spatial", "object", ...).
    """
    try:
        data = json.load(open(path))
    except Exception:
        return {}
    by_suite = defaultdict(lambda: defaultdict(dict))
    for w in data.get("windows", []):
        d = w.get("delay_max")
        if d is None:
            continue
        # a[4] 累加每个任务自己的 replans（=该任务 delay_samples 长度）。
        # 不能用窗口级 w["replans"]（全四套件总数）除以单套件 total —— 会虚高
        # ~4×（10-11 实测 127/ep vs 任务文件实际 ~26/ep）。
        agg = defaultdict(lambda: [0, 0, defaultdict(int), [], 0])
        for t in w.get("tasks", []):
            suite = str(t.get("task_suite", "")).removeprefix("libero_")
            if suite not in SUITES:
                continue
            a = agg[suite]
            a[0] += int(t.get("successes", 0))
            a[1] += int(t.get("total", 0))
            for k, v in (t.get("delay_histogram") or {}).items():
                a[2][int(float(k))] += int(v)
            m = t.get("actual_delay_mean")
            if isinstance(m, (int, float)):
                a[3].append((m, max(int(t.get("total", 0)), 1)))
            r = t.get("replans")
            if isinstance(r, (int, float)):
                a[4] += int(r)
        for suite, (s, tot, hist, means, replans) in agg.items():
            md = (sum(m * wgt for m, wgt in means) / sum(wgt for _, wgt in means)
                  if means else None)
            by_suite[suite][d] = _row(s, tot, md, replans or None, dict(hist))
    return by_suite


def collect():
    """Returns list of (baseline_key, run_label, {delay: row}, path)."""
    runs = []

    # FastWAM + LingBot + Kairos + XPOLICY: delay_sweep_summary.json per run root
    for base, label in (("fastwam", "FastWAM"), ("lingbot-va", "LingBot-VA"),
                        ("kairos", "Kairos"), ("xpolicy", "XPOLICY")):
        for path in sorted(glob.glob(f"{EVAL_ROOT}/{base}/**/delay_sweep_summary.json",
                                     recursive=True)):
            rel = os.path.relpath(path, f"{EVAL_ROOT}/{base}")
            run = rel.split("/")[0] + (" (+" + "/".join(rel.split("/")[1:-1]) + ")"
                                       if rel.count("/") > 1 else "")
            parsed = parse_sweep_summary(path)
            if parsed:
                runs.append((label, run, parsed, os.path.dirname(path)))

    # FastWAM fallback: per-task results when the root sweep summary is absent
    fw_summary_roots = {os.path.dirname(p) for p in
                        glob.glob(f"{EVAL_ROOT}/fastwam/**/delay_sweep_summary.json",
                                  recursive=True)}
    for run_dir in sorted(glob.glob(f"{EVAL_ROOT}/fastwam/libero_*/*/")):
        run_dir = run_dir.rstrip("/")
        if run_dir in fw_summary_roots:
            continue
        parsed = parse_task_result_jsons(run_dir)
        if parsed:
            runs.append(("FastWAM", os.path.basename(run_dir), parsed, run_dir))

    # OpenWAM: combined delay_summary.json grouped per run root (the file may
    # sit at the root or inside a delay_X dir when one delay is run at a time).
    # Wave-4 object runs resolve OPENWAM_EVAL_ROOT against their launch cwd
    # (code/OpenWAM — a symlink into RoboTwin/XPolicyLab/...), so those run
    # roots live under the code tree while artifacts also mirror to EVAL_ROOT;
    # scan both locations.
    OW_GLOB_ROOTS = [f"{EVAL_ROOT}/openwam",
                     os.path.join(CODE_ROOT, "OpenWAM", "openwam")]
    by_root = {}
    for ow_root in OW_GLOB_ROOTS:
        for p in sorted(glob.glob(f"{ow_root}/**/delay_summary.json",
                                  recursive=True)):
            rel = os.path.relpath(p, ow_root)
            parts = rel.split("/")
            if len(parts) >= 2 and parts[-2].startswith("delay"):
                parts = parts[:-1]
            root = "/".join(parts[:-1]) if len(parts) > 1 else "."
            by_root.setdefault(root, []).append(p)
    for root, paths in sorted(by_root.items()):
        parsed = parse_openwam_combined(paths)
        if parsed:
            label = root if root != "." else "libero"
            runs.append(("OpenWAM (dual-system)", label, parsed,
                         os.path.dirname(paths[0])))

    # StarWAM MoT / SharedDiT: per-delay dirs with delay_summary.json
    for base, label in (("starwam", "StarWAM-MoT"),
                        ("starwam-sharedit", "StarWAM-SharedDiT")):
        for run_dir in sorted(glob.glob(f"{EVAL_ROOT}/{base}/*/")):
            by_delay = {}
            for sub in sorted(glob.glob(os.path.join(run_dir, "delay*"))):
                name = os.path.basename(sub)
                try:
                    d = int(name.replace("delay", "").replace("_", ""))
                except ValueError:
                    continue
                try:
                    row = parse_starwam_delay_dir(sub)
                except Exception:
                    row = None
                if row is not None:
                    by_delay[d] = row
            if by_delay:
                runs.append((label, os.path.basename(run_dir.rstrip("/")), by_delay,
                             run_dir.rstrip("/")))

    # OpenWAM tier-3: per-delay dirs with summary.json (task_results[] each
    # carrying successes/trials). Written by the scheduler per completed delay.
    for ow_root in OW_GLOB_ROOTS:
        for path in sorted(glob.glob(f"{ow_root}/*/delay_*/summary.json")):
            name = os.path.basename(os.path.dirname(path))
            try:
                d = int(name.replace("delay", "").replace("_", ""))
            except ValueError:
                continue
            try:
                data = json.load(open(path))
            except Exception:
                continue
            succ = tot = 0
            for t in data.get("task_results", []):
                if not isinstance(t, dict):
                    continue
                tot += t.get("trials", 0) or 0
                succ += t.get("successes", 0) or 0
            if tot == 0:
                continue
            runs.append(("OpenWAM (dual-system)",
                         os.path.basename(os.path.dirname(os.path.dirname(path))),
                         {d: _row(succ, tot)}, os.path.dirname(path)))

    # OpenWAM in-flight fallback: delay dirs without summary.json yet —
    # aggregate the per-task results.json files the scheduler saves per chunk.
    # Object-protocol results.json carry no trials/num_trials field; derive n
    # from the task's trials_AAA_BBB path segment (fallback: successes/rate).
    for ow_root in OW_GLOB_ROOTS:
        for delay_dir in sorted(glob.glob(f"{ow_root}/*/delay_*/")):
            if os.path.exists(os.path.join(delay_dir, "summary.json")):
                continue
            succ = tot = 0
            d = None
            for tp in glob.glob(os.path.join(
                    delay_dir, "videos", "*", "task_*", "trials_*", "attempt_*",
                    "results.json")):
                try:
                    t = json.load(open(tp))
                except Exception:
                    continue
                trs = t.get("trials")
                n = len(trs) if isinstance(trs, list) else (t.get("num_trials") or 0)
                if not n:
                    seg = [x for x in tp.split("/") if x.startswith("trials_")]
                    if seg and seg[-1].count("_") >= 2:
                        lo, hi = seg[-1].split("_")[1:3]
                        n = int(hi) - int(lo) + 1
                    elif t.get("success_rate"):
                        n = round(t.get("successes", 0) / t["success_rate"])
                if not n:
                    continue
                tot += n
                succ += t.get("successes", 0) or 0
                if d is None:
                    d = t.get("max_delay_steps")
            if tot and d is not None:
                runs.append(("OpenWAM (dual-system)",
                             os.path.basename(os.path.dirname(delay_dir.rstrip("/"))),
                             {d: _row(succ, tot)}, delay_dir))

    # LingBot-VA per-delay delay_summary.json when the root sweep file is absent
    have_lingbot_roots = {os.path.dirname(os.path.dirname(p)) for p in
                          glob.glob(f"{EVAL_ROOT}/lingbot-va/**/delay_sweep_summary.json",
                                    recursive=True)}
    for path in sorted(glob.glob(f"{EVAL_ROOT}/lingbot-va/*/delay*/delay_summary.json")):
        run_dir = os.path.dirname(os.path.dirname(path))
        if run_dir in have_lingbot_roots:
            continue
        name = os.path.basename(os.path.dirname(path))
        try:
            d = int(name.replace("delay", "").replace("_", ""))
        except ValueError:
            continue
        try:
            row = parse_delay_summary_file(path)
        except Exception:
            continue
        runs.append(("LingBot-VA", os.path.basename(run_dir), {d: row}, run_dir))

    # LingBot-VA fallback: per-task results while a stream is still running
    lb_summary_roots = {os.path.dirname(p) for p in
                        glob.glob(f"{EVAL_ROOT}/lingbot-va/**/delay_sweep_summary.json",
                                  recursive=True)}
    for run_dir in sorted(glob.glob(f"{EVAL_ROOT}/lingbot-va/*/")):
        run_dir = run_dir.rstrip("/")
        if run_dir in lb_summary_roots or "server_viz" in run_dir:
            continue
        parsed = parse_task_result_jsons(run_dir)
        if parsed:
            runs.append(("LingBot-VA", os.path.basename(run_dir), parsed, run_dir))

    # LeapBot-DualLane-V2: results.json under delay_ dirs
    lb = defaultdict(dict)
    for path in sorted(glob.glob(f"{EVAL_ROOT}/leapbot-dual-lane-v2*/**/results.json",
                                 recursive=True)):
        if os.path.basename(path) != "results.json":
            continue
        rel = os.path.relpath(path, EVAL_ROOT)
        root_name = rel.split("/")[0]
        parsed = parse_leapbot_results(path)
        lb.setdefault(root_name, {}).update(parsed)
    for root_name, parsed in lb.items():
        if parsed:
            runs.append(("LeapBot-DualLane-V2", root_name.replace("leapbot-dual-lane-v2", "lb-dual"),
                         parsed, f"{EVAL_ROOT}/{root_name}"))

    # LeapBotCE: one delay_sweep_summary.json per model root (all four suites
    # inside), produced by scripts/summarize_leapbotce_delays.py. Emit one
    # entry per suite with a synthetic suite-tokened path so suite_of()
    # routes it to the right table; consolidate() merges them back to the
    # common sweep root for the appendix.
    for path in sorted(glob.glob(f"{EVAL_ROOT}/leapbotce/**/delay_sweep_summary.json",
                                 recursive=True)):
        if "_archive" in path:
            continue
        rel = os.path.relpath(path, f"{EVAL_ROOT}/leapbotce")
        # 注意：dtrain20_seed42_110k / _200k / _h200 等训练轮子目录必须放在
        # "dtrain20" 之前判断（子串包含关系）。
        if "dtrain20_seed42_110k" in rel:
            method = "LeapBotCE (d_train=20, 110k)"
        elif "dtrain20_seed42_200k" in rel:
            method = "LeapBotCE (d_train=20, 200k)"
        elif "dtrain20_seed42_h200" in rel:
            method = "LeapBotCE (d_train=20, h200 25k)"
        elif "dtrain20" in rel:
            method = "LeapBotCE (d_train=20)"
        elif "dtrain40" in rel:
            method = "LeapBotCE (d_train=40)"
        else:
            continue
        run_label = rel.split("/")[0]
        by_suite = parse_leapbotce_sweep(path)
        for suite, parsed in by_suite.items():
            if parsed:
                runs.append((method, run_label, parsed, f"{path}/libero_{suite}"))

    # Kairos: LIBERO-Plus category protocol (Camera Viewpoints 60-task subset,
    # 1 trial/task), one libero_plus_summary.json per completed delay run.
    # suite_of() tags these "kairos-plus" so they render appendix-only.
    kairos_rows = {}
    for path in sorted(glob.glob(
            f"{EVAL_ROOT}/kairos/**/by_category_run*/libero_plus_summary.json",
            recursive=True)):
        m = next((p for p in path.split("/") if p.startswith("delay_")), None)
        if m is None:
            continue
        try:
            d = int(m.split("_")[1])
        except ValueError:
            continue
        try:
            g = json.load(open(path)).get("grand_overall", {})
        except Exception:
            continue
        if not g.get("trials"):
            continue
        # several run generations may exist per delay; keep the largest
        if d not in kairos_rows or g["trials"] > kairos_rows[d]["total"]:
            kairos_rows[d] = _row(g.get("successes", 0), g["trials"], None, None, {})
    if kairos_rows:
        runs.append(("Kairos", "libero-plus-camera-viewpoints-60t", kairos_rows,
                     f"{EVAL_ROOT}/kairos/libero_plus"))

    # Archived smoke/pilot roots stay on disk but must never reach the report.
    runs = [r for r in runs if "_archive" not in str(r[3])]
    return runs


def consolidate(runs):
    """Merge duplicate run entries into one per (method, run-root, suite).

    Several parsers shard one physical run into per-delay entries (OpenWAM
    tier-3 summary.json / in-flight fallback, LingBot fallbacks). Without
    this the appendix renders the same run once per delay. Rows merge
    element-wise keeping the max-total entry per delay; the displayed path
    becomes the common parent of the shard paths."""
    groups = defaultdict(list)  # (base, run, suite) -> [(parsed, path)]
    for base, run, parsed, path in runs:
        groups[(base, run, suite_of(path))].append((parsed, path))
    out = []
    for (base, run, _suite), items in groups.items():
        merged = {}
        for parsed, _ in items:
            for d, row in parsed.items():
                cur = merged.get(d)
                if cur is None or row["total"] > cur["total"]:
                    merged[d] = row
        paths = [p for _, p in items]
        try:
            root = os.path.commonpath(paths)
        except ValueError:
            root = paths[0]
        out.append((base, run, merged, root))
    return out


# --------------------------------------------------------------------------
# CloudEdgeVLA-style main table
# --------------------------------------------------------------------------
def merge_per_delay(run_list):
    """One cell per delay: keep the entry with the most episodes."""
    merged = {}
    for run_label, parsed, path in run_list:
        for d, row in parsed.items():
            cur = merged.get(d)
            if cur is None or row["total"] > cur[0]["total"]:
                merged[d] = (row, run_label, path)
    return merged


def draws_per_ep(row, path=None, delay=40):
    """Measured delayed-observation draws per episode for one merged row.

    延迟剂量 = 每回合策略消费延迟观测（并据其提交动作）的次数。优先级：
    sweep 汇总的 ``replans`` 总数 / 回合数 → 直方图质量 / 回合数 →
    （OpenWAM 布局）videos/**/results.json 的 delay_samples 总数 / trial 数
    （按文件头 max_delay_steps 过滤到目标延迟窗）。返回 None 表示无 audit。
    """
    total = row["total"]
    if not total:
        return None
    if row["replans"] is not None:
        return row["replans"] / total
    hist = row["hist"]
    if hist:
        return sum(hist.values()) / total
    if path:
        draws = trials = 0
        # OpenWAM 布局: 汇总行路径可能是 run 根（videos 在 delay_X/videos 下）
        # 或直接是 delay_X 目录（videos 紧邻其下）—— 两层都扫；不过滤延迟窗
        # 会把 5 个窗口混算（10-11 实测 579→352），以文件头 max_delay_steps 为准。
        patterns = [os.path.join(path, "videos", "**", "results.json"),
                    os.path.join(path, "delay_*", "videos", "**", "results.json")]
        seen = set()
        for pat in patterns:
            for rp in glob.glob(pat, recursive=True):
                if rp in seen:
                    continue
                seen.add(rp)
                try:
                    j = json.load(open(rp))
                except Exception:
                    continue
                samples = j.get("delay_samples")
                if not isinstance(samples, list) or not samples:
                    continue
                if int(j.get("max_delay_steps") or -1) != int(delay):
                    continue
                draws += len(samples)
                trs = j.get("trials")
                trials += len(trs) if isinstance(trs, list) else int(trs or 0)
        if trials:
            return draws / trials
    return None


def make_figure(curves):
    """curves: {display_name: {delay: rate_percent}} -> PNG or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return _make_figure_subprocess(curves)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, (name, pts) in enumerate(curves.items()):
        xs = sorted(d for d in pts if pts[d] is not None)
        ys = [pts[d] for d in xs]
        if not xs:
            continue
        ax.plot(xs, ys, marker=markers[i % len(markers)], label=name, linewidth=1.8)
    ax.set_xlabel("Max frame delay (replan steps)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Success rate vs. frame delay (LIBERO-spatial)")
    ax.set_xticks(DELAY_KEYS)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    try:
        fig.savefig(FIG_PATH, dpi=130)
        return True
    except Exception:
        return False
    finally:
        plt.close(fig)


# base python3 may lack matplotlib; retry with a conda env interpreter
_FIG_FALLBACK_PY = "/dahuafs/userdata/2653853/code/.envs/leapbot-va/bin/python"


def _make_figure_subprocess(curves):
    if not os.path.exists(_FIG_FALLBACK_PY):
        return False
    import tempfile
    payload = {"delay_keys": DELAY_KEYS, "fig_path": FIG_PATH,
               "curves": {k: {str(d): v for d, v in pts.items()} for k, pts in curves.items()}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        payload_path = f.name
    script = (
        "import json,sys\n"
        "p=json.load(open(sys.argv[1]))\n"
        "import matplotlib; matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "fig,ax=plt.subplots(figsize=(7.2,4.2))\n"
        "mk=['o','s','^','D','v','P','X','*']\n"
        "for i,(name,pts) in enumerate(p['curves'].items()):\n"
        "    xs=sorted(int(d) for d in pts if pts[d] is not None)\n"
        "    ys=[pts[str(d)] for d in xs]\n"
        "    if xs: ax.plot(xs,ys,marker=mk[i%len(mk)],label=name,linewidth=1.8)\n"
        "ax.set_xlabel('Max frame delay (replan steps)')\n"
        "ax.set_ylabel('Success rate (%)')\n"
        "ax.set_title('Success rate vs. frame delay (LIBERO-spatial)')\n"
        "ax.set_xticks(p['delay_keys']); ax.set_ylim(-5,105)\n"
        "ax.grid(True,alpha=0.3); ax.legend(fontsize=8,loc='best')\n"
        "fig.tight_layout(); fig.savefig(p['fig_path'],dpi=130)\n"
    )
    try:
        r = subprocess.run([_FIG_FALLBACK_PY, "-c", script, payload_path],
                           capture_output=True, timeout=60)
        os.unlink(payload_path)
        return r.returncode == 0 and os.path.exists(FIG_PATH)
    except Exception:
        return False


def render(runs):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    ap = lines.append
    ap("# Baseline 帧延迟鲁棒性评测 — 实时结果")
    ap("")
    ap(f"> 自动生成于 **{now}** · 生成器 `FastWAM/docs/generate_baseline_report.py`")
    ap("> 刷新: `python FastWAM/docs/generate_baseline_report.py`（可随时重复执行，新结果目录自动纳入）")
    ap("")
    ap("**协议**（对齐 CloudEdgeVLA 的 frame-delay 设定）：单流全观测帧延迟。每次 replan 从 `seed=1234` 的 RNG 采样 d∈[0, D]，整组观测同时老化；扫描 D ∈ {0, 5, 10, 20, 40} 步。")
    ap("")
    ap("**标准协议（对齐各基线官方论文/仓库配置）**：LIBERO-spatial 全部 10 任务 × **50 trials/任务**（=每延迟 500 回合，与各仓库官方默认一致：FastWAM `sim_libero.yaml num_trials:50`、StarWAM launch 默认 50、LingBot client `--test-num 50`、OpenWAM scheduler canonical 50）。在此之上叠加帧延迟扫描。表格单元格若标注 (n=k) 为进行中/pilot 数据（k<500），**加粗 = 论文标准规模（n≥500/延迟）**。")
    ap("")
    ap("### 官方配置对照")
    ap("")
    ap("| 基线 | 官方 trials/任务 | 本评测推理配置 | 备注 |")
    ap("|---|---|---|---|")
    ap("| FastWAM | 50 | num_inference_steps=10, replan=10 | 官方 release ckpt (libero_uncond_2cam224) |")
    ap("| StarWAM-MoT | 50 | inference=8, action=8, replan=10, wait=10 | 官方 recipe starwam_libero_mot_wan22_5b |")
    ap("| StarWAM-SharedDiT | 50 | inference=16, action=16, replan=10 | 官方 recipe shared_dit_wan22_5b |")
    ap("| LingBot-VA | 50 | 官方 libero server 配置 | server+client 官方流程；延迟按其原生 chunk 抽样（~46 次/ep） |")
    ap("| OpenWAM (dual-system) | 50 | eef native action, seed42 settle30 协议 | scheduler canonical；延迟每控制步抽样（579 次/ep @D=40） |")
    ap("| Kairos † | LIBERO-Plus 类目协议 | external WAM adapter (DiT 4B+Qwen2.5-VL-7B) | Camera Viewpoints 60 任务子集×1 trial（全类目 1599） |")
    ap("| LeapBot-DualLane-V2 (在研) | 50 | inference=20, replan=10, wait=30 | dual-lane 双流在研模型 |")
    ap("| LeapBotCE (d_train=20) | 50 | inference=10, replan=10, wait=30 | 本仓库训练，全 LIBERO 四套件 50k 步，seed=1234 |")
    ap("| LeapBotCE (d_train=20, 110k) | 50 | inference=10, replan=10, wait=30 | 本仓库训练，d_train=20 加长轮在 110k 步（200k 计划中途，NCCL 崩溃前最后检查点）的评测，seed=1234 |")
    ap("| LeapBotCE (d_train=20, 200k) | 50 | inference=10, replan=10, wait=30 | 本仓库训练，d_train=20 加长轮完整 200k 步（若后续续训完成） |")
    ap("| LeapBotCE (d_train=40) | 50 | inference=10, replan=10, wait=30 | 同上，训练时延上限 40 |")
    ap("")

    # group by baseline key
    by_base = defaultdict(list)
    for base, run, parsed, path in runs:
        by_base[base].append((run, parsed, path))

    # ---------------- main table ----------------
    ap("## 主表：各方法成功率 (%) 对比 — 每个方法一行")
    ap("")
    ap("**LIBERO-spatial，按帧延迟扫描。单元格 = 成功率 (%)；每格数据取该延迟下回合数最多的运行（n 标注见下）**")
    ap("")
    ap("| 方法 | 延迟 0 | 延迟 5 | 延迟 10 | 延迟 20 | 延迟 40 | replans/ep (D=40) |")
    ap("|---|---|---|---|---|---|---|")

    def render_suite(suite, keep_empty):
        """Emit table rows for one task suite; returns (curves, merge_notes).
        keep_empty=True keeps a '—' row for methods without data (main table);
        otherwise methods with no data in this suite are skipped (extended
        suites only show methods that have started)."""
        suite_curves, notes = {}, []
        for display, keys in METHODS:
            run_list = [x for k in keys for x in by_base.get(k, [])
                        if suite_of(x[2]) == suite]
            if not run_list:
                if keep_empty:
                    ap(f"| {display} | — | — | — | — | — | — |")
                    notes.append((display, "（暂无结果）"))
                continue
            merged = merge_per_delay(run_list)
            cells = []
            pts = {}
            for d in DELAY_KEYS:
                if d in merged:
                    row, src, _ = merged[d]
                    rate = 100.0 * row["successes"] / row["total"] if row["total"] else 0.0
                    n = row["total"]
                    # 加粗 = 论文标准规模 n≥500 (10 任务×50 trials)；10≤n<500 视为进行中/pilot，
                    # 显式标注 (n=k) 以免与完成态混淆。
                    if n >= 500:
                        cell = f"**{rate:.1f}**"
                    else:
                        cell = f"{rate:.1f}(n={n})"
                    cells.append(cell)
                    pts[d] = rate
                else:
                    cells.append("—")
            # 延迟剂量列：D=40 下 audit 实测的每回合延迟抽样次数
            r40 = merged.get(40)
            rpe = draws_per_ep(r40[0], r40[2], delay=40) if r40 else None
            cells.append(f"{rpe:.1f}" if rpe is not None else "—")
            ap(f"| {display} | " + " | ".join(cells) + " |")
            suite_curves[display] = pts
            # footnote: per-delay (n, source)
            nmap = {d: (merged[d][0]["total"], merged[d][1]) for d in merged}
            srcs = sorted({src for _, src, _ in merged.values()})
            notes.append((display, f"n 按延迟取最多回合的来源：{nmap}；来源运行：{', '.join(srcs)}"))
        return suite_curves, notes

    curves, merge_notes = render_suite("spatial", keep_empty=True)
    ap("")
    ap("加粗 = 论文标准规模 (n=500/延迟，即 10 任务×50 trials)。`(n=k)` = 该延迟只有 k 回合（pilot/链路验证）。`—` = 该延迟尚未有结果。")
    ap("**replans/ep (D=40)** = audit 实测的每回合延迟抽样次数（策略消费一次延迟观测并提交动作记一次；来源：各仓库 results.json / audit jsonl 的 `replans` 或 `delay_samples` 计数，D=40 窗口）。`—` = 无 audit 数据。含义与跨环境警示见下方「延迟剂量披露」。")
    ap("† Kairos 在其原生基准 LIBERO-Plus (Camera Viewpoints 类目) 上评测，与其他方法的 libero_spatial 不完全可比。")
    ap("")
    ap("### 帧延迟协议实现备注（CloudEdgeVLA 式单流全观测老化）")
    ap("")
    ap("**核心保证：所有方法的延迟均作用于完整观测流（图像 + 本体状态 + 全部模态一起老化），模拟云端部署时整包观测的网络老化；不存在只延迟图像的实现。** 每次 replan 采样一个延迟 d，取 d 步之前的完整观测 dict 交给策略。逐方法核查（代码路径相对各自仓库根目录）：")
    ap("")
    ap("| 方法 | 延迟实现 | 机制 | 采样区间 | 抽样节奏 |")
    ap("|---|---|---|---|---|")
    ap("| FastWAM | `experiments/libero/eval_libero_delayed.py` + `experiments/libero/cloudedge_delay.py` | `history.append(obs)` 存 env 完整观测 dict → `delayed_obs` 整体传 `_predict_action_chunk`；results 标注 `delay_protocol: single_stream_full_observation` | U{1..D} | 每 10 控制步（chunk=10） |")
    ap("| StarWAM-MoT/SharedDiT | `examples/libero/rollout.py`（`history.append(obs)` → `_predict_action_chunk(obs=delayed_obs)`） | env 原始 obs 全量入 deque | U{1..D} | 每 10 控制步（chunk=10） |")
    ap("| OpenWAM (dual-system) | `benchmarks/libero/single_eval.py`（`_snapshot_observation` deep-copy 全 dict → `policy.act(delayed_obs)`，`send_state=True`） | 图像+状态全模态在延迟载荷内 | U{0..D} | **每控制步**（`policy.act` 每 step 采样一次；D=40 实测 579 次/ep） |")
    ap("| LeapBot-DualLane-V2 | `experiments/liberoplus/eval_liberoplus_dual_lane_fixed.py`（env wrapper 每次 step `copy.deepcopy(obs)` 入史，replan 时整体替换） | docstring：\"Delay the whole synchronized observation at every model replan\" | U{0..D} | 每 10 控制步（replan=10） |")
    ap("| LingBot-VA | `evaluation/libero/client_delay.py`（`deque([obs])` → `model.infer(dict(obs=delayed_obs, prompt))`） | 完整观测上送 | U{1..D} | 每原生 chunk（一次 infer 执行整块动作，实测 ~46 次/ep） |")
    ap("| Kairos | `benchmarks/libero_plus/kairos_wam/experiments/libero/eval_libero_single.py`（同 OpenWAM snapshot 模式 → `_predict_action_chunk(obs=model_obs)`） | 全 dict 深拷贝入史 | U{0..D} | 同 OpenWAM snapshot 模式（协议复核中） |")
    ap("| LeapBotCE | `experiments/libero/eval_libero_cloudedge.py` + `experiments/libero/cloudedge_delay.py` | **云端/边缘分流协议（方法原生设计）**：仅云端流老化——cloud 相机拼接图 + cloud 本体状态取自同一 t-d 历史条目，经冻结 Wan 编码为规划 KV；边缘当前双视角（SigLIP）与当前本体状态始终新鲜，仅在 ActionDiT 注入 | U{1..min(D,h)} | 每 10 控制步（replan=10） |")
    ap("")
    ap("**观测契约披露（重要）**：与上表单流全观测老化的基线不同，LeapBotCE 的延迟只作用于云端规划观测（图像+本体状态同源老化），边缘侧当前视角/本体状态保持新鲜——这是该方法的云边分工设计本身，评测如实按其原生协议执行。对比两条曲线时需注明：LeapBotCE 在延迟 D 下仍持有 D 步内新鲜的边缘观测，而基线的全观测已老化 D 步；因此 LeapBotCE 的保持率优势来自\"新鲜边缘补偿\"，论文引用时必须声明该契约差异（LEAPBOTCE.md 基线协议第 5 条）。")
    ap("")
    ap("**采样区间披露**：OpenWAM / LeapBot / Kairos 为 U{0..D}（采到 0 = 本次 replan 用即时观测）；FastWAM / StarWAM / LingBot / LeapBotCE 为 U{1..D}（每次 replan 至少老化 1 步；LeapBotCE 的上限为 min(D, 已有历史)，回合初期历史不足时回退当前帧）。D≥5 时两种区间均值差 <0.5 步，不影响方向性结论；论文按上表如实标注即可。公共参数：`delay_seed=1234`（LeapBotCE 以 cfg.seed=1234 为延迟/动作 RNG 基），D ∈ {0,5,10,20,40}。**抽样节奏按方法不同**（见上表「抽样节奏」列）：chunk=10 系统每 10 控制步采样一次，OpenWAM 每控制步采样一次，LingBot 按其原生 chunk——跨方法比较延迟剂量时必须把节奏一并披露。")
    ap("**审计链**：每个 trial 的 realized delay（`delay_samples`）、`delay_mean` 与 `delay_histogram` 均落盘于各自 results.json，可直接引用作论文协议证明。")
    ap("")
    ap("**延迟剂量披露（replans/ep 列）**：延迟观测的毒害剂量 ≈ 抽样次数 × 实际老化深度，`replans/ep (D=40)` 列给出前者（audit 实测）。LIBERO 各 chunk=10 系统抽样节奏与回合上限相同（每 10 控制步一抽、回台上限 400 步 → 上限 40 次/ep）；D=40 实测 FastWAM / StarWAM×2 为 39.8-39.9（几乎全失败 → 回合跑满）而 LeapBotCE 为 25.9-26.1（成功率过半 → 成功回合提前收尾）——同节奏下这一剂量差异是成功率的**结果**而非原因，解读时方向别搞反。OpenWAM 是真正的例外：其 `policy.act` 每控制步抽样一次（D=40 实测 ~581 次/ep，剂量最高）却最鲁棒——双系统快环在慢规划之间持续提供新鲜执行信号，是架构补偿而非低剂量；说明剂量是必要披露维度但非唯一变量。跨环境对比时节奏与剂量必须一并披露：RoboTwin 泳道 Fast-WAM 官方默认 24 步 replan（实测 16.5 次/ep，仅为 LIBERO 泳道节奏的四成）、AHA-WAM 逐步 replan（实测 ~450 次/ep，27×），详见 `baseline_results_robotwin.md` 的「Fast-WAM 曲线为何远好于 LIBERO 泳道」归因小节。")
    ap("")

    for display, note in merge_notes:
        if note != "（暂无结果）":
            ap(f"- **{display}**: {note}")
    ap("")

    # ---------------- extended suites (object / goal / 10) ----------------
    _suite_titles = {"object": "LIBERO-object", "goal": "LIBERO-goal", "10": "LIBERO-10"}
    for suite in SUITES[1:]:
        has_data = any(suite_of(x[2]) == suite
                       for xs in by_base.values() for x in xs)
        if not has_data:
            continue
        ap("")
        ap(f"## {_suite_titles[suite]}（帧延迟扫描，同协议：delay_seed=1234，replan=10，D∈{{0,5,10,20,40}}）")
        ap("")
        ap("| 方法 | 延迟 0 | 延迟 5 | 延迟 10 | 延迟 20 | 延迟 40 | replans/ep (D=40) |")
        ap("|---|---|---|---|---|---|---|")
        _, suite_notes = render_suite(suite, keep_empty=False)
        ap("")
        ap("加粗 = 论文标准规模 (n=500/延迟，即 10 任务×50 trials)；`(n=k)` = 该延迟只有 k 回合（进行中/pilot）。")
        ap("replans/ep (D=40) = audit 实测的每回合延迟抽样次数（延迟观测消费次数），含义见主表下方披露；`—` = 该模型 D=40 无 audit 数据。")
        for display, note in suite_notes:
            ap(f"- **{display}**: {note}")
    ap("")

    # ---------------- figure ----------------
    if curves and any(any(v is not None for v in pts.values()) for pts in curves.values()):
        if make_figure(curves):
            ap("## 对比曲线")
            ap("")
            ap("![成功率-帧延迟曲线](baseline_delay_curve_libero.png)")
            ap("")

    # ---------------- in progress / blocked ----------------
    ap("## 进行中 / 待办 / 阻塞")
    ap("")
    ap("**Tier-3 论文标准（10 任务 × 50 trials × 5 delay = 每格 500 回合）**，4 个 LIBERO 子任务套件 × 各方法：")
    ap("- **LIBERO-spatial**：FastWAM ✅ / StarWAM-MoT ✅ / StarWAM-SharedDiT ✅ / OpenWAM 仅剩 d40（在跑）/ LingBot·LeapBot 为 pilot 规模（n=10/7，见主表标注）/ Kairos 走 LIBERO-Plus 类目协议（已完成 60 任务子集 × 5 delay）。")
    ap("- **LIBERO-object**：FastWAM d0/d5/d40 ✅，d10/d20 在跑；StarWAM-MoT 5 delay + SharedDiT d0 + OpenWAM 5 delay 全部在跑（2026-09-20 起密集共卡调度）。")
    ap("- **LIBERO-goal**：FastWAM 5 delay 全部在跑。")
    ap("- **LIBERO-10**：FastWAM 5 delay 在跑；StarWAM / OpenWAM 排队接力。")
    ap("- **Kairos（⚠️ 协议复核）**：5 delay 均 60/60 = 100%（LIBERO-Plus Camera Viewpoints，n=60/延迟，1 trial/任务）。全延迟零衰减与其他方法的延迟曲线矛盾，且 task_results 无实际延迟统计字段 —— 延迟协议对 Kairos 评测链路的实际生效性待复核，复核通过前不入对比结论。")
    ap("")
    ap("**调度**：GPU 密集共卡（每卡 2-3 个 eval 共享，显存 50-74GB/卡）；已清除全部 smoke/1-trial 旧结果（归档至 `baseline-evaluations/_archive_smoke_20260920/`）。")
    ap("")
    ap("**阻塞项**：")
    ap("- **RoboTwin 系（Motus / StarWAM-RoboTwin / X-WAM / DreamZero / AHA-WAM）**：资产已就位，等用户侧容器重建（NVIDIA_DRIVER_CAPABILITIES 加 graphics）后开始 Motus smoke。")
    ap("- **DreamZero**：是否纳入 RoboTwin 评测待定。")
    ap("")

    # ---------------- appendix ----------------
    ap("## 附录：各运行明细")
    order_keys = {k: i for i, (_, keys) in enumerate(METHODS) for k in keys}
    baselines = sorted(by_base, key=lambda b: (order_keys.get(b, 99), b))
    for base in baselines:
        ap("")
        ap(f"### {base}")
        for run, parsed, path in by_base[base]:
            ap("")
            ap(f"**{run}** — `{path}`")
            ap("")
            ap("| delay | 成功/回合 | 成功率 | 实际延迟均值 | replans | 直方图(前几档) |")
            ap("|---|---|---|---|---|---|")
            for d in DELAY_KEYS:
                row = parsed.get(d)
                if row is None:
                    ap(f"| {d} | — | — | — | — | — |")
                    continue
                sr = f"{row['successes']}/{row['total']}"
                rate = f"{100.0 * row['successes'] / row['total']:.1f}%" if row["total"] else "—"
                md = f"{row['mean_delay']:.2f}" if row["mean_delay"] is not None else "—"
                rp = str(row["replans"]) if row["replans"] is not None else "—"
                hist = row["hist"]
                tops = sorted(hist.items(), key=lambda kv: int(kv[0]))[:5]
                hstr = ", ".join(f"{k}:{v}" for k, v in tops)
                if len(hist) > 5:
                    hstr += f" …(+{len(hist)-5}档)"
                ap(f"| {d} | {sr} | {rate} | {md} | {rp} | {hstr} |")

    ap("")
    ap("## GPU / 进程快照（生成时刻）")
    ap("")
    ap("```")
    for line in gpu_snapshot().splitlines():
        ap(line)
    ap("```")

    return "\n".join(lines) + "\n"


def gpu_snapshot():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
        return out.strip()
    except Exception:
        return "(nvidia-smi 不可用)"


def main():
    runs = consolidate(collect())
    md = render(runs)
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(md)
    print(f"wrote {OUT_PATH} ({len(runs)} run-roots)")


if __name__ == "__main__":
    main()
