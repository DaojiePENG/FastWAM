"""Aggregate recorded hardware runs without treating requests as independent trials.

Usage: python analyze_testbed.py results/testbed_20261029
Only complete schema-v2 runs enter model/transport summaries. Smoke and pilot
runs are excluded by filename; all source files and excluded files are retained.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
import statistics


def describe(values):
    values = sorted(values)
    if not values:
        return None
    def q(p):
        x = (len(values) - 1) * p
        i = int(x)
        return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (x - i)
    return {"n": len(values), "mean": statistics.mean(values),
            "p50": q(.5), "p95": q(.95), "min": values[0], "max": values[-1]}


def write_csv(path, rows):
    if rows:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    transport = defaultdict(list)
    iperf = defaultdict(list)
    excluded = []
    signatures = {}
    for path in sorted(args.directory.glob("*.json")):
        if any(word in path.stem for word in ("smoke", "setup_overlap", "pilot", "failed")):
            excluded.append(path.name)
            continue
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            excluded.append(path.name)
            continue
        if data.get("schema_version") == 2 and "throughput" in data and "mode" in data:
            if data.get("status") not in ("ok", "measured_with_failures"):
                excluded.append(path.name)
                continue
            measured = [r for r in data["records"] if not r["warmup"]]
            if len(measured) != data["n"] or len(measured) != data["throughput"]["attempted"]:
                raise ValueError(f"Incomplete formal model batch: {path}")
            key = (data["stage"], data["mode"], data["schedule"])
            signature = json.dumps({k: data.get(k) for k in
                                    ("cloud_url", "edge_url", "input", "seed", "action_horizon",
                                     "executed_steps_per_replan", "num_inference_steps", "http_connection")},
                                   sort_keys=True)
            if key in signatures and signatures[key] != signature:
                raise ValueError(f"Mixed model recipes/inputs in condition {key}: {path}")
            signatures[key] = signature
            groups[key].append((path.name, data))
        elif data.get("schema_version") == 2 and "probes" in data:
            if data.get("status") not in ("ok", "measured_with_failures"):
                excluded.append(path.name)
                continue
            for probe in data["probes"]:
                measured = [r for r in probe["records"] if not r["warmup"]]
                if len(measured) != data["n"] or probe["successful"] + probe["failed"] != data["n"]:
                    raise ValueError(f"Incomplete formal transport batch: {path}")
                key = (data["operation"], probe["synthetic_body_bytes"],
                       probe["endpoint"], data.get("http_connection"))
                transport[key].append((path.name, probe))
        elif path.name.startswith(("iperf_before_", "iperf_after_")) and "end" in data:
            group, direction = path.stem.split("_")[1:3]
            iperf[(group, direction)].append((path.name, data["end"]["sum_received"]["bits_per_second"] / 1e6))
    model_rows, models = [], []
    for (stage, mode, schedule), runs in groups.items():
        measured = [r for _, d in runs for r in d["records"] if not r["warmup"]]
        valid = [r for r in measured if r["ok"]]
        times = {key: describe([r[key] for r in valid if isinstance(r.get(key), (int, float))])
                 for key in sorted({k for r in valid for k in r if k.endswith("_ms")})}
        rates = [d["throughput"]["completed_operations_per_s"] for _, d in runs]
        elapsed = sum(d["throughput"]["measured_elapsed_s"] for _, d in runs)
        item = {"stage": stage, "mode": mode, "schedule": schedule, "runs": len(runs),
                "source_files": [n for n, _ in runs], "attempted": len(measured),
                "successful": len(valid), "failed": len(measured) - len(valid),
                "rate_per_run": rates, "rate_mean": statistics.mean(rates),
                "rate_min": min(rates), "rate_max": max(rates),
                "aggregate_rate": len(valid) / elapsed, "timings_ms": times}
        models.append(item)
        for filename, run in runs:
            t = run["throughput"]
            latency = run["legs"].get("round_trip_ms", {})
            model_rows.append({"file": filename, "stage": stage, "mode": mode, "schedule": schedule,
                               "successful": t["successful"], "failed": t["failed"],
                               "elapsed_s": t["measured_elapsed_s"],
                               "completed_operations_per_s": t["completed_operations_per_s"],
                               **{f"latency_{k}_ms": latency.get(k) for k in ("mean", "p50", "p95")}})
    transport_rows, transports = [], []
    for (operation, size, endpoint, connection), runs in transport.items():
        samples = [r for _, d in runs for r in d["records"] if not r["warmup"]]
        good = [r for r in samples if r["ok"]]
        row = {"operation": operation, "bytes": size, "endpoint": endpoint,
               "connection": connection, "runs": len(runs),
               "n": len(good), "failed": len(samples) - len(good),
               **{k: v for k, v in (describe([r["request_ms"] for r in good]) or {}).items()}}
        transport_rows.append(row)
        transports.append({**row, "source_files": [n for n, _ in runs]})
    iperf_rows = [{"group": g, "direction": d, "source_files": [n for n, _ in runs],
                  "mbit_s": describe([x for _, x in runs])} for (g, d), runs in iperf.items()]
    result = {"replication": "Repeated technical requests on the same hosts; descriptive percentiles only.",
              "models": models, "transport": transports, "iperf": iperf_rows, "excluded_files": excluded}
    (args.directory / "analysis_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(args.directory / "model_runs.csv", model_rows)
    write_csv(args.directory / "http_payload_summary.csv", transport_rows)
    write_csv(args.directory / "iperf_summary.csv", [
        {"group": r["group"], "direction": r["direction"], **(r["mbit_s"] or {})} for r in iperf_rows])
    print(json.dumps({"model_conditions": len(models), "transport_conditions": len(transports),
                      "iperf_groups": len(iperf_rows)}, indent=2))


if __name__ == "__main__":
    main()
