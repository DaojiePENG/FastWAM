"""Bandwidth mini-study: zero-shot payload variants vs. measured deployment cost.

Re-runs the codec bench loop with two inference-time payload knobs applied to
the SAME trained codec (no retraining, no fine-tuning):

  - ``pool_grid``    re-targets the codec's adaptive average pooling so fewer
                     planning tokens cross the wire (trained grid is 7x8=56);
  - ``payload_dtype`` int8 per-token symmetric quantization of z (scales ride
                     in the response header, dequantized on the edge).

For each variant (grid x dtype) the script records the payload size, the
isolated z transfer leg, the edge KV reconstruction leg, the cloud codec
overhead, the edge denoise leg, and the int8 z relative error.  Task
retention (success rate) is NOT part of this script -- that requires the
full LIBERO evaluation protocol, not a loopback replay.

Usage:
    python bw_sweep.py --n 20 --warmup 3 --out results/bw_sweep.json
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import time
from pathlib import Path

import numpy as np
import requests

from bench_client import DEFAULT_PROMPT, load_frames, post, rtt_baseline, stats
from protocol import pack_message, unpack_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", default="http://127.0.0.1:8600")
    parser.add_argument("--edge", default="http://127.0.0.1:8601")
    parser.add_argument("--n", type=int, default=20, help="measured replans per variant")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--grids", default="7x8,5x6,4x4,3x4,2x3",
        help="comma-separated pooling grids (deployed = 7x8)")
    parser.add_argument("--dtypes", default="bf16,int8")
    parser.add_argument(
        "--instruction", default="put the bowl on the stove")
    parser.add_argument("--out", default="")
    parser.add_argument("--tag", default="bw_sweep")
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def run_variant(args, grid: tuple[int, int], dtype: str, prompt: str,
                primary: np.ndarray, wrist: np.ndarray, proprio: np.ndarray,
                rtt_ms: float) -> dict:
    label = f"{grid[0]}x{grid[1]}_{dtype}"
    grid_header = [grid[0], grid[1]] if grid != (7, 8) else None
    print(f"\n[sweep] variant {label} (n={args.n}, warmup={args.warmup})")
    records = []
    payload_bytes_seen = set()
    rel_errs = []
    total = args.warmup + args.n
    for i in range(total):
        replan_id = f"{label}_r{i}"
        obs_body = pack_message(
            {"prompt": prompt, "proprio": proprio.tolist(), "replan_id": replan_id,
             **({"pool_grid": grid_header} if grid_header else {}),
             **({"payload_dtype": dtype} if dtype != "bf16" else {})},
            [primary, wrist],
        )
        t0 = time.perf_counter()
        r1 = post(f"{args.cloud}/encode_cloud", obs_body, args.timeout)
        t1 = time.perf_counter()
        cloud_header, cloud_tensors = unpack_message(r1.content)
        payload_bytes_seen.add(int(cloud_header["payload_bytes"]))
        if cloud_header.get("z_rel_err_pct") is not None:
            rel_errs.append(float(cloud_header["z_rel_err_pct"]))

        t2 = time.perf_counter()
        r2 = post(
            f"{args.edge}/payload",
            pack_message({"session_id": "bench", "replan_id": replan_id,
                          "seq_len": cloud_header["seq_len"],
                          "tokens_per_frame": cloud_header["tokens_per_frame"],
                          **({"payload_dtype": dtype} if dtype != "bf16" else {}),
                          **({"qscale": cloud_header["qscale"]}
                             if dtype == "int8" else {})},
                         cloud_tensors),
            args.timeout,
        )
        t3 = time.perf_counter()

        step_body = pack_message(
            {"session_id": "bench", "proprio": proprio.tolist(),
             "action_horizon": 32, "num_inference_steps": 10,
             "sigma_shift": None, "replan_id": replan_id},
            [primary, wrist],
        )
        t4 = time.perf_counter()
        r3 = post(f"{args.edge}/step", step_body, args.timeout)
        t5 = time.perf_counter()

        step_header, step_tensors = unpack_message(r3.content)
        action = step_tensors[0].numpy()
        cloud_timings = cloud_header["timings"]
        edge_timings = step_header["timings"]
        record = {
            "replan_id": replan_id,
            "warmup": i < args.warmup,
            "cloud_vae_ms": cloud_timings.get("vae", 0.0),
            "cloud_prefill_ms": cloud_timings.get("prefill", 0.0),
            "cloud_codec_ms": cloud_timings.get("codec", 0.0),
            "cloud_compute_ms": float(r1.headers["X-Cloud-Compute-Ms"]),
            "z_transfer_ms": (t3 - t2) * 1e3 - rtt_ms,
            "kv_reinit_ms": float(r2.headers["X-Kv-Reinit-Ms"]),
            "edge_denoise_ms": edge_timings.get("denoise", 0.0),
            "edge_compute_ms": float(r3.headers["X-Edge-Compute-Ms"]),
            "round_trip_ms": (t5 - t0) * 1e3,
            "payload_bytes": int(cloud_header["payload_bytes"]),
            "seq_len": int(cloud_header["seq_len"]),
            "action_finite": bool(np.isfinite(action).all()),
        }
        if not record["action_finite"]:
            raise RuntimeError(f"non-finite actions at {replan_id}: {record}")
        records.append(record)

    measured = [r for r in records if not r["warmup"]]
    leg_names = ("cloud_codec_ms", "z_transfer_ms", "kv_reinit_ms",
                 "edge_denoise_ms", "edge_compute_ms", "round_trip_ms")
    legs = {name: stats([r[name] for r in measured]) for name in leg_names}
    payload_bytes = int(statistics.mode(
        r["payload_bytes"] for r in measured))
    summary = {
        "label": label,
        "grid": list(grid),
        "dtype": dtype,
        "tokens": measured[0]["seq_len"],
        "payload_bytes": payload_bytes,
        "z_rel_err_pct": (float(np.median(rel_errs)) if rel_errs else None),
        "legs": legs,
    }
    print(f"[sweep] {label}: tokens {summary['tokens']}  "
          f"payload {payload_bytes / 1e3:.1f} kB  "
          f"z_err {summary['z_rel_err_pct']}%")
    for name, agg in legs.items():
        print(f"  {name:18s} mean {agg['mean']:9.3f}  p50 {agg['p50']:9.3f}  p95 {agg['p95']:9.3f}")
    return {"summary": summary, "records": records}


def main() -> None:
    args = parse_args()
    prompt = DEFAULT_PROMPT.format(task=args.instruction)
    primary, wrist = load_frames(args)
    rng = np.random.default_rng(1)
    proprio = np.concatenate([
        rng.uniform(-0.1, 0.1, 3), rng.uniform(-0.3, 0.3, 3), np.full(2, 0.02),
    ]).astype(np.float32)

    cloud_health = requests.get(f"{args.cloud}/health", timeout=30).json()
    edge_health = requests.get(f"{args.edge}/health", timeout=30).json()
    print(f"[sweep] cloud: {cloud_health}")
    print(f"[sweep] edge:  {edge_health}")

    # task session (T5 context) once, outside the per-replan legs
    context_resp = post(
        f"{args.cloud}/encode_prompt",
        pack_message({"prompt": prompt, "session_id": "bench"}, []),
        args.timeout,
    )
    _, tensors = unpack_message(context_resp.content)
    post(
        f"{args.edge}/session",
        pack_message({"session_id": "bench", "prompt": prompt},
                     [tensors[0], tensors[1]]),
        args.timeout,
    )
    rtt_ms = rtt_baseline(args.cloud, args.edge, args.timeout)
    print(f"[sweep] RTT baseline (p50 of 60 empty POSTs): {rtt_ms:.3f} ms")

    grids = [tuple(int(v) for v in item.split("x")) for item in args.grids.split(",")]
    dtypes = args.dtypes.split(",")
    variants = {}
    for grid in grids:
        for dtype in dtypes:
            out = run_variant(args, grid, dtype, prompt, primary, wrist, proprio, rtt_ms)
            variants[f"{grid[0]}x{grid[1]}_{dtype}"] = out

    result = {
        "tag": args.tag,
        "ts": time.strftime("%F %T"),
        "host": socket.gethostname(),
        "cloud_url": args.cloud,
        "edge_url": args.edge,
        "cloud_health": cloud_health,
        "edge_health": edge_health,
        "rtt_baseline_ms": rtt_ms,
        "n": args.n,
        "warmup": args.warmup,
        "instruction": args.instruction,
        "variants": {k: v["summary"] for k, v in variants.items()},
        "records": {k: v["records"] for k, v in variants.items()},
    }
    print("\n[sweep] summary (p50 legs, ms)")
    print(f"  {'variant':14s} {'tokens':>6s} {'kB':>8s} {'z_err%':>7s} "
          f"{'codec':>7s} {'xfer':>7s} {'kv_re':>7s} {'denoise':>8s} {'rt':>8s}")
    for label, s in result["variants"].items():
        legs = s["legs"]
        print(f"  {label:14s} {s['tokens']:6d} {s['payload_bytes'] / 1e3:8.1f} "
              f"{(s['z_rel_err_pct'] if s['z_rel_err_pct'] is not None else float('nan')):7.3f} "
              f"{legs['cloud_codec_ms']['p50']:7.2f} {legs['z_transfer_ms']['p50']:7.2f} "
              f"{legs['kv_reinit_ms']['p50']:7.2f} {legs['edge_denoise_ms']['p50']:8.2f} "
              f"{legs['round_trip_ms']['p50']:8.1f}")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[sweep] wrote {out_path}")


if __name__ == "__main__":
    main()
