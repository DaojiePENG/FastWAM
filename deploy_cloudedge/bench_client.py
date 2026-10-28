"""Latency benchmark client for the LeapBot-CE cloud-edge deployment.

Plays the robot's control loop against the two FastAPI services:

    t0 -> POST cloud /encode_cloud (aged composite + proprio)   -> z, t1
    t2 -> POST edge  /payload      (z, 0.34 MB)                 -> ack, t3
    t4 -> POST edge  /step         (fresh views + proprio)     -> actions, t5

Attribution:
  - cloud compute / edge compute: CUDA-event sub-timings carried in the
    framed response headers (VAE, prefill, codec / SigLIP, denoise);
  - z transfer: client wall-clock of the /payload call minus the measured
    empty-POST RTT baseline;
  - uplink legs: server receive stamps minus client send stamps (same host
    on the debug pod -> shared CLOCK_MONOTONIC; cross-host uses the wall
    stamps, so the same client works on the 4090D/5080 testbed).

Everything is logged per replan and aggregated to mean / p50 / p95.
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

from protocol import pack_message, unpack_message

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloud", default="http://127.0.0.1:8600")
    parser.add_argument("--edge", default="http://127.0.0.1:8601")
    parser.add_argument("--n", type=int, default=50, help="measured replans")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--instruction", default="put the bowl on the stove",
        help="LIBERO-style task instruction embedded in the prompt template",
    )
    parser.add_argument("--primary", default="", help="agentview frame png (raw camera)")
    parser.add_argument("--wrist", default="", help="wrist frame png (raw camera)")
    parser.add_argument("--out", default="")
    parser.add_argument("--tag", default="h800_lan")
    parser.add_argument(
        "--raw", action="store_true",
        help="no-codec interface: /encode_cloud_raw + /payload_raw (raw K/V cache "
             "crosses the wire) instead of the deployed codec payload")
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def load_frames(args) -> tuple[np.ndarray, np.ndarray]:
    def _load(path: str):
        if not path:
            return None
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    primary, wrist = _load(args.primary), _load(args.wrist)
    if primary is None or wrist is None:
        asset_dir = (Path(__file__).resolve().parents[2] / "LeapBotCEPaper"
                     / "Figures" / "assets" / "mainfig")
        for name, target in (("libero_frame_08.png", 0), ("libero_frame_60.png", 1)):
            candidate = asset_dir / name
            if not candidate.exists():
                continue
            from PIL import Image

            frame = np.asarray(Image.open(candidate).convert("RGB"), dtype=np.uint8)
            if target == 0 and primary is None:
                primary = frame
            elif target == 1 and wrist is None:
                wrist = frame
    if primary is None or wrist is None:
        rng = np.random.default_rng(0)
        primary = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        wrist = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        print("[bench] warning: no frames found; using random noise cameras")
    return primary, wrist


def post(url: str, body: bytes, timeout: float) -> requests.Response:
    return requests.post(
        url, data=body, headers={"Content-Type": "application/octet-stream"},
        timeout=timeout,
    )


def header_float(response: requests.Response, name: str) -> float:
    return float(response.headers[name])


def rtt_baseline(cloud: str, edge: str, timeout: float, rounds: int = 30) -> float:
    samples = []
    for _ in range(rounds):
        for base in (cloud, edge):
            t0 = time.perf_counter()
            post(f"{base}/ping", b"0" * 1024, timeout)
            samples.append((time.perf_counter() - t0) * 1e3)
    return float(statistics.median(samples))


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
    }


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
    mode = "raw" if args.raw else "codec"
    encode_url = f"{args.cloud}/{'encode_cloud_raw' if args.raw else 'encode_cloud'}"
    payload_url = f"{args.edge}/{'payload_raw' if args.raw else 'payload'}"
    kv_ms_header = "X-Kv-Place-Ms" if args.raw else "X-Kv-Reinit-Ms"
    print(f"[bench] mode {mode}")
    print(f"[bench] cloud: {cloud_health}")
    print(f"[bench] edge:  {edge_health}")
    payload_bytes = 0 if args.raw else int(cloud_health["payload_bytes"])

    # task session: the cloud encodes the prompt once (T5), the edge registers
    # the context once per task — amortized outside the per-replan legs.
    context_resp = post(
        f"{args.cloud}/encode_prompt",
        pack_message({"prompt": prompt, "session_id": "bench"}, []),
        args.timeout,
    )
    header, tensors = unpack_message(context_resp.content)
    post(
        f"{args.edge}/session",
        pack_message({"session_id": "bench", "prompt": prompt},
                     [tensors[0], tensors[1]]),
        args.timeout,
    )
    print(f"[bench] session ready (context_len={header.get('context_len', 'n/a')})")

    rtt_ms = rtt_baseline(args.cloud, args.edge, args.timeout)
    print(f"[bench] RTT baseline (p50 of 60 empty POSTs): {rtt_ms:.3f} ms")

    records = []
    total = args.warmup + args.n
    for i in range(total):
        replan_id = f"r{i}"
        obs_body = pack_message(
            {"prompt": prompt, "proprio": proprio.tolist(), "replan_id": replan_id},
            [primary, wrist],
        )
        t0 = time.perf_counter()
        r1 = post(encode_url, obs_body, args.timeout)
        t1 = time.perf_counter()
        cloud_header, cloud_tensors = unpack_message(r1.content)
        if payload_bytes == 0:
            payload_bytes = int(cloud_header["payload_bytes"])

        t2 = time.perf_counter()
        r2 = post(
            payload_url,
            pack_message({"session_id": "bench", "replan_id": replan_id,
                          "seq_len": cloud_header["seq_len"],
                          "tokens_per_frame": cloud_header["tokens_per_frame"],
                          **({"num_layers": cloud_header["num_layers"]}
                             if args.raw else {})},
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
            "cloud_uplink_ms": (float(r1.headers["X-Recv-Start"]) - t0) * 1e3,
            "cloud_vae_ms": cloud_timings.get("vae", 0.0),
            "cloud_prefill_ms": cloud_timings.get("prefill", 0.0),
            "cloud_codec_ms": cloud_timings.get("codec", 0.0),
            "cloud_compute_ms": header_float(r1, "X-Cloud-Compute-Ms"),
            "z_transfer_ms": (t3 - t2) * 1e3 - rtt_ms,
            "kv_reinit_ms": header_float(r2, kv_ms_header),
            "edge_uplink_ms": (float(r3.headers["X-Recv-Start"]) - t4) * 1e3,
            "edge_siglip_ms": edge_timings.get("siglip", 0.0),
            "edge_denoise_ms": edge_timings.get("denoise", 0.0),
            "edge_compute_ms": header_float(r3, "X-Edge-Compute-Ms"),
            "round_trip_ms": (t5 - t0) * 1e3,
            "action_abs_mean": float(np.abs(action).mean()),
            "action_finite": bool(np.isfinite(action).all()),
            "action_shape": list(action.shape),
        }
        if not record["action_finite"]:
            raise RuntimeError(f"non-finite actions at {replan_id}: {record}")
        records.append(record)
        if i == args.warmup - 1:
            print(f"[bench] warmup done ({args.warmup} replans)")
        if (i + 1) % 10 == 0:
            print(f"[bench] {i + 1}/{total} replans")

    measured = [r for r in records if not r["warmup"]]
    leg_names = ("cloud_uplink_ms", "cloud_vae_ms", "cloud_prefill_ms",
                 "cloud_codec_ms", "cloud_compute_ms", "z_transfer_ms",
                 "kv_reinit_ms", "edge_uplink_ms", "edge_siglip_ms",
                 "edge_denoise_ms", "edge_compute_ms", "round_trip_ms")
    legs = {name: stats([r[name] for r in measured]) for name in leg_names}
    print(f"\n[bench] per-leg stats over {len(measured)} replans (payload {payload_bytes} B)")
    for name, agg in legs.items():
        print(f"  {name:18s} mean {agg['mean']:9.3f}  p50 {agg['p50']:9.3f}  p95 {agg['p95']:9.3f}")

    result = {
        "tag": args.tag,
        "mode": mode,
        "ts": time.strftime("%F %T"),
        "host": socket.gethostname(),
        "cloud_url": args.cloud,
        "edge_url": args.edge,
        "cloud_health": cloud_health,
        "edge_health": edge_health,
        "payload_bytes": payload_bytes,
        "instruction": args.instruction,
        "prompt": prompt,
        "n": args.n,
        "warmup": args.warmup,
        "rtt_baseline_ms": rtt_ms,
        "legs": legs,
        "records": records,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"[bench] wrote {out_path}")


if __name__ == "__main__":
    main()
