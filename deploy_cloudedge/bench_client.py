"""Client-clock latency and capacity benchmark for the split deployment.

No clocks from different machines are subtracted. HTTP durations include
upload, queueing, server work and download; response-body read durations are
client observations (some bytes may already have arrived with the headers).
The stock protocol relays z cloud -> client -> edge, so both copies are
counted. GPU event timings are attribution, not whole-request timings.

The default is one closed-loop stream. --schedule pipeline overlaps the next
cloud request with the current edge payload+step, with at most one cloud
lookahead. Edge operations remain serial and every returned replan_id is
checked: /step consumes the latest cache of its session. No robot is actuated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import socket
import statistics
import sys
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import requests
import torch

from protocol import pack_message, unpack_message

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)
DEPRECATED_METRICS = {
    "cloud_uplink_ms": "Invalid cross-host perf_counter subtraction; not measured.",
    "edge_uplink_ms": "Invalid cross-host perf_counter subtraction; not measured.",
    "z_transfer_ms": "Old payload HTTP duration minus pooled RTT included installation "
                     "and omitted the cloud response leg. No one-way transfer is measured.",
    "rtt_baseline_ms": "Pooled cloud/edge RTT replaced by per-endpoint RTT samples.",
}
MEASUREMENT_NOTES = {
    "clock": "All request and replan durations use this client's perf_counter only.",
    "topology": "cloud -> client -> edge relay; payload crosses two HTTP legs.",
    "request_ms": "From dispatch until complete response body; includes upload, HTTP, "
                  "server queue/host/GPU work and response download.",
    "response_headers_wait_ms": "Dispatch until response headers; not upload-only or TTFB.",
    "response_download_ms": "Time reading response body after headers were exposed to "
                            "requests; buffered/previously arrived bytes are included. "
                            "Not pure network time or a one-way latency.",
    "payload_server_install_ms": "Edge X-Recv-Done minus X-Recv-Start (same server clock): "
                                 "queue, unpack, H2D and installation; starts after body reception.",
    "payload_request_residual_ms": "Payload request duration minus server install interval; "
                                  "includes bidirectional network and other HTTP/server overhead. "
                                  "Not a one-way network latency.",
    "body_bytes": "Application HTTP body only, including LBC1 framing; excludes HTTP/TCP/IP headers.",
    "gpu_compute_ms": "Server CUDA-event substeps; excludes other host/server work.",
    "capacity": "Successful chunks / measured batch elapsed, including failed attempt time; "
                "warmup and one-time prompt/session/cache preparation excluded.",
    "actions": "Each step generates 32 actions by default; deployment consumes 10 before "
               "replanning. Neither action rate is measured robot control frequency.",
    "warmup": "Warmup batch uses the selected schedule, drains fully, then measured batch starts.",
    "stages": "full=cloud+payload+step; cloud=encode only; payload=repeated install of fixed "
              "precomputed payload; edge=step only with fixed preinstalled cache.",
    "concurrency": "One stream; pipeline has one cloud lookahead and serial edge calls. "
                   "Not multi-session or open-loop saturation capacity.",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", default="http://127.0.0.1:8600")
    parser.add_argument("--edge", default="http://127.0.0.1:8601")
    parser.add_argument("--n", type=int, default=50, help="measured attempts (failures count)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--instruction", default="put the bowl on the stove")
    parser.add_argument("--primary", default="", help="agentview RGB frame")
    parser.add_argument("--wrist", default="", help="wrist RGB frame")
    parser.add_argument("--out", default="")
    parser.add_argument("--tag", default="cloudedge")
    parser.add_argument("--raw", action="store_true", help="relay raw per-layer K/V")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--ping-rounds", type=int, default=10)
    parser.add_argument("--stage", choices=("full", "cloud", "payload", "edge"), default="full")
    parser.add_argument("--schedule", choices=("serial", "pipeline"), default="serial")
    parser.add_argument("--session-id", default="", help="default: unique benchmark UUID")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--executed-steps", type=int, default=10,
                        help="declared deployment consumption per chunk, not inference steps")
    parser.add_argument("--inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="edge sampling seed")
    parser.add_argument("--connection", choices=("keep-alive", "close"), default="keep-alive")
    args = parser.parse_args(argv)
    if args.n < 1 or args.warmup < 0 or args.ping_rounds < 1:
        parser.error("n and ping-rounds must be positive; warmup must be nonnegative")
    if not 0 < args.executed_steps <= args.action_horizon or args.inference_steps < 1:
        parser.error("require 0 < executed-steps <= action-horizon and positive inference-steps")
    if args.schedule == "pipeline" and args.stage != "full":
        parser.error("pipeline schedule is defined for --stage full only")
    return args


def load_frames(args) -> tuple[np.ndarray, np.ndarray]:
    def load(path):
        if not path:
            return None
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    primary, wrist = load(args.primary), load(args.wrist)
    sources = {"primary": args.primary, "wrist": args.wrist}
    asset_dir = (Path(__file__).resolve().parents[2] / "LeapBotCEPaper"
                 / "Figures" / "assets" / "mainfig")
    if primary is None:
        p = asset_dir / "libero_frame_08.png"
        primary = load(p) if p.exists() else None
        if primary is not None:
            sources["primary"] = str(p)
    if wrist is None:
        p = asset_dir / "libero_frame_60.png"
        wrist = load(p) if p.exists() else None
        if wrist is not None:
            sources["wrist"] = str(p)
    if primary is None or wrist is None:
        rng = np.random.default_rng(0)
        if primary is None:
            primary = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            sources["primary"] = "fixed_random_uint8_seed0"
        if wrist is None:
            wrist = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            sources["wrist"] = "fixed_random_uint8_seed0"
        print("[bench] warning: missing frame(s) replaced by fixed random RGB; no robot actuation")
    args._frame_sources = sources
    return primary, wrist


def post(url: str, body: bytes, timeout: float) -> requests.Response:
    """Compatibility helper for bw_sweep; always reject HTTP error responses."""
    response = requests.post(url, data=body,
                             headers={"Content-Type": "application/octet-stream"}, timeout=timeout)
    response.raise_for_status()
    return response


def header_float(response: requests.Response, name: str) -> float:
    return float(response.headers[name])


def stats(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {"n": len(array), "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95))}


def rtt_baseline(cloud: str, edge: str, timeout: float, rounds: int = 30) -> float:
    """Legacy compatibility only; new results never pool endpoint RTTs."""
    warnings.warn("Pooled RTT is deprecated; never subtract it to infer network latency.",
                  DeprecationWarning, stacklevel=2)
    samples = []
    for _ in range(rounds):
        for base in (cloud, edge):
            start = time.perf_counter()
            post(f"{base}/ping", b"0" * 1024, timeout)
            samples.append((time.perf_counter() - start) * 1e3)
    return float(statistics.median(samples))


def timed_post(session, url, body, timeout, measurements, name):
    """Populate metrics even on transport/HTTP errors; return fully buffered response."""
    metric = measurements[name] = {"url": url, "request_body_bytes": len(body), "ok": False}
    start = time.perf_counter()
    try:
        with session.post(url, data=body, headers={"Content-Type": "application/octet-stream"},
                          timeout=timeout, stream=True) as response:
            headers_at = time.perf_counter()
            metric["response_headers_wait_ms"] = (headers_at - start) * 1e3
            content = response.content
            done = time.perf_counter()
            metric.update(status_code=response.status_code, response_body_bytes=len(content),
                          response_download_ms=(done - headers_at) * 1e3)
            for header in ("X-Cloud-Compute-Ms", "X-Edge-Compute-Ms", "X-Kv-Reinit-Ms",
                           "X-Kv-Place-Ms"):
                if header in response.headers:
                    metric[header.lower().replace("-", "_")] = float(response.headers[header])
            if "X-Recv-Start" in response.headers and "X-Recv-Done" in response.headers:
                metric["server_install_ms"] = (
                    float(response.headers["X-Recv-Done"]) - float(response.headers["X-Recv-Start"])
                ) * 1e3
            response.raise_for_status()
            metric["ok"] = True
            return response
    except Exception as exc:
        metric["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metric["request_ms"] = (time.perf_counter() - start) * 1e3


class Benchmark:
    def __init__(self, args):
        self.args = args
        self.prompt = DEFAULT_PROMPT.format(task=args.instruction)
        self.primary, self.wrist = load_frames(args)
        self.proprio = [0.0] * 6 + [0.02, 0.02]
        self.session_id = args.session_id or f"bench-{uuid.uuid4().hex}"
        self.cloud = requests.Session()
        self.edge = requests.Session()
        for session in (self.cloud, self.edge):
            session.trust_env = False  # Direct endpoint URLs, no ambient HTTP proxy.
            session.headers["Connection"] = args.connection
        self.encode_url = f"{args.cloud}/{'encode_cloud_raw' if args.raw else 'encode_cloud'}"
        self.payload_url = f"{args.edge}/{'payload_raw' if args.raw else 'payload'}"
        self.health = {}
        self.setup_requests = {}
        self.rtt = {}
        self.fixed = None

    def setup(self):
        endpoints = [("cloud", self.cloud, self.args.cloud)]
        if self.args.stage != "cloud":
            endpoints.append(("edge", self.edge, self.args.edge))
        for name, session, base in endpoints:
            response = session.get(f"{base}/health", timeout=min(self.args.timeout, 30))
            response.raise_for_status()
            health = response.json()
            if health.get("status") != "ok" or health.get("stage") != name:
                raise RuntimeError(f"{name} service not ready: {health}")
            self.health[name] = health
            print(f"[bench] {name}: {health}")
            samples = []
            for i in range(self.args.ping_rounds):
                item = {}
                echo = timed_post(session, f"{base}/ping", b"0" * 1024,
                                  self.args.timeout, item, "ping")
                if echo.content != b"0" * 1024:
                    raise ValueError(f"{name}/ping response corrupted")
                samples.append(item["ping"])
            self.rtt[name] = {"request_ms": stats([s["request_ms"] for s in samples]),
                              "request_body_bytes": 1024, "records": samples}
        if self.args.stage == "cloud":
            # Warm task context once; /encode_cloud reuses the prompt cache.
            timed_post(self.cloud, f"{self.args.cloud}/encode_prompt",
                       pack_message({"prompt": self.prompt, "session_id": self.session_id}, []),
                       self.args.timeout, self.setup_requests, "encode_prompt")
            return
        context_response = timed_post(
            self.cloud, f"{self.args.cloud}/encode_prompt",
            pack_message({"prompt": self.prompt, "session_id": self.session_id}, []),
            self.args.timeout, self.setup_requests, "encode_prompt")
        _, tensors = unpack_message(context_response.content)
        response = timed_post(
            self.edge, f"{self.args.edge}/session",
            pack_message({"session_id": self.session_id, "prompt": self.prompt}, tensors[:2]),
            self.args.timeout, self.setup_requests, "session")
        if response.json().get("session_id") != self.session_id:
            raise ValueError("edge session acknowledgement mismatch")
        if self.args.stage in ("edge", "payload"):
            self.fixed = self.cloud_stage("fixed-cache", False)
            if self.fixed.get("error"):
                raise RuntimeError(self.fixed["error"])
            if self.args.stage == "edge":
                self.install(self.fixed)
                if self.fixed.get("error"):
                    raise RuntimeError(self.fixed["error"])

    def cloud_stage(self, replan_id, warmup):
        record = {"replan_id": replan_id, "warmup": warmup, "ok": False, "requests": {},
                  "started_perf": time.perf_counter(), "started_utc": datetime.now(timezone.utc).isoformat()}
        try:
            start = time.perf_counter()
            body = pack_message({"prompt": self.prompt, "proprio": self.proprio,
                                 "replan_id": replan_id}, [self.primary, self.wrist])
            record["cloud_observation_pack_ms"] = (time.perf_counter() - start) * 1e3
            response = timed_post(self.cloud, self.encode_url, body, self.args.timeout,
                                  record["requests"], "cloud")
            start = time.perf_counter()
            header, tensors = unpack_message(response.content)
            record["cloud_response_unpack_ms"] = (time.perf_counter() - start) * 1e3
            if header.get("replan_id") != replan_id:
                raise ValueError("cloud replan_id mismatch")
            record["payload_bytes"] = int(sum(t.numel() * t.element_size() for t in tensors))
            if record["payload_bytes"] != int(header["payload_bytes"]):
                raise ValueError("cloud declared payload_bytes mismatch")
            record["cloud_timings"] = header.get("timings", {})
            record["cloud_compute_ms"] = header_float(response, "X-Cloud-Compute-Ms")
            record["cloud_header"] = header
            record["_tensors"] = tensors
            record["cloud_ready_perf"] = time.perf_counter()
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    def install(self, record):
        header = record["cloud_header"]
        start = time.perf_counter()
        payload_header = {"session_id": self.session_id, "replan_id": record["replan_id"],
                          "seq_len": header["seq_len"], "tokens_per_frame": header["tokens_per_frame"]}
        if self.args.raw:
            payload_header["num_layers"] = header["num_layers"]
        body = pack_message(payload_header, record["_tensors"])
        record["payload_pack_ms"] = (time.perf_counter() - start) * 1e3
        response = timed_post(self.edge, self.payload_url, body, self.args.timeout,
                              record["requests"], "payload")
        if response.json().get("replan_id") != record["replan_id"]:
            raise ValueError("edge payload replan_id mismatch")
        record["kv_reinit_ms"] = header_float(
            response, "X-Kv-Place-Ms" if self.args.raw else "X-Kv-Reinit-Ms")

    def step(self, record, expected_replan_id):
        start = time.perf_counter()
        body = pack_message({"session_id": self.session_id, "proprio": self.proprio,
                             "action_horizon": self.args.action_horizon,
                             "num_inference_steps": self.args.inference_steps,
                             "sigma_shift": None, "replan_id": record["replan_id"],
                             "seed": self.args.seed}, [self.primary, self.wrist])
        record["step_observation_pack_ms"] = (time.perf_counter() - start) * 1e3
        response = timed_post(self.edge, f"{self.args.edge}/step", body, self.args.timeout,
                              record["requests"], "step")
        start = time.perf_counter()
        header, tensors = unpack_message(response.content)
        action = tensors[0].numpy()
        record["step_response_unpack_ms"] = (time.perf_counter() - start) * 1e3
        record["returned_replan_id"] = header.get("replan_id")
        if header.get("replan_id") != expected_replan_id:
            raise ValueError(f"stale/wrong edge cache: expected {expected_replan_id}, "
                             f"got {header.get('replan_id')}")
        record["edge_timings"] = header.get("timings", {})
        record["edge_compute_ms"] = header_float(response, "X-Edge-Compute-Ms")
        finite = bool(np.isfinite(action).all())
        record.update(action_shape=list(action.shape),
                      action_abs_mean=float(np.abs(action).mean()) if finite else None,
                      action_finite=finite)
        if action.shape != (self.args.action_horizon, 7) or not record["action_finite"]:
            raise ValueError(f"invalid action output: {record['action_shape']}, "
                             f"finite={record['action_finite']}")

    def finish(self, record):
        try:
            if "error" in record:
                return record
            if self.args.stage == "full":
                record["edge_queue_wait_ms"] = (time.perf_counter() - record["cloud_ready_perf"]) * 1e3
                self.install(record)
                self.step(record, record["replan_id"])
            elif self.args.stage == "payload":
                self.install(record)
            elif self.args.stage == "edge":
                self.step(record, self.fixed["replan_id"])
            record["ok"] = True
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            record["completed_perf"] = time.perf_counter()
            record["round_trip_ms"] = (record["completed_perf"] - record["started_perf"]) * 1e3
            for name, metric in record["requests"].items():
                record[f"{name}_request_ms"] = metric["request_ms"]
                for suffix in ("response_headers_wait_ms", "response_download_ms", "request_body_bytes",
                               "response_body_bytes"):
                    if suffix in metric:
                        record[f"{name}_{suffix}"] = metric[suffix]
            payload = record["requests"].get("payload", {})
            if "server_install_ms" in payload:
                record["payload_server_install_ms"] = payload["server_install_ms"]
                record["payload_request_residual_ms"] = payload["request_ms"] - payload["server_install_ms"]
            for leg, timings in (("cloud", record.get("cloud_timings", {})),
                                 ("edge", record.get("edge_timings", {}))):
                for key, value in timings.items():
                    record[f"{leg}_{key}_ms"] = value
            for name in ("cloud_uplink_ms", "edge_uplink_ms", "z_transfer_ms"):
                record[name] = None
            record.pop("_tensors", None)
        return record

    def fixed_record(self, index, warmup):
        record = {"replan_id": f"{'w' if warmup else 'r'}{index}", "warmup": warmup,
                  "ok": False, "requests": {}, "started_perf": time.perf_counter(),
                  "started_utc": datetime.now(timezone.utc).isoformat()}
        if self.args.stage == "payload":
            record["cloud_header"] = self.fixed["cloud_header"]
            record["_tensors"] = self.fixed["_tensors"]
            record["payload_bytes"] = self.fixed["payload_bytes"]
        return record

    def batch(self, count, warmup):
        records = []
        started = time.perf_counter()
        if self.args.schedule == "pipeline" and count:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cloud-prefetch") as pool:
                pending = pool.submit(self.cloud_stage, f"{'w' if warmup else 'r'}0", warmup)
                for i in range(count):
                    record = pending.result()
                    if i + 1 < count:
                        pending = pool.submit(self.cloud_stage, f"{'w' if warmup else 'r'}{i + 1}", warmup)
                    records.append(self.finish(record))
                    if (i + 1) % 10 == 0:
                        print(f"[bench] {'warmup' if warmup else 'measured'} {i + 1}/{count}", flush=True)
        else:
            for i in range(count):
                if self.args.stage in ("full", "cloud"):
                    record = self.cloud_stage(f"{'w' if warmup else 'r'}{i}", warmup)
                else:
                    record = self.fixed_record(i, warmup)
                records.append(self.finish(record))
                if (i + 1) % 10 == 0:
                    print(f"[bench] {'warmup' if warmup else 'measured'} {i + 1}/{count}", flush=True)
        return records, time.perf_counter() - started


def summarize(records, elapsed_s, args):
    successful = [r for r in records if r["ok"]]
    metrics = sorted({key for r in successful for key, val in r.items()
                      if key.endswith("_ms") and isinstance(val, (int, float))})
    legs = {key: stats([r[key] for r in successful if isinstance(r.get(key), (int, float))])
            for key in metrics}
    # Legacy keys are retained explicitly null, never filled with a substitute quantity.
    legs.update({key: None for key in ("cloud_uplink_ms", "edge_uplink_ms", "z_transfer_ms")})
    rate = len(successful) / elapsed_s if elapsed_s else 0.0
    emits_actions = args.stage in ("full", "edge")
    throughput = {
        "measured_elapsed_s": elapsed_s, "attempted": len(records),
        "successful": len(successful), "failed": len(records) - len(successful),
        "completed_operations_per_s": rate,
        "successful_http_requests_per_s": sum(
            metric["ok"] for r in records for metric in r["requests"].values()) / elapsed_s,
        "completed_replans_per_s": rate if args.stage == "full" else None,
        "action_chunks_per_s": rate if emits_actions else None,
        "generated_action_vectors_per_s": rate * args.action_horizon if emits_actions else None,
        "declared_consumed_action_vectors_per_s": rate * args.executed_steps if emits_actions else None,
        "stage": args.stage, "schedule": args.schedule,
    }
    return legs, throughput


def main(argv=None) -> int:
    args = parse_args(argv)
    benchmark = Benchmark(args)
    result = {"schema_version": 2, "tag": args.tag, "mode": "raw" if args.raw else "codec",
              "ts": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
              "cloud_url": args.cloud, "edge_url": args.edge, "session_id": benchmark.session_id,
              "n": args.n, "warmup": args.warmup, "stage": args.stage, "schedule": args.schedule,
              "instruction": args.instruction, "prompt": benchmark.prompt, "seed": args.seed,
              "action_horizon": args.action_horizon, "executed_steps_per_replan": args.executed_steps,
              "num_inference_steps": args.inference_steps, "http_connection": args.connection,
              "client_software": {"python": sys.version, "platform": platform.platform(),
                                  "machine": platform.machine(), "torch": torch.__version__,
                                  "torch_cuda_build": torch.version.cuda,
                                  "numpy": np.__version__, "requests": requests.__version__,
                                  "torch_num_threads": torch.get_num_threads(),
                                  "torch_num_interop_threads": torch.get_num_interop_threads(),
                                  "thread_environment": {name: os.environ.get(name) for name in (
                                      "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                      "TOKENIZERS_PARALLELISM")},
                                  "bench_client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  "protocol_sha256": hashlib.sha256(Path(__file__).with_name("protocol.py").read_bytes()).hexdigest()},
              "input": {"primary": args._frame_sources["primary"],
                        "wrist": args._frame_sources["wrist"],
                        "primary_rgb_sha256": hashlib.sha256(benchmark.primary.tobytes()).hexdigest(),
                        "wrist_rgb_sha256": hashlib.sha256(benchmark.wrist.tobytes()).hexdigest(),
                        "primary_shape": list(benchmark.primary.shape),
                        "wrist_shape": list(benchmark.wrist.shape),
                        "observations": "fixed replay; no physical robot actuation"},
              "measurement_notes": MEASUREMENT_NOTES, "deprecated_metrics": DEPRECATED_METRICS,
              "rtt_baseline_ms": None, "records": [], "status": "setup"}
    exit_code = 1
    try:
        benchmark.setup()
        warmup, _ = benchmark.batch(args.warmup, True)
        result["records"].extend(warmup)
        if any(not r["ok"] for r in warmup):
            raise RuntimeError("Warmup failed; measured batch not started")
        measured, elapsed = benchmark.batch(args.n, False)
        result["records"].extend(measured)
        legs, throughput = summarize(measured, elapsed, args)
        result.update(legs=legs, throughput=throughput,
                      status="ok" if throughput["failed"] == 0 else "measured_with_failures")
        result["payload_bytes"] = next((r["payload_bytes"] for r in measured if "payload_bytes" in r),
                                            benchmark.fixed.get("payload_bytes") if benchmark.fixed else None)
        for name, agg in legs.items():
            if agg is not None:
                print(f"[bench] {name}: p50={agg['p50']:.3f} p95={agg['p95']:.3f} ms")
        print(f"[bench] throughput: {json.dumps(throughput)}")
        exit_code = int(throughput["failed"] != 0)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(f"[bench] ERROR {result['error']}", flush=True)
    finally:
        result.update(cloud_health=benchmark.health.get("cloud"), edge_health=benchmark.health.get("edge"),
                      rtt=benchmark.rtt, setup_requests=benchmark.setup_requests,
                      failure_count=sum(not r["ok"] for r in result["records"]))
        benchmark.cloud.close()
        benchmark.edge.close()
        if args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"[bench] wrote {output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
