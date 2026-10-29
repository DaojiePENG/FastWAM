"""Synthetic transport-only HTTP transactions, without model/GPU dependencies.

--operation download sends a 1 B request and receives N synthetic bytes;
upload sends N B and receives a small byte-count acknowledgement; echo sends
N B and receives the same N B (/ping also exists in the model servers).
Durations use only the client clock and measure complete HTTP transactions,
not one-way physical link latency. Body goodput is application-level only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import http.client
import json
from pathlib import Path
import socket
import time
from urllib.parse import urlsplit


def stats(values):
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    ordered = sorted(values)
    def percentile(p):
        position = (len(ordered) - 1) * p
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    return {"n": len(values), "mean": sum(values)/len(values), "p50": percentile(.5), "p95": percentile(.95)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", action="append", required=True, help="base URL; repeat for independent probes")
    parser.add_argument("--operation", choices=("echo", "download", "upload"), default="echo")
    parser.add_argument("--sizes", default="1024,344064,36126720", help="comma-separated synthetic body sizes")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--connection", choices=("keep-alive", "close"), default="keep-alive")
    parser.add_argument("--tag", default="")
    parser.add_argument("--repetition", type=int, default=1, help="technical run index")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sizes = [int(size) for size in args.sizes.split(",")]
    if args.n < 1 or args.warmup < 0 or any(size < 0 for size in sizes):
        parser.error("positive n, nonnegative warmup and body sizes required")
    result = {"schema_version": 2, "host": socket.gethostname(), "ts": datetime.now(timezone.utc).isoformat(),
              "tag": args.tag, "repetition": args.repetition,
              "replication_unit": "technical repeated HTTP transactions between the same hosts with fixed synthetic bodies",
              "measurement": "synthetic transport-only HTTP transaction completion, application body bytes only",
              "operation": args.operation,
              "notes": {"request_ms": "Dispatch through complete response body using client perf_counter only.",
                        "request_upload_write_ms": "Writing headers/body to socket; includes local buffer and TCP "
                                                   "backpressure, not one-way upload latency.",
                        "response_download_ms": "Body-read time after response headers; includes buffered bytes; "
                                                "not isolated physical transfer time.",
                        "goodput": "Successful chosen-direction application body bits / measured batch elapsed. "
                                   "Includes HTTP turnaround, host copies, failures and client validation time. "
                                   "Echo aggregate counts both directions. Not iperf bandwidth.",
                        "data": "Fixed synthetic x bytes, not model-generated z or LBC1 serialization.",
                        "warmup": "Excluded independently for every endpoint and size; connections reuse by default."},
              "http_connection": args.connection, "n": args.n, "warmup": args.warmup, "probes": []}
    failed = 0
    try:
        for endpoint in args.endpoint:
            parsed = urlsplit(endpoint)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError(f"invalid HTTP endpoint: {endpoint}")
            for size in sizes:
                conn_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
                connection = conn_type(parsed.hostname, parsed.port, timeout=args.timeout)
                body = b"0" if args.operation == "download" else b"x" * size
                expected = b"x" * size
                path = {"echo": "/ping", "download": f"/download?bytes={size}", "upload": "/upload"}[args.operation]
                path = parsed.path.rstrip("/") + path
                probe = {"endpoint": endpoint, "operation": args.operation, "synthetic_body_bytes": size,
                         "request_body_bytes": len(body), "records": []}
                result["probes"].append(probe)
                measured_start = None
                try:
                    for i in range(args.warmup + args.n):
                        if i == args.warmup:
                            measured_start = time.perf_counter()
                        record = {"index": i, "warmup": i < args.warmup, "ok": False,
                                  "started_utc": datetime.now(timezone.utc).isoformat()}
                        probe["records"].append(record)
                        started = time.perf_counter()
                        try:
                            connection.request("POST", path, body=body,
                                               headers={"Content-Type": "application/octet-stream",
                                                        "Connection": args.connection})
                            uploaded = time.perf_counter()
                            record["request_upload_write_ms"] = (uploaded-started)*1e3
                            response = connection.getresponse()
                            headers_at = time.perf_counter()
                            record.update(status_code=response.status,
                                          response_headers_wait_ms=(headers_at-started)*1e3)
                            received = response.read()
                            done = time.perf_counter()
                            record.update(response_body_bytes=len(received), response_download_ms=(done-headers_at)*1e3,
                                          request_ms=(done-started)*1e3)
                            if response.status != 200:
                                raise ValueError(f"HTTP {response.status}: {received[:200]!r}")
                            if args.operation == "upload":
                                if json.loads(received).get("received_bytes") != size:
                                    raise ValueError("upload byte count mismatch")
                            elif received != expected:
                                raise ValueError("response content mismatch")
                            record["ok"] = True
                        except Exception as exc:
                            record["error"] = f"{type(exc).__name__}: {exc}"
                            record.setdefault("request_ms", (time.perf_counter()-started)*1e3)
                            failed += 1
                            connection.close()
                        finally:
                            if args.connection == "close":
                                connection.close()
                    elapsed = time.perf_counter() - measured_start
                finally:
                    connection.close()
                measured = [r for r in probe["records"] if not r["warmup"]]
                good = [r for r in measured if r["ok"]]
                for key in ("request_ms", "request_upload_write_ms", "response_headers_wait_ms", "response_download_ms"):
                    probe[key] = stats([r[key] for r in good])
                multiplier = 2 if args.operation == "echo" else 1
                probe.update(measured_elapsed_s=elapsed, successful=len(good), failed=len(measured)-len(good),
                             operations_per_s=len(good)/elapsed,
                             application_body_mbit_per_s=multiplier*size*len(good)*8/elapsed/1e6)
                print(f"[link] {args.operation} {endpoint} {size} B: "
                      f"p50={probe['request_ms']['p50']} ms, failed={probe['failed']}", flush=True)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        failed += 1
    finally:
        result["failure_count_including_warmup"] = failed
        result["status"] = "ok" if not failed else "failed"
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[link] wrote {output}")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
