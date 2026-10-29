"""Synthetic transport-only HTTP server; standard library, no model/GPU dependency.

POST /ping echoes its request. POST /download?bytes=N responds with N fixed
synthetic bytes to a tiny request. POST /upload reads the body and returns a
small byte-count acknowledgement. These are HTTP application transactions,
not model inference, LBC1/KV processing, or robot control. Common sizes are
preallocated so timed measurements do not include first-use allocation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import platform
import socket
from urllib.parse import parse_qs, urlsplit


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, *args):
        pass

    def reply(self, body, code=200, content_type="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Probe-Kind", "synthetic-transport-only")
        if self.headers.get("Connection", "").lower() == "close":
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def json_reply(self, value, code=200):
        self.reply(json.dumps(value).encode(), code, "application/json")

    def do_GET(self):
        if urlsplit(self.path).path == "/health":
            return self.json_reply({"status": "ok", "stage": "synthetic-transport-only",
                                    "host": socket.gethostname(), "machine": platform.machine(),
                                    "python": platform.python_version(), "preallocated_sizes": list(self.server.bodies),
                                    "started_utc": self.server.started_utc})
        return self.json_reply({"error": "POST /ping, /download?bytes=N, or /upload"}, 404)

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 <= size <= self.server.max_bytes:
                self.close_connection = True
                return self.json_reply({"error": "request body size out of bounds"}, 413)
            body = self.rfile.read(size)
            if len(body) != size:
                self.close_connection = True
                return self.json_reply({"error": "truncated body"}, 400)
            parsed = urlsplit(self.path)
            if parsed.path == "/ping":
                return self.reply(body)
            if parsed.path == "/upload":
                return self.json_reply({"received_bytes": len(body)})
            if parsed.path == "/download":
                output_size = int(parse_qs(parsed.query)["bytes"][0])
                if not 0 <= output_size <= self.server.max_bytes:
                    return self.json_reply({"error": "response body size out of bounds"}, 413)
                payload = self.server.bodies.get(output_size)
                if payload is None:
                    payload = b"x" * output_size
                return self.reply(payload)
            return self.json_reply({"error": "unknown path"}, 404)
        except (ValueError, KeyError, IndexError) as exc:
            return self.json_reply({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8630)
    parser.add_argument("--max-bytes", type=int, default=67108864)
    parser.add_argument("--preallocate", default="1024,344064,36126720")
    args = parser.parse_args()
    sizes = [int(value) for value in args.preallocate.split(",") if value]
    if any(not 0 <= size <= args.max_bytes for size in sizes):
        parser.error("preallocated body size exceeds max-bytes")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.max_bytes = args.max_bytes
    server.bodies = {size: b"x" * size for size in sizes}
    server.started_utc = datetime.now(timezone.utc).isoformat()
    print(f"[transport-only] {args.host}:{args.port}, preallocated={sizes}, no GPU/model", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
