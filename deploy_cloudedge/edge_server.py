"""LeapBot-CE edge stage service (FastAPI).

Hosts the 1.02B action expert, the frozen SigLIP grounding stack, and the
proprio encoder.  The cloud payload ``z`` (0.34 MB bf16) arrives on
``/payload`` and is re-instantiated into per-layer K/V tokens immediately on
arrival (checkpoint's own per-layer scale vectors — zero transmission cost);
``/step`` denoises the action chunk from the latest arrived payload against
fresh camera views and current proprio.  A combined one-shot
``/infer_action_edge`` is provided for deployments that prefer a single
round trip; the bench client uses the split (/payload then /step) so the z
transfer leg is measured in isolation.

Boot sequence: full model build + delta ckpt, then the cloud-side weights
(Wan 5B video expert, VAE, T5 text encoder) are freed so the process mirrors
an edge-only host (fits a 16 GB-class GPU later; the 4090D cloud / 5080 edge
split needs no code change, only different hosts and ports).

Usage (debug H800 pod; see run_h800_debug.sh):
    CUDA_VISIBLE_DEVICES=0 python edge_server.py \
        --port 8601 --ckpt <step_xxxxxx.pt> --stats <dataset_stats.json>
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import torch
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from protocol import pack_message, unpack_message
from server_common import (
    compose_deploy_cfg,
    compose_image,
    edge_denoise_parts,
    load_policy,
    rebuild_cache,
    strip_cloud_weights,
    views_from_composite,
)
from fastwam.models.wan22.leapbotce import CloudPlanningCache

app = FastAPI(title="LeapBot-CE edge stage")
_STATE: dict = {}
_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8601)
    parser.add_argument("--task", default="libero_all_leapbotce_codec_lora")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mot-pretrained",
        default="/mnt/c2-worldmodel/2638088/fastwam/fastwam_release/libero_uncond_2cam224.pt",
    )
    return parser.parse_args()


def boot(args) -> dict:
    overrides = [
        f"model.mot_pretrained_path={args.mot_pretrained}",
        "model.load_text_encoder=true",
        "model.skip_dit_load_from_pretrain=false",
    ]
    cfg = compose_deploy_cfg(args.task, overrides)
    policy = load_policy(cfg, args.ckpt, args.stats, args.device)
    model = policy.model
    if model.cloud_kv_codec is None:
        raise RuntimeError("deployed checkpoint has no planning codec scales")
    codec = model.cloud_kv_codec
    state = {
        "policy": policy,
        "scale_k": codec.scale_k.detach().clone().to(model.device),
        "scale_v": codec.scale_v.detach().clone().to(model.device),
        "num_tokens": int(codec.num_tokens),
        "num_layers": int(model.mot.num_layers),
        "sessions": {},   # session_id -> (context, context_mask)
        "latest": {},     # session_id -> (CloudPlanningCache, replan_id)
        "ckpt": args.ckpt,
    }
    strip_cloud_weights(model)
    torch.set_grad_enabled(False)
    return state


@app.get("/health")
def health():
    return {
        "status": "ok" if _STATE else "booting",
        "stage": "edge",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "payload_tokens": _STATE.get("num_tokens"),
        "ckpt": _STATE.get("ckpt"),
    }


@app.post("/ping")
async def ping(request: Request):
    body = await request.body()
    return Response(content=body, media_type="application/octet-stream")


@app.post("/session")
async def session(request: Request):
    """Register the language context (cloud /encode_prompt output) for a task."""
    header, tensors = unpack_message(await request.body())
    model = _STATE["policy"].model
    context = tensors[0].to(device=model.device, dtype=model.torch_dtype)
    mask = tensors[1].bool().to(device=context.device)
    session_id = header.get("session_id") or header.get("prompt", "session")
    _STATE["sessions"][session_id] = (context, mask)
    return JSONResponse({"session_id": session_id, "context_len": int(context.shape[1])})


@app.post("/payload")
async def payload(request: Request):
    """Receive the latest cloud payload z; re-instantiate per-layer K/V now."""
    body = await request.body()
    recv_start = time.perf_counter()
    header, reinit_ms, recv_done = await run_in_threadpool(_payload_sync, body, recv_start)
    return JSONResponse(
        {"replan_id": header.get("replan_id"), "kv_reinit_ms": round(reinit_ms, 4)},
        headers={
            "X-Kv-Reinit-Ms": f"{reinit_ms:.3f}",
            "X-Recv-Start": repr(recv_start),
            "X-Recv-Done": repr(recv_done),
        },
    )


def _payload_sync(body: bytes, recv_start: float) -> tuple[dict, float, float]:
    with _LOCK:
        header, tensors = unpack_message(body)
        z = tensors[0].to(device=_STATE["policy"].model.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        context, mask = _STATE["sessions"][header["session_id"]]
        cache = rebuild_cache(z, _STATE["scale_k"], _STATE["scale_v"], context, mask,
                              int(header["seq_len"]), int(header["tokens_per_frame"]))
        torch.cuda.synchronize()
        reinit_ms = (time.perf_counter() - t0) * 1e3
        _STATE["latest"][header["session_id"]] = (cache, header.get("replan_id"))
        return header, reinit_ms, time.perf_counter()


@app.post("/payload_raw")
async def payload_raw(request: Request):
    """No-codec interface: receive the raw per-layer K/V cache and place it.

    Counterfactual of /payload — the cache arrives ready-made (no per-layer
    scale reconstruction), so the server-side cost is unpack + H2D + cache
    assembly only.
    """
    body = await request.body()
    recv_start = time.perf_counter()
    header, placed_ms, recv_done = await run_in_threadpool(_payload_raw_sync, body, recv_start)
    return JSONResponse(
        {"replan_id": header.get("replan_id"), "kv_place_ms": round(placed_ms, 4)},
        headers={
            "X-Kv-Place-Ms": f"{placed_ms:.3f}",
            "X-Recv-Start": repr(recv_start),
            "X-Recv-Done": repr(recv_done),
        },
    )


def _payload_raw_sync(body: bytes, recv_start: float) -> tuple[dict, float, float]:
    with _LOCK:
        t0 = time.perf_counter()
        header, tensors = unpack_message(body)
        model = _STATE["policy"].model
        num_layers = int(header["num_layers"])
        keys = [t.to(device=model.device, dtype=torch.bfloat16) for t in tensors[:num_layers]]
        values = [t.to(device=model.device, dtype=torch.bfloat16) for t in tensors[num_layers:]]
        context, mask = _STATE["sessions"][header["session_id"]]
        cache = CloudPlanningCache(keys, values, context, mask,
                                   int(header["seq_len"]), int(header["tokens_per_frame"]))
        torch.cuda.synchronize()
        placed_ms = (time.perf_counter() - t0) * 1e3
        _STATE["latest"][header["session_id"]] = (cache, header.get("replan_id"))
        return header, placed_ms, time.perf_counter()


@app.post("/step")
async def step(request: Request):
    """One edge stage: latest payload + fresh views + proprio -> action chunk."""
    body = await request.body()
    recv_start = time.perf_counter()
    payload_body, header = await run_in_threadpool(_step_sync, body, recv_start)
    timings = header["timings"]
    return Response(
        content=payload_body,
        media_type="application/octet-stream",
        headers={
            "X-Edge-Compute-Ms": f"{header['edge_compute_ms']:.3f}",
            "X-Siglip-Ms": f"{timings.get('siglip', 0.0):.3f}",
            "X-Denoise-Ms": f"{timings.get('denoise', 0.0):.3f}",
            "X-Recv-Start": repr(recv_start),
            "X-Server-Wall": repr(time.time()),
        },
    )


def _step_sync(body: bytes, recv_start: float) -> tuple[bytes, dict]:
    with _LOCK:
        policy = _STATE["policy"]
        model = policy.model
        header, arrays = unpack_message(body)
        primary, wrist = arrays[0].numpy(), arrays[1].numpy()
        cache, replan_id = _STATE["latest"][header["session_id"]]
        composite = compose_image(policy, primary, wrist)
        views = views_from_composite(model, composite)
        proprio_np = np.asarray(header["proprio"], dtype=np.float32)
        proprio = torch.as_tensor(proprio_np).unsqueeze(0).to(model.device, model.torch_dtype)
        sigma = header.get("sigma_shift")
        out, timings = edge_denoise_parts(
            model, cache, views, int(header.get("action_horizon", 32)), proprio,
            int(header.get("num_inference_steps", 10)),
            None if sigma in (None, "") else float(sigma),
            header.get("seed"),
            rand_device=str(header.get("rand_device", "cpu")),
        )
        out_header = {
            "replan_id": replan_id,
            "timings": {k: round(v, 4) for k, v in timings.items()},
            "edge_compute_ms": sum(timings.values()),
            "recv_start_perf": recv_start,
            "recv_start_wall": time.time(),
        }
        return pack_message(out_header, [out["action"]]), out_header


@app.post("/infer_action_edge")
async def infer_action_edge(request: Request):
    """Combined one-shot: z + fresh inputs -> action chunk (single round trip)."""
    body = await request.body()
    recv_start = time.perf_counter()
    payload_body, header = await run_in_threadpool(_infer_sync, body, recv_start)
    return Response(
        content=payload_body,
        media_type="application/octet-stream",
        headers={"X-Edge-Compute-Ms": f"{header['edge_compute_ms']:.3f}",
                 "X-Recv-Start": repr(recv_start)},
    )


def _infer_sync(body: bytes, recv_start: float) -> tuple[bytes, dict]:
    with _LOCK:
        policy = _STATE["policy"]
        model = policy.model
        header, arrays = unpack_message(body)
        z = arrays[0].to(device=model.device, dtype=torch.bfloat16)
        primary, wrist = arrays[1].numpy(), arrays[2].numpy()
        context, mask = _STATE["sessions"][header["session_id"]]
        cache = rebuild_cache(z, _STATE["scale_k"], _STATE["scale_v"], context, mask,
                              int(header["seq_len"]), int(header["tokens_per_frame"]))
        views = views_from_composite(model, compose_image(policy, primary, wrist))
        proprio_np = np.asarray(header["proprio"], dtype=np.float32)
        proprio = torch.as_tensor(proprio_np).unsqueeze(0).to(model.device, model.torch_dtype)
        sigma = header.get("sigma_shift")
        out, timings = edge_denoise_parts(
            model, cache, views, int(header.get("action_horizon", 32)), proprio,
            int(header.get("num_inference_steps", 10)),
            None if sigma in (None, "") else float(sigma),
            header.get("seed"),
            rand_device=str(header.get("rand_device", "cpu")),
        )
        header_out = {
            "timings": {k: round(v, 4) for k, v in timings.items()},
            "edge_compute_ms": sum(timings.values()),
            "recv_start_perf": recv_start,
        }
        return pack_message(header_out, [out["action"]]), header_out


def main() -> None:
    import uvicorn

    args = parse_args()
    _STATE.clear()
    _STATE.update(boot(args))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
