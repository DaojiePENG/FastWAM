"""LeapBot-CE cloud stage service (FastAPI).

Hosts the released Wan video expert + the planning-token codec.  Per replan
it receives the (aged) two-camera composite + proprio, runs

    VAE encode -> one Wan prefill pass -> KV codec,

and returns the only cross-network object: the planning payload ``z``
(pool_grid 7x8 = 56 tokens x 3072, bf16, 0.34 MB).  Sub-step latencies are
measured with CUDA events and returned both in the response header and as
wall-clock stamps for the bench client's leg attribution.

Boot sequence: full model build + delta ckpt + a split-vs-reference self-test
(split path must reproduce ``encode_cloud``/``infer_action_edge``), then the
edge-side weights (1B action expert + SigLIP prefix stack) are freed so the
process mirrors a cloud-only host.

Usage (debug H800 pod; see run_h800_debug.sh):
    CUDA_VISIBLE_DEVICES=3 python cloud_server.py \
        --port 8600 --ckpt <step_xxxxxx.pt> --stats <dataset_stats.json>
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
    composite_to_tensor,
    edge_denoise_parts,
    encode_cloud_parts,
    encode_cloud_raw_parts,
    encode_prompt_cached,
    load_policy,
    rebuild_cache,
    strip_edge_weights,
    views_from_composite,
)

app = FastAPI(title="LeapBot-CE cloud stage")
_STATE: dict = {}
_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--task", default="libero_all_leapbotce_codec_lora")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mot-pretrained",
        default="/mnt/c2-worldmodel/2638088/fastwam/fastwam_release/libero_uncond_2cam224.pt",
        help="released FastWAM checkpoint the run trained against (frozen base)",
    )
    parser.add_argument("--skip-self-test", action="store_true")
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
        raise RuntimeError("deployed checkpoint has no planning codec; nothing to transmit")
    num_tokens = int(model.cloud_kv_codec.num_tokens)
    payload_bytes = num_tokens * int(model.cloud_kv_codec.cache_dim) * 2  # bf16

    if not args.skip_self_test:
        max_diff = run_self_test(policy)
        print(f"[cloud] self-test split-vs-reference max|diff| = {max_diff:.3e}", flush=True)
        if max_diff > 1e-2:
            raise RuntimeError(f"self-test FAILED (max diff {max_diff:.3e})")
        # Rebuild the prompt cache entry the self-test used; encode_prompt is
        # deterministic so the cached context stays valid after stripping.
        strip_edge_weights(model)
    else:
        strip_edge_weights(model)

    torch.set_grad_enabled(False)
    return {
        "policy": policy,
        "num_tokens": num_tokens,
        "payload_bytes": payload_bytes,
        "num_layers": int(model.mot.num_layers),
        "ckpt": args.ckpt,
    }


def run_self_test(policy) -> float:
    """Split path vs reference encode_cloud + infer_action_edge (same seed)."""
    model = policy.model
    rng = np.random.default_rng(0)
    primary = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    wrist = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    proprio_np = np.concatenate([np.zeros(6, np.float32), np.full(2, 0.02, np.float32)])
    proprio = torch.as_tensor(proprio_np).unsqueeze(0).to(model.device, model.torch_dtype)
    composite = composite_to_tensor(model, np.concatenate(
        [primary, wrist], axis=1))  # both already 224 -> shape_meta resize is identity
    prompt = "self test: put the bowl on the stove"

    reference = model.infer_action_edge(
        model.encode_cloud(composite, prompt=prompt, cloud_proprio=proprio),
        model._split_views(composite, model.edge_num_views),
        32, edge_proprio=proprio, num_inference_steps=10, seed=0,
    )

    context, mask = encode_prompt_cached(policy, prompt)
    z, seq_len, tpf, _ = encode_cloud_parts(model, composite, context, mask, proprio)
    cache = rebuild_cache(z, model.cloud_kv_codec.scale_k, model.cloud_kv_codec.scale_v,
                          context, mask, seq_len, tpf)
    split, _ = edge_denoise_parts(
        model, cache, model._split_views(composite, model.edge_num_views),
        32, proprio, 10, None, seed=0)
    return float((reference["action"] - split["action"]).abs().max())


@app.get("/health")
def health():
    return {
        "status": "ok" if _STATE else "booting",
        "stage": "cloud",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "payload_tokens": _STATE.get("num_tokens"),
        "payload_bytes": _STATE.get("payload_bytes"),
        "ckpt": _STATE.get("ckpt"),
    }


@app.post("/ping")
async def ping(request: Request):
    body = await request.body()
    return Response(content=body, media_type="application/octet-stream")


@app.post("/encode_prompt")
async def encode_prompt(request: Request):
    header, _ = unpack_message(await request.body())
    prompt = header["prompt"]
    context, mask = encode_prompt_cached(_STATE["policy"], prompt)
    started = time.perf_counter()
    body = pack_message(
        {"prompt": prompt, "server_ts": time.time(), "perf_ts": started},
        [context, mask],
    )
    return Response(content=body, media_type="application/octet-stream")


@app.post("/encode_cloud")
async def encode_cloud(request: Request):
    """One replan's cloud stage: composite+proprio in, planning payload z out."""
    body = await request.body()
    recv_start = time.perf_counter()
    result = await run_in_threadpool(_encode_cloud_sync, body, recv_start)
    payload, header = result
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "X-Cloud-Compute-Ms": f"{header['cloud_compute_ms']:.3f}",
            "X-Recv-Start": repr(recv_start),
            "X-Server-Wall": repr(time.time()),
        },
    )


def _encode_cloud_sync(body: bytes, recv_start: float) -> tuple[bytes, dict]:
    with _LOCK:
        policy = _STATE["policy"]
        model = policy.model
        header, arrays = unpack_message(body)
        primary, wrist = arrays[0].numpy(), arrays[1].numpy()
        composite = compose_image(policy, primary, wrist)
        image = composite_to_tensor(model, composite)
        proprio_np = np.asarray(header["proprio"], dtype=np.float32)
        proprio = torch.as_tensor(proprio_np).unsqueeze(0).to(model.device, model.torch_dtype)
        context, mask = encode_prompt_cached(policy, header["prompt"])
        z, seq_len, tpf, timings = encode_cloud_parts(model, image, context, mask, proprio)
        out_header = {
            "replan_id": header.get("replan_id"),
            "seq_len": seq_len,
            "tokens_per_frame": tpf,
            "payload_bytes": int(z.numel() * z.element_size()),
            "timings": {k: round(v, 4) for k, v in timings.items()},
            "cloud_compute_ms": sum(timings.values()),
            "recv_start_perf": recv_start,
            "recv_start_wall": time.time(),
        }
        return pack_message(out_header, [z]), out_header


@app.post("/encode_cloud_raw")
async def encode_cloud_raw(request: Request):
    """No-codec interface: composite+proprio in, the raw per-layer K/V cache out.

    Counterfactual endpoint for the transmission comparison — same VAE +
    prefill, but the payload is the uncompressed 30-layer K/V cache
    (2 x layers x video tokens x 3072 bf16 ~ 36 MB) instead of z.
    """
    body = await request.body()
    recv_start = time.perf_counter()
    payload, header = await run_in_threadpool(_encode_cloud_raw_sync, body, recv_start)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "X-Cloud-Compute-Ms": f"{header['cloud_compute_ms']:.3f}",
            "X-Recv-Start": repr(recv_start),
            "X-Server-Wall": repr(time.time()),
        },
    )


def _encode_cloud_raw_sync(body: bytes, recv_start: float) -> tuple[bytes, dict]:
    with _LOCK:
        policy = _STATE["policy"]
        model = policy.model
        header, arrays = unpack_message(body)
        primary, wrist = arrays[0].numpy(), arrays[1].numpy()
        composite = compose_image(policy, primary, wrist)
        image = composite_to_tensor(model, composite)
        proprio_np = np.asarray(header["proprio"], dtype=np.float32)
        proprio = torch.as_tensor(proprio_np).unsqueeze(0).to(model.device, model.torch_dtype)
        context, mask = encode_prompt_cached(policy, header["prompt"])
        cache_k, cache_v, seq_len, tpf, timings = encode_cloud_raw_parts(
            model, image, context, mask, proprio)
        tensors = list(cache_k) + list(cache_v)
        cloud_compute_ms = sum(timings.values())
        out_header = {
            "replan_id": header.get("replan_id"),
            "seq_len": seq_len,
            "tokens_per_frame": tpf,
            "num_layers": len(cache_k),
            "payload_bytes": int(sum(t.numel() * t.element_size() for t in tensors)),
            "timings": {k: round(v, 4) for k, v in timings.items()},
            "cloud_compute_ms": cloud_compute_ms,
            "recv_start_perf": recv_start,
            "recv_start_wall": time.time(),
        }
        return pack_message(out_header, tensors), out_header


def main() -> None:
    import uvicorn

    args = parse_args()
    _STATE.clear()
    _STATE.update(boot(args))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
