"""Run a real 6B full-prefix forward/backward smoke test on one LIBERO sample."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data._utils.collate import default_collate

from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging


register_default_resolvers()
logger = get_logger(__name__)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def run_smoke(cfg: DictConfig) -> dict:
    smoke = cfg.get("smoke")
    if smoke is None:
        raise ValueError("missing +smoke configuration")
    checkpoint = Path(str(smoke.checkpoint)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output_dir))
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)

    precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    dtype = _mixed_precision_to_model_dtype(precision)
    device = str(smoke.get("device", "cuda:0"))
    target_history_blocks = int(smoke.get("history_blocks", 50))

    dataset = instantiate(cfg.data.train)
    selected_index = None
    for dataset_index, frame_index in enumerate(dataset._valid_replan_indices):
        block = dataset._episode_step[frame_index] // dataset.replan_steps
        if block == target_history_blocks:
            selected_index = dataset_index
            break
    if selected_index is None:
        raise ValueError(f"dataset contains no sample with H={target_history_blocks}")
    sample = default_collate([dataset[selected_index]])
    actual_history_blocks = int(sample["history_valid_blocks"].sum().item())
    if actual_history_blocks != target_history_blocks:
        raise AssertionError((actual_history_blocks, target_history_blocks))

    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    model.load_checkpoint(str(checkpoint), optimizer=None)
    model.eval()
    model.requires_grad_(False)
    model.configure_trainable_parameters()
    if model.proprio_encoder is not None:
        model.proprio_encoder.train()
        model.proprio_encoder.requires_grad_(True)

    seed = int(cfg.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(torch.device(device))
    native_reference = None
    if bool(smoke.get("compare_native", smoke.get("compare_native_h0", True))):
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=dtype, enabled=precision != "no"
        ):
            native_loss, native_metrics = FastWAM.training_loss(
                model, sample, tiled=False
            )
        native_reference = {
            "loss": float(native_loss.detach().float().item()),
            "loss_video": float(native_metrics["loss_video"]),
            "loss_action": float(native_metrics["loss_action"]),
        }
        del native_loss
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state, torch.device(device))
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    start = time.perf_counter()
    with torch.autocast(
        device_type="cuda", dtype=dtype, enabled=precision != "no"
    ):
        loss, metrics = model.training_loss(sample)
    torch.cuda.synchronize(torch.device(device))
    forward_s = time.perf_counter() - start

    backward_start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(torch.device(device))
    backward_s = time.perf_counter() - backward_start
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
    if not gradients:
        raise AssertionError("no trainable parameter received a gradient")
    finite_gradients = all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
    if not finite_gradients:
        raise AssertionError("non-finite gradient in full-prefix smoke test")

    result = {
        "kind": "full_prefix_training_smoke",
        "checkpoint": str(checkpoint),
        "dataset_index": selected_index,
        "history_blocks": actual_history_blocks,
        "causal_mode": model.causal_mode,
        "history_training_mode": model.history_training_mode,
        "loss": float(loss.detach().float().item()),
        "metrics": metrics,
        "forward_s": forward_s,
        "backward_s": backward_s,
        "peak_gpu_allocated_gib": float(
            torch.cuda.max_memory_allocated(torch.device(device)) / 2**30
        ),
        "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
        "parameters_with_grad": int(sum(parameter.numel() for parameter in trainable if parameter.grad is not None)),
        "finite_gradients": finite_gradients,
    }
    if native_reference is not None:
        incremental_values = {
            "loss": result["loss"],
            "loss_video": float(metrics["loss_video_d30"]),
            "loss_action": float(metrics["loss_action_d30"]),
        }
        deltas = {
            key: abs(incremental_values[key] - native_reference[key])
            for key in incremental_values
        }
        result["native_reference"] = native_reference
        result["native_absolute_delta"] = deltas
        if target_history_blocks == 0 and max(deltas.values()) > 5.0e-3:
            raise AssertionError(
                "H0 incremental/native loss mismatch exceeds BF16 tolerance: "
                f"incremental={incremental_values} native={native_reference} "
                f"delta={deltas}"
            )
        if (
            model.causal_mode == "action_aggregator"
            and deltas["loss_video"] > 5.0e-3
        ):
            raise AssertionError(
                "action_aggregator video path must remain native-equivalent because "
                "historical K/V is consumed only by ActionDiT: "
                f"incremental={incremental_values} native={native_reference} "
                f"delta={deltas}"
            )
    output_path = output_dir / "full_prefix_smoke.json"
    _write_json(output_path, result)
    logger.info("Full-prefix smoke complete: %s", output_path)
    return result


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(log_level=logging.INFO, is_main_process=True)
    run_smoke(cfg)


if __name__ == "__main__":
    main()
