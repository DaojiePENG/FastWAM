import logging
import math
import json
import inspect
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, LambdaLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(self.model)
        # File checkpoints must be loaded before the optimizer and DeepSpeed
        # FP32 master weights are created. Otherwise the first optimizer step
        # can overwrite the newly loaded BF16 parameters with stale masters.
        if self.resume and not Path(str(self.resume)).is_dir():
            self._load_resume_weights_before_optimizer()
        group_getter = getattr(self.model, "optimizer_parameter_groups", None)
        if group_getter is None:
            trainable_params = [
                parameter for parameter in self.model.parameters() if parameter.requires_grad
            ]
        else:
            trainable_params = group_getter(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        total_count = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        logger.info(
            "Trainable parameters: %.3f B / %.3f B (%.2f%%); optimizer groups=%s",
            trainable_count / 1e9,
            total_count / 1e9,
            100.0 * trainable_count / total_count,
            [
                {
                    "name": group.get("group_name", f"group_{index}"),
                    "lr": group["lr"],
                    "params": sum(parameter.numel() for parameter in group["params"]),
                }
                for index, group in enumerate(self.optimizer.param_groups)
            ],
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        self.anchor_batch_contract = self.train_sampler.anchor_batch_contract()
        if self.anchor_batch_contract is not None:
            logger.info(
                "Verified optimizer anchor mixing: %s",
                self.anchor_batch_contract,
            )
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.metric_ema_beta = 0.98
        self.metric_ema: dict[str, float] = {}

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.distributed_training_topology = (
            self._assert_deepspeed_training_topology()
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_full_training_state_after_prepare()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )
        self.wandb_run.config.update(
            {
                "derived/train_dataset_size": len(self.train_dataset),
                "derived/world_size": self.accelerator.num_processes,
                "derived/effective_global_batch_size": (
                    self.batch_size
                    * self.accelerator.num_processes
                    * self.gradient_accumulation_steps
                ),
                "derived/optimizer_steps": self.max_steps,
                "derived/loss_ema_beta": self.metric_ema_beta,
                **(
                    {
                        f"derived/{key}": value
                        for key, value in self.anchor_batch_contract.items()
                    }
                    if self.anchor_batch_contract is not None
                    else {}
                ),
            },
            allow_val_change=True,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_deepspeed_training_topology(self) -> dict[str, int] | None:
        """Verify the topology resolved by the prepared DeepSpeed engine."""

        if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
            return None

        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        config = None if plugin is None else getattr(plugin, "deepspeed_config", None)
        try:
            configured_zero_stage = int(config["zero_optimization"]["stage"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "DeepSpeed plugin is missing a concrete zero_optimization.stage"
            ) from error

        accessor_names = {
            "micro_batch_size_per_gpu": "train_micro_batch_size_per_gpu",
            "gradient_accumulation_steps": "gradient_accumulation_steps",
            "global_batch_size": "train_batch_size",
            "zero_stage": "zero_optimization_stage",
        }
        actual: dict[str, int] = {}
        for field, accessor_name in accessor_names.items():
            accessor = getattr(self.model, accessor_name, None)
            if not callable(accessor):
                raise RuntimeError(
                    "prepared DeepSpeed engine is missing topology accessor "
                    f"{accessor_name}"
                )
            try:
                actual[field] = int(accessor())
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"DeepSpeed topology accessor {accessor_name} returned "
                    "a non-integer value"
                ) from error

        expected = {
            "micro_batch_size_per_gpu": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "global_batch_size": (
                self.batch_size
                * self.gradient_accumulation_steps
                * self.accelerator.num_processes
            ),
            "zero_stage": configured_zero_stage,
        }
        mismatches = {
            field: (expected[field], actual[field])
            for field in expected
            if actual[field] != expected[field]
        }
        if mismatches:
            details = ", ".join(
                f"{field}: expected={expected_value} actual={actual_value}"
                for field, (expected_value, actual_value) in mismatches.items()
            )
            raise RuntimeError(f"DeepSpeed training topology mismatch: {details}")

        logger.info(
            "Verified DeepSpeed topology: micro_batch_per_gpu=%d "
            "grad_accum=%d global_batch=%d world_size=%d zero_stage=%d",
            actual["micro_batch_size_per_gpu"],
            actual["gradient_accumulation_steps"],
            actual["global_batch_size"],
            self.accelerator.num_processes,
            actual["zero_stage"],
        )
        return actual

    def _clip_and_validate_gradient_norm(self) -> float:
        """Clip gradients and make every rank fail before stepping on non-finite norms."""

        grad_norm = self.accelerator.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm
        )
        local_grad_norm = torch.as_tensor(
            grad_norm,
            device=self.accelerator.device,
            dtype=torch.float32,
        ).detach()
        if local_grad_norm.numel() != 1:
            raise RuntimeError(
                "clip_grad_norm_ must return one scalar per rank, got "
                f"shape={tuple(local_grad_norm.shape)}"
            )
        gathered_grad_norms = self.accelerator.gather(
            local_grad_norm.reshape(1)
        ).reshape(-1)
        if not bool(torch.isfinite(gathered_grad_norms).all().item()):
            values = gathered_grad_norms.cpu().tolist()
            raise FloatingPointError(
                "non-finite clipped gradient norm across ranks; "
                f"rank_values={values}"
            )
        return float(gathered_grad_norms.mean().item())

    def _optimizer_step_with_validated_gradients(self) -> float:
        global_grad_norm = self._clip_and_validate_gradient_norm()
        self.optimizer.step()
        return global_grad_norm

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            # Multiplicative decay preserves the LR ratio between the
            # full-action and higher-LR video-LoRA optimizer groups.
            def cosine_multiplier(step: int) -> float:
                progress = min(max(step, 0), remaining_steps) / remaining_steps
                return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

            main_scheduler = LambdaLR(self.optimizer, lr_lambda=cosine_multiplier)
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _load_resume_weights_before_optimizer(self):
        """Load a file checkpoint before optimizer and DeepSpeed initialization."""
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        if resume_path.is_dir():
            return
        logger.info(
            "Preloading weight checkpoint before optimizer/DeepSpeed initialization: %s",
            resume,
        )
        payload = self.model.load_checkpoint(str(resume_path), optimizer=None)
        if self.accelerator.is_main_process:
            logger.info(
                "Weight checkpoint payload preloaded: step=%s keys=%s",
                payload.get("step", None) if isinstance(payload, dict) else None,
                sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
            )
        logger.warning(
            "Preloaded .pt model weights before optimizer/DeepSpeed initialization; "
            "optimizer/scheduler/global step were not restored."
        )

    def _resume_full_training_state_after_prepare(self):
        """Restore a directory checkpoint after model and optimizer preparation."""
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        if not resume_path.is_dir():
            return
        logger.info("Resuming full training state from directory: %s", resume)
        self.load_training_state(str(resume_path))

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model):
        model.eval()
        model.requires_grad_(False)
        trainable_configurer = getattr(model, "configure_trainable_parameters", None)
        if trainable_configurer is None:
            model.dit.train()
            model.dit.requires_grad_(True)
        else:
            trainable_configurer()
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)
        auxiliary_getter = getattr(model, "auxiliary_trainable_modules", None)
        if auxiliary_getter is not None:
            for module in auxiliary_getter():
                module.train()
                module.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        sample_was_unbatched = video.ndim == 4
        if sample_was_unbatched:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        result = {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }
        # Preserve model-specific supervision instead of silently reducing an
        # extended dataset item to FastWAM's six legacy fields.  LeapBot relies
        # on this for the complete real observation/action prefix, its padding
        # masks, absolute episode clocks, and full-prefix assertion.  The
        # validation dataset is normally indexed one item at a time, hence all
        # additional tensors receive the same leading batch dimension as video.
        converted = {
            "video",
            "prompt",
            "action",
            "proprio",
            "context",
            "context_mask",
        }
        for key, value in sample.items():
            if key in converted:
                continue
            if isinstance(value, torch.Tensor) and sample_was_unbatched:
                value = value.unsqueeze(0)
            result[key] = value
        return result

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()

        if "history_video" in sample:
            # The generic FastWAM `infer()` path below generates video without
            # LeapBot's observation->executed-action memory state machine.  Its
            # action/video metrics would therefore be mislabeled H=0 results.
            # Keep the correctly history-conditioned incremental loss here; policy
            # metrics belong to the external LIBERO memory rollout evaluator.
            gathered_val_loss = self.accelerator.gather_for_metrics(
                torch.tensor(
                    [val_loss], device=self.accelerator.device, dtype=torch.float32
                )
            )
            if was_dit_training:
                self._set_dit_only_train_mode()
            return {
                "val_loss": float(gathered_val_loss.mean().item()),
                "history_conditioned": True,
                "rollout_metrics_skipped": True,
            }
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    @staticmethod
    def _run_contract_metadata() -> dict[str, str]:
        metadata = {}
        contract = os.environ.get("LEAPBOT_RUN_CONTRACT_SHA256")
        commit = os.environ.get("LEAPBOT_CODE_COMMIT")
        if contract:
            metadata["run_contract_sha256"] = contract
        if commit:
            metadata["code_commit"] = commit
        return metadata

    @staticmethod
    def _validate_resume_run_contract(payload: dict) -> None:
        expected = Wan22Trainer._run_contract_metadata()
        for key, value in expected.items():
            actual = payload.get(key)
            if actual != value:
                raise ValueError(
                    "training-state run contract mismatch for "
                    f"{key}: checkpoint={actual!r} current={value!r}"
                )

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        epoch = int(self.epoch)
        batch_in_epoch = int(self.batch_in_epoch)
        micro_batches_per_epoch = int(len(self.train_loader))
        if batch_in_epoch == micro_batches_per_epoch:
            # Store the next unconsumed position.  Accelerate does not advance
            # DataLoaderShard.iteration when an iterator is already empty, so an
            # unnormalized end-of-epoch cursor would replay the completed epoch.
            epoch += 1
            batch_in_epoch = 0
        elif batch_in_epoch < 0 or batch_in_epoch > micro_batches_per_epoch:
            raise RuntimeError(
                "invalid dataloader cursor while saving: "
                f"batch={batch_in_epoch} length={micro_batches_per_epoch}"
            )
        payload = {
            "global_step": int(self.global_step),
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "dataset_length": int(len(self.train_dataset)),
            "batch_size_per_process": int(self.batch_size),
            "num_processes": int(self.accelerator.num_processes),
            "micro_batches_per_epoch": micro_batches_per_epoch,
            **self._run_contract_metadata(),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        state_file = Path(state_dir) / "trainer_state.json"
        payload = None
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self._validate_resume_run_contract(payload)
        elif self._run_contract_metadata():
            raise ValueError(
                "contract-bound training state is missing trainer_state.json: "
                f"{state_file}"
            )

        # Validate cheap metadata before loading optimizer/model shards so a
        # stale output directory cannot partially mutate the current run.
        self.accelerator.load_state(input_dir=state_dir)
        if payload is not None:
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                expected_resume_contract = {
                    "dataset_length": int(len(self.train_dataset)),
                    "batch_size_per_process": int(self.batch_size),
                    "num_processes": int(self.accelerator.num_processes),
                    "micro_batches_per_epoch": int(len(self.train_loader)),
                }
                for key, expected in expected_resume_contract.items():
                    if key in payload and int(payload[key]) != expected:
                        raise ValueError(
                            "dataloader resume contract changed for "
                            f"{key}: checkpoint={payload[key]} current={expected}"
                        )
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                micro_batches_per_epoch = int(len(self.train_loader))
                if self.batch_in_epoch == micro_batches_per_epoch:
                    # Backward compatibility for states written before cursor
                    # normalization was introduced.
                    self.epoch += 1
                    self.batch_in_epoch = 0
                elif self.batch_in_epoch < 0 or self.batch_in_epoch > micro_batches_per_epoch:
                    raise ValueError(
                        "invalid dataloader cursor in trainer state: "
                        f"batch={self.batch_in_epoch} length={micro_batches_per_epoch}"
                    )
                # DataLoaderShard uses relative iterations 0,1,... after a
                # restart.  Pin that relative clock to zero and add the saved
                # absolute epoch in our sampler.
                set_loader_epoch = getattr(self.train_loader, "set_epoch", None)
                if set_loader_epoch is not None:
                    set_loader_epoch(0)
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        accumulated_loss_sum = 0.0
        accumulated_sample_count = 0
        accumulated_metric_sums: dict[str, float] = {}
        accumulated_metric_counts: dict[str, float] = {}
        accumulated_metric_maxima: dict[str, float] = {}

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)
                step_aware_model = self.accelerator.unwrap_model(self.model)
                if hasattr(step_aware_model, "set_training_step"):
                    step_aware_model.set_training_step(self.global_step)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                sample_video = sample.get("video") if isinstance(sample, dict) else None
                if not isinstance(sample_video, torch.Tensor) or sample_video.ndim < 1:
                    raise ValueError(
                        "training samples must contain a batched `video` tensor so "
                        "gradient-accumulation metrics can be weighted exactly"
                    )
                micro_batch_samples = int(sample_video.shape[0])
                if micro_batch_samples <= 0:
                    raise ValueError("training micro-batch must contain at least one sample")
                accumulated_loss_sum += float(loss.detach()) * micro_batch_samples
                accumulated_sample_count += micro_batch_samples
                metric_weights = {
                    key.removeprefix("__metric_weight__"): float(value)
                    for key, value in loss_dict.items()
                    if key.startswith("__metric_weight__")
                }
                for key, value in loss_dict.items():
                    if key.startswith("__metric_weight__"):
                        continue
                    scalar = float(value)
                    if key.endswith("_max"):
                        accumulated_metric_maxima[key] = max(
                            accumulated_metric_maxima.get(key, scalar), scalar
                        )
                    else:
                        metric_count = float(
                            metric_weights.get(key, micro_batch_samples)
                        )
                        if not math.isfinite(metric_count) or metric_count < 0:
                            raise ValueError(
                                f"metric weight for {key!r} must be finite and non-negative"
                            )
                        accumulated_metric_sums[key] = (
                            accumulated_metric_sums.get(key, 0.0)
                            + scalar * metric_count
                        )
                        accumulated_metric_counts[key] = (
                            accumulated_metric_counts.get(key, 0.0)
                            + metric_count
                        )
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    global_grad_norm = (
                        self._optimizer_step_with_validated_gradients()
                    )
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    local_loss_stats = torch.tensor(
                        [accumulated_loss_sum, float(accumulated_sample_count)],
                        device=loss.device,
                        dtype=torch.float64,
                    )
                    gathered_loss_stats = self.accelerator.gather(
                        local_loss_stats.reshape(1, 2)
                    ).reshape(-1, 2)
                    global_loss = float(
                        gathered_loss_stats[:, 0].sum().item()
                        / gathered_loss_stats[:, 1].sum().clamp_min(1.0).item()
                    )
                    global_loss_metrics = {}
                    for key, local_sum in accumulated_metric_sums.items():
                        local_metric_stats = torch.tensor(
                            [local_sum, accumulated_metric_counts[key]],
                            device=loss.device,
                            dtype=torch.float64,
                        )
                        gathered_metric_stats = self.accelerator.gather(
                            local_metric_stats.reshape(1, 2)
                        ).reshape(-1, 2)
                        global_metric_count = gathered_metric_stats[:, 1].sum()
                        if float(global_metric_count.item()) > 0:
                            global_loss_metrics[key] = float(
                                gathered_metric_stats[:, 0].sum().item()
                                / global_metric_count.item()
                            )
                    for key, local_maximum in accumulated_metric_maxima.items():
                        gathered_metric = self.accelerator.gather(
                            torch.tensor(
                                local_maximum, device=loss.device, dtype=torch.float32
                            ).reshape(1)
                        )
                        global_loss_metrics[key] = float(gathered_metric.max().item())
                    accumulated_loss_sum = 0.0
                    accumulated_sample_count = 0
                    accumulated_metric_sums.clear()
                    accumulated_metric_counts.clear()
                    accumulated_metric_maxima.clear()
                    ema_inputs = {"loss": global_loss, **global_loss_metrics}
                    for key, value in ema_inputs.items():
                        previous = self.metric_ema.get(key, value)
                        self.metric_ema[key] = (
                            self.metric_ema_beta * previous
                            + (1.0 - self.metric_ema_beta) * value
                        )
                    if self.accelerator.device.type == "cuda":
                        peak_allocated = torch.tensor(
                            torch.cuda.max_memory_allocated(self.accelerator.device),
                            device=loss.device,
                            dtype=torch.float64,
                        )
                        peak_reserved = torch.tensor(
                            torch.cuda.max_memory_reserved(self.accelerator.device),
                            device=loss.device,
                            dtype=torch.float64,
                        )
                        global_peak_allocated_gib = float(
                            self.accelerator.gather(peak_allocated).max().item() / 2**30
                        )
                        global_peak_reserved_gib = float(
                            self.accelerator.gather(peak_reserved).max().item() / 2**30
                        )
                    else:
                        global_peak_allocated_gib = 0.0
                        global_peak_reserved_gib = 0.0

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        samples_per_second = (
                            steps_per_sec
                            * self.batch_size
                            * self.accelerator.num_processes
                            * self.gradient_accumulation_steps
                        )
                        description += (
                            "lr=%.2e grad_norm=%.4f speed=%.2f step/s, %.2f samples/s "
                            "peak_allocated=%.2fGiB peak_reserved=%.2fGiB eta=%s"
                        ) % (
                            current_lr,
                            global_grad_norm,
                            steps_per_sec,
                            samples_per_second,
                            global_peak_allocated_gib,
                            global_peak_reserved_gib,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": samples_per_second,
                            "performance/peak_gpu_allocated_gib": global_peak_allocated_gib,
                            "performance/peak_gpu_reserved_gib": global_peak_reserved_gib,
                        }
                        for index, group in enumerate(self.optimizer.param_groups):
                            group_name = str(group.get("group_name", f"group_{index}"))
                            wandb_payload[f"train/lr_{group_name}"] = float(group["lr"])
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        for key, value in self.metric_ema.items():
                            wandb_payload[f"train/{key}_ema"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                            )
                            if "psnr_rd" in metrics:
                                description += " infer_psnr=%.4f infer_ssim=%.4f" % (
                                    metrics["psnr_rd"],
                                    metrics["ssim_rd"],
                                )
                            elif metrics.get("rollout_metrics_skipped", False):
                                description += " memory_rollout_metrics=external_only"
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                            }
                            for key in (
                                "psnr_rg",
                                "ssim_rg",
                                "psnr_rd",
                                "ssim_rd",
                                "psnr_dg",
                                "ssim_dg",
                            ):
                                if key in metrics:
                                    eval_payload[f"eval/{key}"] = float(metrics[key])
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    ckpt_info = None
                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        if ckpt_info is None:
                            ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
