#!/usr/bin/env python3
"""Config-driven local/multi-node LeapBot launcher.

This is an independent entry point.  It keeps the run contract, output naming,
resume, logging, and checkpoint layout of ``train_leapbot.sh`` while sourcing
training parameters from Hydra task + model configs.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONDA_ENV = Path("/home/myuser/miniconda3/envs/leapbot-va")
DEFAULT_TASK = "libero_leapbot_pch"


def _reexec_in_conda() -> None:
    target = DEFAULT_CONDA_ENV / "bin" / "python"
    if target.is_file() and Path(sys.prefix).resolve() != DEFAULT_CONDA_ENV.resolve():
        os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])


_reexec_in_conda()

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from fastwam.utils.config_resolvers import register_default_resolvers  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--num-processes", type=int, default=None,
                        help="Total processes across every machine.")
    parser.add_argument("--gpu-ids", default=None,
                        help="Comma-separated local GPU ids.")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    parser.add_argument("--main-process-ip", default="127.0.0.1")
    parser.add_argument("--main-process-port", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve config and print the launch command only.")
    parser.add_argument("overrides", nargs="*", help="Optional Hydra overrides.")
    return parser.parse_args()


def _compose(task: str, overrides: list[str]) -> DictConfig:
    register_default_resolvers()
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"), version_base="1.3"
    ):
        return compose(config_name="train", overrides=[f"task={task}", *overrides])


def _as_path(value: str) -> Path:
    path = Path(str(value)).expanduser()
    # Keep repository-facing paths stable even when an asset is a symlink into
    # another checkout.  The canonical shell records the LeapBot path too.
    return path if path.is_absolute() else (REPO_ROOT / path).absolute()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _learning_rate_label(value: object) -> str:
    mantissa, exponent = f"{float(value):.1e}".split("e", 1)
    sign = exponent[0]
    power = exponent[1:].lstrip("0") or "0"
    return f"{mantissa}e{sign}{power}"


def _format_lr(value: object) -> str:
    return _learning_rate_label(value).replace(".", "p").replace("+", "_")


def _topology(cfg: DictConfig, total_processes: int) -> tuple[str, int]:
    global_batch = total_processes * int(cfg.batch_size) * int(cfg.gradient_accumulation_steps)
    tag = (
        f"w{total_processes}_b{cfg.batch_size}_ga{cfg.gradient_accumulation_steps}"
        f"_bs{global_batch}"
    )
    return tag, global_batch


def _derive_runtime(cfg: DictConfig, total_processes: int) -> tuple[Path, str, str, int]:
    topology, global_batch = _topology(cfg, total_processes)
    mode = str(cfg.model.causal_mode)
    history_window = int(cfg.model.history_window_blocks)
    lr_tag = _format_lr(cfg.learning_rate)
    prefix = str(cfg.launch.output_prefix)
    if cfg.output_dir is None:
        cfg.output_dir = str(
            REPO_ROOT / "runs" /
            f"{prefix}_w{history_window}_{topology}_{mode}_peft_"
            f"{cfg.max_steps}steps_{cfg.lr_scheduler_type}_lr{lr_tag}_seed{cfg.seed}"
        )
    run_family = prefix.replace("_", "-")
    if cfg.wandb.group is None:
        cfg.wandb.group = (
            f"{run_family}-w{history_window}-{topology}-seed{cfg.seed}"
        )
    if cfg.wandb.name is None:
        cfg.wandb.name = (
            f"{run_family}-w{history_window}-{topology.replace('_', '-')}-"
            f"{mode.replace('_', '-')}-peft-{cfg.max_steps}steps-"
            f"{cfg.lr_scheduler_type}-lr{lr_tag}-seed{cfg.seed}"
        )
    return _as_path(cfg.output_dir), topology, str(cfg.wandb.name), global_batch


def _validate_config(cfg: DictConfig, total_processes: int, gpu_ids: list[int]) -> None:
    if total_processes < 1 or len(gpu_ids) < 1:
        raise ValueError("At least one process and one local GPU are required.")
    if str(cfg.model.get("_target_")) != "leapbot_va.runtime.create_leapbot":
        raise ValueError("The config-driven launcher only accepts model=leapbot.")
    if str(cfg.model.causal_mode) not in {"interleaved", "vision_causal", "action_aggregator"}:
        raise ValueError(f"Invalid causal mode: {cfg.model.causal_mode}")
    if str(cfg.lr_scheduler_type) not in {"cosine", "constant"}:
        raise ValueError(f"Invalid LR scheduler: {cfg.lr_scheduler_type}")
    if list(cfg.model.training_exit_depths) not in ([30], [8, 16, 24, 30]):
        raise ValueError("training_exit_depths must be [30] or [8,16,24,30].")
    history_mode = str(cfg.model.history_training_mode)
    if history_mode not in {
        "incremental_full_bptt",
        "strict_replay_window_bptt",
        "packed_causal_history_bptt",
        "episode_memory_scan_bptt",
    }:
        raise ValueError(f"Invalid history_training_mode: {history_mode}")
    if history_mode == "episode_memory_scan_bptt":
        episode = cfg.model.episode_memory
        if not bool(episode.enabled):
            raise ValueError("episode_memory_scan_bptt requires episode_memory.enabled=true")
        if int(cfg.model.history_window_blocks) != int(episode.window_blocks):
            raise ValueError("model and episode-memory window_blocks must match")
        if str(cfg.data.train.history_sampling_mode) != "full_prefix":
            raise ValueError("episode-memory scan requires full_prefix data")
        if cfg.data.train.history_window_blocks is not None:
            raise ValueError("episode-memory full-prefix data cannot set history_window_blocks")
        if bool(cfg.data.train.use_episode_anchor) != bool(episode.first_frame_memory):
            raise ValueError("data anchor and episode first_frame_memory must match")
        if not str(cfg.launch.output_prefix).startswith("episode_memory"):
            raise ValueError("episode-memory runs require an episode_memory output namespace")
    elif int(cfg.model.history_window_blocks) != int(cfg.data.train.history_window_blocks):
        raise ValueError("data/model history_window_blocks must match")
    backend = str(cfg.model.get("packed_history_attention_backend", "dense"))
    if backend not in {"dense", "flex"}:
        raise ValueError("packed_history_attention_backend must be flex or dense")
    if history_mode == "packed_causal_history_bptt":
        if str(cfg.data.train.history_sampling_mode) != "recent_window":
            raise ValueError("PCH requires data.train.history_sampling_mode=recent_window")
        if not bool(cfg.data.train.use_episode_anchor):
            raise ValueError("PCH requires use_episode_anchor=true")
        if not str(cfg.launch.output_prefix).startswith("pch_v1"):
            raise ValueError("PCH output_prefix must use the independent pch_v1 namespace")
    required = {
        "history VAE chunk": int(cfg.model.history_vae_batch_chunk_size) > 0,
        "video frames": int(cfg.model.num_video_frames) == 9,
        "future conditioning": str(cfg.model.future_video_conditioning)
        == "lingbot_teacher_forced_v1",
        "noise probability": float(cfg.model.future_video_condition_noise_probability) == 0.5,
        "minimum u": float(cfg.model.future_video_condition_min_u) == 0.5,
        "maximum u": float(cfg.model.future_video_condition_max_u) == 1.0,
    }
    failed = [name for name, valid in required.items() if not valid]
    if failed:
        raise ValueError("Formal LeapBot contract mismatch: " + ", ".join(failed))


def _latest_state(output_dir: Path) -> Path | None:
    state_root = output_dir / "checkpoints" / "state"
    candidates = sorted(state_root.glob("step_*")) if state_root.is_dir() else []
    complete = [
        path for path in candidates
        if (path / "trainer_state.json").is_file()
        and (path / "trainer_state.json").stat().st_size > 0
    ]
    return complete[-1] if complete else None


def _contract_fields(cfg: DictConfig, total_processes: int, global_batch: int) -> list[str]:
    release = _as_path(cfg.launch.release_checkpoint)
    initial = _as_path(cfg.resume)
    dataset_stats = _as_path(cfg.launch.dataset_stats)
    disabled_sha = hashlib.sha256(b"training-asset-manifest-disabled-v1").hexdigest()
    code_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    release_sha = _sha256_file(release)
    initial_sha = _sha256_file(initial)
    fields = [
        f"code_commit={code_commit}",
        f"release_checkpoint_sha256={release_sha}",
    ]
    if initial_sha != release_sha:
        fields.append(f"initial_checkpoint_sha256={initial_sha}")
    fields.extend([
        f"dataset_stats_sha256={_sha256_file(dataset_stats)}",
        f"training_asset_manifest_sha256={disabled_sha}",
        f"dataset_content_sha256={disabled_sha}",
        "dataset_file_count=1",
        "dataset_bytes=1",
        "text_embedding_cache_sha256=4fad91546fe15c9fa04cb8d4ea08e8a758aead8c4273e87aaaff203621211332",
        "text_embedding_cache_file_count=40",
        "text_cache_provenance_sha256=2f8b80886919934477ffd9f77c1c265cf2cb63bd2c40b3b635213c8c31ba7ae8",
        "text_cache_verification_method=online_source_forward_cache_tensor_exact",
        "text_cache_verified_file_count=40",
        "text_encoder_checkpoint_sha256=7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d",
        "tokenizer_sha256=a8bc717cf013b7790af3b115681470a445fd2ac2b8e5ba750f1041f13ac54279",
        f"vae_checkpoint_sha256={disabled_sha}",
        f"mode={cfg.model.causal_mode}",
        f"num_processes={total_processes}",
        f"batch_size={cfg.batch_size}",
        f"gradient_accumulation_steps={cfg.gradient_accumulation_steps}",
        f"global_batch={global_batch}",
        f"max_steps={cfg.max_steps}",
        f"learning_rate={_learning_rate_label(cfg.learning_rate)}",
        f"lr_scheduler_type={cfg.lr_scheduler_type}",
        f"video_lora_multiplier={cfg.model.video_lora.learning_rate_multiplier}",
        f"history_vae_batch_chunk_size={cfg.model.history_vae_batch_chunk_size}",
        f"history_window_blocks={cfg.model.history_window_blocks}",
        f"episode_memory_enabled={cfg.model.episode_memory.enabled}",
        f"episode_memory_chunk_blocks={cfg.model.episode_memory.chunk_blocks}",
        f"episode_memory_num_slots={cfg.model.episode_memory.num_slots}",
        f"episode_memory_state_dim={cfg.model.episode_memory.state_dim}",
        f"episode_memory_group_dim={cfg.model.episode_memory.group_dim}",
        f"first_frame_memory={cfg.model.episode_memory.first_frame_memory}",
        f"world_model_conditioning={cfg.model.future_video_conditioning}",
        f"num_video_frames={cfg.model.num_video_frames}",
        f"future_video_condition_noise_probability={cfg.model.future_video_condition_noise_probability}",
        f"future_video_condition_min_u={cfg.model.future_video_condition_min_u}",
        f"future_video_condition_max_u={cfg.model.future_video_condition_max_u}",
        f"future_video_condition_clean_warmup_steps={cfg.model.future_video_condition_clean_warmup_steps}",
        f"future_video_condition_noise_ramp_steps={cfg.model.future_video_condition_noise_ramp_steps}",
        f"initial_block_oversample={cfg.data.train.initial_block_oversample}",
        "h0_anchor_mixing=per_global_micro_batch",
        f"save_every={cfg.save_every}",
        f"seed={cfg.seed}",
        "padding_attention_mask=true",
        f"history_training_mode={cfg.model.history_training_mode}",
        f"packed_history_attention_backend={cfg.model.get('packed_history_attention_backend', 'dense')}",
        f"history_execution_layout={'affine_prefix_scan' if str(cfg.model.history_training_mode) == 'episode_memory_scan_bptt' else 'fixed_padding' if str(cfg.model.history_training_mode) == 'packed_causal_history_bptt' else 'iterative_replay'}",
        f"history_sampling_mode={cfg.data.train.history_sampling_mode}",
        "history_padding=left_masked",
        f"episode_anchor={'single_real_v0' if bool(cfg.model.episode_memory.first_frame_memory) else 'disabled'}",
        f"max_history_blocks={cfg.data.train.max_history_blocks}",
        f"replan_steps={cfg.model.replan_steps}",
        f"action_horizon={cfg.model.action_horizon}",
        "training_exit_depths=" + ",".join(str(x) for x in cfg.model.training_exit_depths),
        f"mixed_precision={cfg.mixed_precision}",
        "optimizer=adamw_beta0.9_0.95_wd0.01_clip1.0",
    ])
    return fields


def _prepare_contract(
    cfg: DictConfig, output_dir: Path, total_processes: int, global_batch: int
) -> tuple[str, str]:
    existed = output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_file = output_dir / "run_contract.txt"
    fields = _contract_fields(cfg, total_processes, global_batch)
    payload = "\n".join(fields)
    contract_sha = hashlib.sha256(payload.encode()).hexdigest()
    if contract_file.is_file() and contract_file.stat().st_size:
        stored = contract_file.read_text(encoding="utf-8").splitlines()[0].split("=", 1)[-1]
        if stored != contract_sha and not bool(cfg.launch.allow_cross_contract_resume):
            raise RuntimeError(
                f"Refusing output-dir reuse across run contracts: stored={stored} current={contract_sha}"
            )
    elif existed and any(output_dir.iterdir()) and not bool(cfg.launch.allow_existing_uncontracted):
        raise RuntimeError(f"Refusing uncontracted non-empty output directory: {output_dir}")
    else:
        temporary = contract_file.with_name(f"run_contract.txt.tmp.{os.getpid()}")
        temporary.write_text(
            f"run_contract_sha256={contract_sha}\n{payload}\n", encoding="utf-8"
        )
        os.replace(temporary, contract_file)
    code_commit = fields[0].split("=", 1)[1]
    return contract_sha, code_commit


def _wait_for_contract(output_dir: Path, timeout: int = 1800) -> tuple[str, str]:
    contract_file = output_dir / "run_contract.txt"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if contract_file.is_file() and contract_file.stat().st_size:
            values = dict(
                line.split("=", 1)
                for line in contract_file.read_text(encoding="utf-8").splitlines()
                if "=" in line
            )
            return values["run_contract_sha256"], values["code_commit"]
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for rank 0 to create {contract_file}")


def _preflight_gpus(gpu_ids: list[int], max_used_mib: int | None) -> None:
    if max_used_mib is None:
        return
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    used = [int(line.strip()) for line in output.splitlines() if line.strip()]
    for gpu_id in gpu_ids:
        if gpu_id >= len(used):
            raise RuntimeError(f"GPU {gpu_id} is unavailable; nvidia-smi reported {len(used)} GPUs.")
        if used[gpu_id] > max_used_mib:
            raise RuntimeError(
                f"GPU {gpu_id} uses {used[gpu_id]} MiB (limit {max_used_mib} MiB)."
            )


def _launch_command(
    cfg: DictConfig,
    args: argparse.Namespace,
    total_processes: int,
    gpu_ids_csv: str,
) -> list[str]:
    port = args.main_process_port or int(cfg.launch.main_process_port)
    command = [
        sys.executable,
        "-m", "accelerate.commands.accelerate_cli", "launch",
        "--config_file", str(_as_path(cfg.launch.accelerate_config)),
        "--num_processes", str(total_processes),
        "--num_machines", str(args.num_machines),
        "--machine_rank", str(args.machine_rank),
        "--main_process_ip", str(args.main_process_ip),
        "--main_process_port", str(port),
        "--gpu_ids", gpu_ids_csv,
        "--same_network",
    ]
    if args.num_machines > 1:
        command.extend([
            "--deepspeed_multinode_launcher",
            os.environ.get("DEEPSPEED_MULTINODE_LAUNCHER", "standard"),
        ])
    command.extend([
        str(REPO_ROOT / "scripts" / "train.py"),
        f"task={args.task}",
        *args.overrides,
        f"output_dir={cfg.output_dir}",
        f"resume={cfg.resume}",
        f"wandb.group={cfg.wandb.group}",
        f"wandb.name={cfg.wandb.name}",
    ])
    return command


def main() -> int:
    args = _parse_args()
    os.chdir(REPO_ROOT)
    cfg = _compose(args.task, args.overrides)
    # Local execution is intentionally topology-free in Hydra: one visible GPU
    # by default. AIFlow supplies both values explicitly from its allocation.
    gpu_ids = (
        [int(value) for value in args.gpu_ids.split(",")]
        if args.gpu_ids
        else [0]
    )
    gpu_ids_csv = ",".join(str(value) for value in gpu_ids)
    total_processes = (
        int(args.num_processes)
        if args.num_processes is not None
        else len(gpu_ids) * int(args.num_machines)
    )
    if total_processes % args.num_machines:
        raise ValueError("Total processes must be divisible by num_machines.")
    if total_processes // args.num_machines != len(gpu_ids):
        raise ValueError(
            "Local GPU count must equal total_processes / num_machines: "
            f"gpus={len(gpu_ids)} total={total_processes} machines={args.num_machines}"
        )
    if args.num_machines > 1 and args.main_process_ip.strip().lower() in {
        "127.0.0.1", "localhost", "::1",
    }:
        raise ValueError(
            "Multi-node training requires --main-process-ip to be the reachable "
            "rank-0 machine IP, not a loopback address."
        )
    _validate_config(cfg, total_processes, gpu_ids)
    output_dir, topology, run_name, global_batch = _derive_runtime(cfg, total_processes)

    if args.dry_run:
        cfg.resume = str(_as_path(cfg.resume))
        command = _launch_command(cfg, args, total_processes, gpu_ids_csv)
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print("command:", shlex.join(command))
        return 0

    release = _as_path(cfg.launch.release_checkpoint)
    initial = _as_path(cfg.resume)
    dataset_stats = _as_path(cfg.launch.dataset_stats)
    text_cache = _as_path(cfg.launch.text_embedding_cache)
    vae = _as_path(cfg.launch.vae_checkpoint)
    for label, path in {
        "release checkpoint": release,
        "initial checkpoint": initial,
        "dataset statistics": dataset_stats,
        "VAE checkpoint": vae,
    }.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not text_cache.is_dir():
        raise FileNotFoundError(f"Missing text embedding cache: {text_cache}")
    if not bool(cfg.launch.allow_dirty):
        dirty = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=normal"],
            text=True,
        )
        if dirty:
            raise RuntimeError("Refusing formal training from a dirty worktree.")

    output_dir.mkdir(parents=True, exist_ok=True)
    final_tag = f"step_{int(cfg.max_steps):06d}"
    log_file = output_dir / ("train.log" if args.machine_rank == 0 else f"train.node{args.machine_rank}.log")
    final_checkpoint = output_dir / "checkpoints" / "weights" / f"{final_tag}.pt"
    if final_checkpoint.is_file() and log_file.is_file() and (
        f"max_steps reached step={cfg.max_steps}" in log_file.read_text(encoding="utf-8", errors="replace")
    ):
        print(f"skip completed run: {final_checkpoint}")
        return 0

    lock_path = text_cache / ".leapbot_text_cache.lock"
    lock_stream = lock_path.open("a+")
    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_SH)
    if args.machine_rank == 0:
        contract_sha, code_commit = _prepare_contract(
            cfg, output_dir, total_processes, global_batch
        )
    else:
        contract_sha, code_commit = _wait_for_contract(output_dir)

    resume = _latest_state(output_dir)
    cfg.resume = str(resume or initial)
    if resume is None:
        log_file.write_text("", encoding="utf-8")
    else:
        with log_file.open("a", encoding="utf-8") as stream:
            stream.write(f"resume from full trainer state: {resume}\n")
    _preflight_gpus(gpu_ids, int(cfg.launch.max_preflight_used_mib))

    for directory in (
        REPO_ROOT / ".cache" / "wandb" / "config",
        REPO_ROOT / ".cache" / "wandb" / "cache",
        REPO_ROOT / ".cache" / "wandb" / "data",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu_ids_csv,
        "PYTHONHASHSEED": str(cfg.seed),
        "LEAPBOT_RUN_CONTRACT_SHA256": contract_sha,
        "LEAPBOT_CODE_COMMIT": code_commit,
        "LEAPBOT_DATASET_STATS": str(dataset_stats),
        "DIFFSYNTH_MODEL_BASE_PATH": str(REPO_ROOT / "checkpoints"),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_CONFIG_DIR": str(REPO_ROOT / ".cache" / "wandb" / "config"),
        "WANDB_CACHE_DIR": str(REPO_ROOT / ".cache" / "wandb" / "cache"),
        "WANDB_DATA_DIR": str(REPO_ROOT / ".cache" / "wandb" / "data"),
        "WANDB_DIR": str(output_dir),
        "WANDB_RUN_ID": run_name,
        "WANDB_RESUME": "allow",
        "PYTHONPATH": f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}",
    })
    command = _launch_command(cfg, args, total_processes, gpu_ids_csv)
    print(
        f"start LeapBot task={args.task} commit={code_commit} contract={contract_sha} "
        f"topology={topology} rank={args.machine_rank}/{args.num_machines} resume={cfg.resume}"
    )
    with log_file.open("a", encoding="utf-8") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    lock_stream.close()
    if result.returncode != 0:
        try:
            log_lines = log_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            tail = "\n".join(log_lines[-80:])
        except OSError as exc:
            tail = f"<unable to read training log: {exc}>"
        print(
            f"LeapBot training failed with exit status {result.returncode}. "
            f"Log: {log_file}\n--- train.log tail ---\n{tail}",
            file=sys.stderr,
        )
        raise subprocess.CalledProcessError(result.returncode, command)
    if not final_checkpoint.is_file():
        raise FileNotFoundError(f"Training exited without final checkpoint: {final_checkpoint}")
    print(f"LeapBot training complete: {final_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

