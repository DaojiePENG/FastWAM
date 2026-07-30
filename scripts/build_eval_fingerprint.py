#!/usr/bin/env python3
"""Compose an evaluation config and atomically write its schema-3 fingerprint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leapbot_va.eval_contract import (  # noqa: E402
    build_result_contract,
    build_runtime_contract,
    resolve_dataset_stats_path,
    resolve_libero_task_and_initial_states,
)
from leapbot_va.eval_fingerprint import (  # noqa: E402
    atomic_write_json,
    build_evaluation_fingerprint,
    sha256_file,
)


def _register_resolvers() -> None:
    resolvers = {
        "eval": eval,
        "max": lambda x: max(x),
        "split": lambda s, idx: s.split("/")[int(idx)],
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint to hash; defaults to the composed cfg.ckpt.",
    )
    parser.add_argument(
        "--checkpoint-sha256",
        help="Previously computed digest (avoids hashing in this preflight only).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="The exact Hydra overrides that will be passed to eval_libero_single.py.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    _register_resolvers()
    with initialize_config_dir(
        config_dir=str((ROOT / "configs").resolve()), version_base="1.3"
    ):
        cfg = compose(
            config_name=args.config_name,
            overrides=list(args.overrides),
            return_hydra_config=True,
        )

    choices = OmegaConf.to_container(cfg.hydra.runtime.choices, resolve=True)
    dataset_stats_path = resolve_dataset_stats_path(cfg)
    _, task, initial_states_path = resolve_libero_task_and_initial_states(cfg)
    runtime_contract = build_runtime_contract(
        cfg,
        config_name=args.config_name,
        hydra_choices=choices,
        dataset_stats_path=dataset_stats_path,
        source_root=ROOT,
    )
    result_contract = build_result_contract(
        cfg, task=task, initial_states_path=initial_states_path
    )

    configured_checkpoint = None if cfg.ckpt is None else Path(str(cfg.ckpt))
    checkpoint_path = args.checkpoint or configured_checkpoint
    if checkpoint_path is None:
        raise ValueError("pass --checkpoint or compose a non-null cfg.ckpt")
    if (
        args.checkpoint is not None
        and configured_checkpoint is not None
        and args.checkpoint.resolve() != configured_checkpoint.resolve()
    ):
        raise ValueError(
            "--checkpoint must identify the same file as composed cfg.ckpt: "
            f"argument={args.checkpoint.resolve()} "
            f"cfg={configured_checkpoint.resolve()}"
        )
    checkpoint_sha256 = (
        str(args.checkpoint_sha256)
        if args.checkpoint_sha256 is not None
        else sha256_file(checkpoint_path)
    )
    fingerprint = build_evaluation_fingerprint(
        checkpoint_sha256=checkpoint_sha256,
        runtime_contract=runtime_contract,
        result_contract=result_contract,
    )
    atomic_write_json(args.output, fingerprint)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
