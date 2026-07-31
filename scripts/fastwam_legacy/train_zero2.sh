#!/usr/bin/env bash
set -euo pipefail

# Upstream FastWAM compatibility launcher; not part of LeapBot formal training.

NPROC_PER_NODE="${1:?Usage: bash scripts/fastwam_legacy/train_zero2.sh <nproc_per_node> [hydra_overrides...]}"
shift

EXTRA_ARGS=("$@")
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"

if [[ "$NUM_MACHINES" != "1" ]] || [[ "$MACHINE_RANK" != "0" ]]; then
  printf '%s\n' \
    "This legacy FastWAM wrapper is single-machine only (NNODES=1, NODE_RANK=0)." \
    "Use an external cluster launcher that passes the complete Accelerate/DeepSpeed multi-node topology." \
    >&2
  exit 2
fi
if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  printf 'nproc_per_node must be a positive integer, got: %s\n' \
    "$NPROC_PER_NODE" >&2
  exit 2
fi

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
  esac
done

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

echo "[launch] mode=single_machine nproc_per_node=${NPROC_PER_NODE} run_id=${RUN_ID}"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes "${NPROC_PER_NODE}" \
  scripts/train.py \
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
