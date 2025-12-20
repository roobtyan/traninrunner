#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _CUDA_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  GPUS="${#_CUDA_ARR[@]}"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    GPUS="1"
  fi
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$GPUS}"

LAUNCH=(torchrun)
if ! command -v torchrun >/dev/null 2>&1; then
  LAUNCH=(python -m torch.distributed.run)
fi

if [[ -n "${NNODES:-}" ]]; then
  : "${NODE_RANK:?NODE_RANK is required when NNODES is set}"
  : "${MASTER_ADDR:?MASTER_ADDR is required when NNODES is set}"
  : "${MASTER_PORT:?MASTER_PORT is required when NNODES is set}"
  exec "${LAUNCH[@]}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --module trainrunner.run \
    -- "$@"
else
  exec "${LAUNCH[@]}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --standalone \
    --module trainrunner.run \
    -- "$@"
fi

