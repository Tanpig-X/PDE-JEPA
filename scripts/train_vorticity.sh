#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export DATA_ROOT="${DATA_ROOT:-$PWD/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo 'NPROC_PER_NODE must be a positive integer' >&2
  exit 2
fi
stage="${1:-all}"
if (($#)); then shift; fi

run_stage() {
  local current="$1"
  shift
  case "$current" in
    pretrain|cooldown)
      if ((32 % NPROC_PER_NODE)); then echo 'NPROC_PER_NODE must divide 32' >&2; exit 2; fi
      torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m pde_jepa.train \
        --task vorticity --stage "$current" --set "data.batch_size=$((32 / NPROC_PER_NODE))" "$@"
      ;;
    pag)
      if ((16 % NPROC_PER_NODE)); then echo 'NPROC_PER_NODE must divide 16' >&2; exit 2; fi
      torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m pde_jepa.train \
        --task vorticity --stage pag --set "data.batch_size=$((16 / NPROC_PER_NODE))" "$@"
      ;;
    psp|decoder)
      python -m pde_jepa.train --task vorticity --stage "$current" "$@"
      ;;
    *) echo "Unknown stage: $current (pretrain, cooldown, pag, psp, decoder, all)" >&2; exit 2 ;;
  esac
}

if [[ "$stage" == all ]]; then
  for current in pretrain cooldown pag psp decoder; do
    run_stage "$current" "$@"
  done
else
  run_stage "$stage" "$@"
fi
