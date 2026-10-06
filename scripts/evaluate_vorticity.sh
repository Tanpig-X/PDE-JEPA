#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export DATA_ROOT="${DATA_ROOT:-$PWD/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/outputs}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python -m pde_jepa.train --task vorticity --stage decoder --evaluate-only \
  --checkpoint "$OUTPUT_ROOT/decoder/best.pt" "$@"
