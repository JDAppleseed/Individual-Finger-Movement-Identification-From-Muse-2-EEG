#!/usr/bin/env bash
set -euo pipefail
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python 3_evaluate_model.py "$@"
