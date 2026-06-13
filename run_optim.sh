#!/usr/bin/env bash
# Run CMA-ES optimisation for all UQ methods (5 generations, population of 4).
# Leave running in the background, e.g.:
#   nohup ./run_optim.sh > run_optim.out 2>&1 &

set -euo pipefail

POP_SIZE=4
N_GEN=5
DROPOUT_ITERS=10
BASE_OUT="results/optim"

echo "===== Temperature Scaling ====="
python optimize.py -e ts -p "$POP_SIZE" -n "$N_GEN" -o "$BASE_OUT/ts"

echo "===== Monte Carlo Dropout ====="
python optimize.py -e mcd -p "$POP_SIZE" -n "$N_GEN" -i "$DROPOUT_ITERS" -o "$BASE_OUT/mcd"

echo "===== Levenshtein Monte Carlo Dropout ====="
python optimize.py -e lmcd -p "$POP_SIZE" -n "$N_GEN" -i "$DROPOUT_ITERS" -o "$BASE_OUT/lmcd"

echo "===== All optimisations complete ====="
