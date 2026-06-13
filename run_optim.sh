#!/usr/bin/env bash
# Run CMA-ES optimisation for all UQ methods (5 generations, population of 4).
# Sleep/suspend is disabled, so just run it directly and let it power off when done:
#   ./run_optim.sh
#
# Set SHUTDOWN=0 below to keep the machine on after the run.
# On Ubuntu (systemd), `systemctl poweroff` is normally allowed for a local
# user via polkit. If it isn't, allow passwordless `sudo systemctl poweroff`.

set -uo pipefail

POP_SIZE=4
N_GEN=5
DROPOUT_ITERS=10
BASE_OUT="results/optim"
SHUTDOWN=1

shutdown_on_exit() {
    local exit_code=$?
    if [[ "$exit_code" -ne 0 ]]; then
        echo "===== Script exited with error (code $exit_code) ====="
    fi
    if [[ "$SHUTDOWN" == "1" ]]; then
        echo "===== Powering off in 60s (Ctrl-C to cancel) ====="
        sleep 60
        systemctl poweroff || sudo systemctl poweroff
    fi
}
trap shutdown_on_exit EXIT

echo "===== Temperature Scaling ====="
python optimize.py -e ts -p "$POP_SIZE" -n "$N_GEN" -o "$BASE_OUT/ts"

echo "===== Monte Carlo Dropout ====="
python optimize.py -e mcd -p "$POP_SIZE" -n "$N_GEN" -i "$DROPOUT_ITERS" -o "$BASE_OUT/mcd"

echo "===== Levenshtein Monte Carlo Dropout ====="
python optimize.py -e lmcd -p "$POP_SIZE" -n "$N_GEN" -i "$DROPOUT_ITERS" -o "$BASE_OUT/lmcd"

echo "===== All optimisations complete ====="
