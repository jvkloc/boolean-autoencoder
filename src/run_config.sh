#!/bin/bash
# Runs one B⊕LD configuration completely: η search (ids 1-5, in parallel, each with
# 24 workers), η choice, final runs (ids 1-100, MAX_JOBS at a time), summary.
# From bae/src with the bae environment active:
#   nohup bash run_config.sh bold_tanh_noscale_layer.py > ../logs/bold_tanh_noscale_layer.log 2>&1 &
# Finished runs are skipped, so the same command resumes an interrupted run.

set -e
CONFIG=$1
if [ -z "$CONFIG" ]; then echo "usage: bash run_config.sh <config>.py"; exit 1; fi
NAME=$(basename "$CONFIG" .py)
LOG_DIR=../logs/$NAME
MAX_JOBS=100
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one core per process
mkdir -p "$LOG_DIR"

echo "$(date) $NAME: eta search"
for i in 1 2 3 4 5; do
    python -u "$CONFIG" search "$i" > "$LOG_DIR/search_$i.log" 2>&1 &
done
wait
python -u "$CONFIG" choose

echo "$(date) $NAME: final runs"
for i in {1..100}; do
    python -u "$CONFIG" final "$i" > "$LOG_DIR/final_$i.log" 2>&1 &
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
        wait -n
    done
done
wait
python -u "$CONFIG" summary
echo "$(date) $NAME: done, see ../results/$NAME/summary.md"
