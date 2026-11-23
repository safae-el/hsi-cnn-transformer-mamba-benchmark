#!/usr/bin/env bash
# Survey revision queue. Run from the repository root.
#
# Session by session (recommended):
#   bash revision/queue.sh status           what is done, what is left, estimated hours
#   bash revision/queue.sh next 60          run missing runs for about 60 minutes, then stop
#   bash revision/queue.sh one              run only the next missing run
#   touch revision/STOP                     (from another terminal) stop after the current run
# Ctrl+C also works: the run in progress is lost and simply redone next session.
#
# Whole stages (when the laptop can stay on for hours):
#   bash revision/queue.sh smoke            split check + 5-epoch runs, separate CSV
#   nohup bash revision/queue.sh night1 > logs/rev_night1.log 2>&1 &    (random, scarcity, disjoint)
#   nohup bash revision/queue.sh night2 > logs/rev_night2.log 2>&1 &    (all bands)
#   bash revision/queue.sh e5               efficiency; nothing else on the GPU
#   nohup bash revision/queue.sh e6 > logs/rev_e6.log 2>&1 &            (MambaHSI, optional)
# Both ways write to the same results/revision/runs.csv and skip finished runs.
set -euo pipefail
cd "$(dirname "$0")/.."
# activate your Python environment before running
mkdir -p logs
SEEDS="42 123 456 1 2 3 4 5 6 7"
R="python revision/run_revision.py"
P="python revision/plan.py"

case "${1:-}" in
  status)
    $P status ;;
  one)
    ARGS=$($P next)
    if [ -z "$ARGS" ]; then echo "everything is done"; exit 0; fi
    $R $ARGS 2>&1 | tee -a logs/rev_sessions.log
    $P status ;;
  next)
    BUDGET=$(( ${2:-60} * 60 ))
    START=$(date +%s)
    rm -f revision/STOP
    echo "session started $(date '+%F %T'), budget ${2:-60} min" | tee -a logs/rev_sessions.log
    while true; do
      if [ -f revision/STOP ]; then echo "STOP file found, stopping"; rm -f revision/STOP; break; fi
      LEFT=$(( (BUDGET - ($(date +%s) - START)) / 60 ))
      ARGS=$($P next --fit "$LEFT")
      if [ -z "$ARGS" ]; then echo "no missing run fits in the remaining ${LEFT} min"; break; fi
      $R $ARGS 2>&1 | tee -a logs/rev_sessions.log
    done
    echo "session ended $(date '+%F %T')" | tee -a logs/rev_sessions.log
    $P status ;;
  smoke)
    rm -f results/revision/runs_smoke.csv
    $R --check_splits --seeds 42 123 456
    $R --split random   --seeds 42 --epochs 5 --out results/revision/runs_smoke.csv
    $R --split disjoint --seeds 42 --epochs 5 --out results/revision/runs_smoke.csv
    $R --split random   --seeds 42 --epochs 5 --n_pca 0 --micro_batch 16 --models HybridSN SpectralFormer S2Mamba \
       --datasets indian_pines --out results/revision/runs_smoke.csv
    echo "SMOKE DONE"
    python -c "import pandas as p; d=p.read_csv('results/revision/runs_smoke.csv'); print(d[['dataset','model','split','n_pca','n_train','n_test','n_test_classes','OA','mIoU','train_s_per_epoch']].to_string(index=False))" ;;
  night1)
    $R --split random   --seeds $SEEDS
    $R --split random   --seeds $SEEDS --n_train 5 --datasets indian_pines
    $R --split disjoint --seeds $SEEDS
    echo "NIGHT1 DONE" ;;
  night2)
    $R --split random --n_pca 0 --micro_batch 16 --seeds 42 123 456 1 2
    echo "NIGHT2 DONE" ;;
  e5)
    python revision/efficiency.py ;;
  e6)
    $R --models MambaHSI --split random   --seeds $SEEDS
    $R --models MambaHSI --split disjoint --seeds $SEEDS
    echo "E6 DONE" ;;
  *)
    echo "usage: bash revision/queue.sh status | next [minutes] | one | smoke | night1 | night2 | e5 | e6"; exit 1 ;;
esac
