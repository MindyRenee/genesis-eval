#!/usr/bin/env bash
# Transfer-compounding eval: fixed protocol, no per-task tuning.
# Pinned PYTHONHASHSEED because agent seeds derive from hash((name,
# attempts)) — without it, replicate runs are not reproducible.
#
# Lives outside the Genesis checkout — point GENESIS_DIR at the repo
# (default: a sibling checkout named "genesis").
set -u
EVAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GENESIS_DIR="${GENESIS_DIR:-$EVAL_ROOT/../genesis}"
export PYTHONPATH="$GENESIS_DIR/python"
export PYTHONHASHSEED=0
cd "$EVAL_ROOT"
python3 eval_transfer/run_eval.py "$@"
echo "SWEEP DONE"
