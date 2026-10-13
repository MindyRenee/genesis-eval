#!/usr/bin/env bash
# Held-out ARC-3 sweep: fixed protocol, no per-game tuning.
# seed 0, 1000-action budget, official recording + JSONL telemetry.
#
# Lives outside the Genesis checkout — point GENESIS_DIR at the repo
# (default: a sibling checkout named "genesis").
set -u
EVAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GENESIS_DIR="${GENESIS_DIR:-$EVAL_ROOT/../genesis}"
export PYTHONPATH="$GENESIS_DIR/python"
GAMES="cd82 sc25 tu93 tn36 wa30 dc22 ka59 r11l s5i5 ar25 vc33 g50t lp85 ft09 su15 bp35 cn04 sk48 sb26 re86 sp80 tr87 lf52 m0r0 ls20"
OUT="$EVAL_ROOT/eval_heldout/results.tsv"
printf "game\tresult\tsteps\tlevels\tresets\tstate\n" > "$OUT"
for g in $GAMES; do
    line=$(timeout 900 python3 "$EVAL_ROOT/arc_adapter.py" \
        --game "$g" --seed 0 --actions 1000 \
        --telemetry "$EVAL_ROOT/eval_heldout/telemetry/${g}.jsonl" \
        --env-dir "$EVAL_ROOT/environment_files_heldout" 2>/dev/null | tail -1)
    echo "== $g: $line"
    echo -e "$g\t$line" | sed 's/:/\t/' >> "$OUT"
done
echo "SWEEP DONE"
