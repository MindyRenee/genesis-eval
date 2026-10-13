# ARC world-play telemetry — 2026-09-24

Recovered from Genesis's live state directory
(`~/.local/share/genesis-public/`) on 2026-10-04.

## What these are

Per-step telemetry and progress from an **external ARC
(`arc_adapter.py`) training/eval run**, not from Genesis herself. The
writer is `world_play.py` in this repository, which persists progress to
`<data_dir>/world_play.json` and per-step records under
`<data_dir>/world_telemetry/`. Genesis's own code never references
either path, and neither name appears in any commit on any branch — so
the files were landing in her state directory because that is where the
harness was pointed.

## Why they were moved, not deleted

Nothing in Genesis reads them, but they are the only record of these
three episodes. Moving them here keeps her state directory to files her
own code creates, without destroying someone else's results.

## Provenance

| file | created | steps |
|---|---|---|
| `telemetry/bt11.jsonl` | 09-24 17:52 | 115 |
| `telemetry/ar25.jsonl` | 09-24 18:28 | 7,533 |
| `telemetry/bp35.jsonl` | 09-24 19:13 | 1,805 |
| `world_play.json` | 09-24 19:10 | progress: 25 sessions, 9,218 steps, 107 episodes, 106 deaths, 1 win |

Schema per step: `action`, `avatar_color`, `data`, `elapsed_ms`,
`levels_completed`, `score`, `state`, `step`.

Note these records have **no `frame_change` field**, unlike
`eval_heldout/telemetry/{cd82,sc25,tu93}.jsonl` from the same day — an
earlier revision of `world_play.py`. The world IDs are also disjoint
(`ar25`/`bp35`/`bt11` vs `cd82`/`sc25`/`tu93`), so this is a **separate
run** from the held-out sweep and is deliberately not filed under
`eval_heldout/`, whose `results.tsv` does not account for these.

## Result summary

From `world_play.json`: 1 win across 106 deaths — `bt11` reached
`best_progress` 1.0, while `ar25` (7,533 actions) and `bp35` (1,805
actions) never cleared a level. The two long runs are the interesting
data here: they are what a long-horizon credit-assignment failure looks
like from the inside.
