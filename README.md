# genesis-eval

The external measurement rig for
[Genesis](https://github.com/MindyRenee/Genesis) — nothing here ships
with her. The suite exists to separate *what works* from *what we claim
works*; a failed probe is a documented gap, not a hidden one.

## Layout

- `validation/` — the probe suite. Substrate (exact-math mechanism
  checks), capability (end-to-end behavior), dynamics (in-vivo
  ablation), breakthrough (stimuli a lookup table cannot answer).
  `rust-probes/` is a sibling crate that exercises the production
  `genesis` crate through its public API only. See
  `validation/README.md` for tiers, methodology, and the run commands.
- `eval_heldout/`, `eval_transfer/` — ARC held-out and transfer sweeps:
  `results.tsv`, per-step `telemetry/`, and `summary.txt`. The negative
  results are in these files on purpose.
- `eval_arc_worldplay_20260924/` — per-step telemetry from an external
  ARC training/eval harness, preserved because it is the only record of
  those episodes.
- `arc_adapter.py`, `world_play.py`, `tests/` — the external harness
  and its tests.

## What is not here

`environment_files/` and `environment_files_heldout/` are excluded on
purpose: ARC environments are distributed by the `arcengine` SDK, and
republishing the held-out corpus would contaminate the very evaluation
it exists to keep honest. Run the SDK to fetch your own copies.

## Run

Requires a side-by-side checkout of `genesis` — see
`validation/README.md`.
