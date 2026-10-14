# ARC-AGI-3 practice artifacts

Replayable `arcengine` recordings of Genesis playing public ARC-AGI-3
practice environments, plus run manifests and her cognition log where the
run captured one.

**This is practice evidence, not an official scorecard.** Every run here
used `operation_mode="offline"` — local environment files, scorecard ids
minted locally by the SDK, nothing submitted to the ARC service. The
boundary between local diagnostics and official evidence is defined in
Genesis's `docs/ARC-AGI-3.md`; per that standard none of this is a
claimed result on the hidden set.

## Results at a glance

29 recorded sessions across three periods; `RESULTS.tsv` enumerates all
of them, committed or not.

- **bt11** (world-play eval, 2026-09-24): a 5-level game, **completed
  twice** — 88 and 127 steps — inside the same session set that opened
  with an 802-step `GAME_OVER` reaching only level 1. The mechanics were
  learned during play, not before it.
- **ls20** (eval sweep, 2026-10-13): the opening level
  (`levels_completed = 1`) was completed in 10 of 19 runs with
  manifests, never deeper. Early Oct-11 practice sessions reached no
  levels; the completion rate appeared with play. Two sweep directories
  have no manifest — counted as 0 here rather than guessed.
- **ar25** (world-play eval, 2026-09-24): a 7,620-step session that
  reached level 1. The 96 MB recording is not committed; its telemetry
  is in `eval_arc_worldplay_20260924/telemetry/ar25.jsonl`.

## Layout

- `ls20-sweep-20261013/<run>/` — four complete run directories:
  `meta.json` (outcome, git hash, her composed mechanics summary),
  `thoughts.log` (her cognition stream during the episode), and the
  recording under `recordings/<scorecard_id>/`.
- `ls20-early-20261011/` — the four Oct-11 practice sessions (0 levels).
- `bt11-20260924/` — all five Sep-24 bt11 sessions including both WINs.

## How to verify

Recordings are the standard `arcengine` JSONL stream: each line is
`{"timestamp", "data"}` where `data` carries the full 64×64 frame, the
`action_input` (including a `reasoning` provenance block on runs new
enough to emit it), `state`, `levels_completed`, and `win_levels`. Step
any file line-by-line and diff her stated reasoning against the pixels.

To reproduce a run from the Genesis checkout:

```bash
./run.sh --offline        # mind up, network closed
# then at the prompt: /arc ls20-9607627b --steps 200
# recordings land in <data dir>/arc/recordings/<scorecard_id>/
```

## Decision provenance (`reasoning.source`)

- `field_leader` — the learned policy's deliberate route: a plan through
  her learned physics toward cells with learned value.
- `probe` — she deliberately contacted an entity whose effect was still
  unlearned; mechanic-testing behavior.
- `calibrate` — least-evidence action sampling while the policy calibrates.
- `rng_no_body` — random fallback when no embodied signal was available;
  it appears in the early sessions and is exactly what it looks like.

## Her own account of the mechanics

From `ls20-sweep-20261013/20261013T172117Z-ls20/meta.json`, composed by
her language engine at run end — this is her *stated* model of ls20,
learned from consequences alone:

> "I am the color-12 thing; 21 things around me; color-0 things nothing
> me; color-1 things nothing me; color-8 things can be collected;
> color-9 things block me; color-11 things can be collected — 75 places;
> 75 routes — 6 spatial skills"

The grammar is hers. The categories — self, collectible, lethal,
blocked — are the categories her world model actually stores.

## What this does not show

- No official scorecard exists yet; nothing here is a submission.
- Level completions are shallow. The gap between "completes an opening
  level" and "plans across the level structure" is documented in
  `docs/ARC-AGI-3.md` as open work, not hidden here.
