# Genesis Validation Suite — external experiment

Reproducible evidence about whether Genesis is *actually* intelligent —
what the code computes, what the system does with it, and where the
architecture falls short.

**This is a temporary experiment, not project code.** The suite lives
outside the genesis repo (in `genesis-eval/`); nothing here ships with
her. The loop: probe → get results → refactor/adjust *her* code where
a finding warrants it → re-probe to verify. The probes are disposable
measurement rigs; the fixes are deliberate changes to the production
codebase, made separately in the genesis repo.

## Honest framing (read this first)

- **Nothing here is a proof of AGI.** Passing probes demonstrate
  specific causal dependencies and capabilities, not intelligence in
  the general sense. The suite exists to separate *what works* from
  *what we claim works*.
- **Failures are findings.** A failed probe is not hidden — it is a
  documented gap in the architecture, which is often the more valuable
  result.
- **Stimuli are generated, not chosen.** Breakthrough-tier inputs are
  procedurally generated pseudowords and patterns (per-run seeds,
  recorded in the report). Nothing in the corpus can answer them by
  lookup.
- **Two backends, honestly labeled.** `offline` uses a declared
  in-memory client stand-in; `live` uses a real `GenesisClient` over a
  socket. Reports identify which backend produced each result — the
  offline arm never masquerades as daemon-verified.

## Tiers

```
substrate     mechanism-level exact-math checks  (does the code compute it?)
capability    end-to-end behaviors via think()   (does the system do it?)
dynamics      in-vivo ablation — mechanism on/off, same input, diff state
breakthrough  behaviors a lookup table cannot produce (the intelligence audit)
```

The neurochemical/interoception Rust probes live in `rust-probes/tests/`
(`validation_neurodynamics`, `validation_neuro_map`,
`validation_interoception`, `validation_dose_response`) — a sibling
crate with a path dependency on genesis, run via `cargo test` inside
`rust-probes/`.

## Run

```bash
# Python tiers — PYTHONPATH needs her python/ + this repo root.
# G is the directory holding both checkouts side by side.
G=<checkout parent>   # contains genesis/ and genesis-eval/
PYTHONPATH=$G/genesis/python:$G/genesis-eval \
    python3 -m validation.run_probes
PYTHONPATH=$G/genesis/python:$G/genesis-eval \
    python3 -m validation.run_probes --tier breakthrough

# Live arm — requires a dedicated scratch daemon socket (never the
# user's real Genesis; teaching writes real LTM episodes)
PYTHONPATH=$G/genesis/python:$G/genesis-eval \
    python3 -m validation.run_probes --backend live \
    --socket /path/to/scratch-daemon.sock

# Rust tiers — sibling crate with a path dependency on genesis;
# probes exercise the production crate through its public API only
cd rust-probes && cargo test
```

Reports land in `validation/results/probes-<ts>-<backend>.json` —
probe id, claim, tier, evidence type, backend, measurements (expected /
observed / tolerance / Wilson CI / exact binomial p), verbatim
responses, notes, pass/fail. The dose-response sweep logs additionally
to `validation/results/dose-response-sweep.txt`.

## Methodology

**Substrate** (`probes/substrate.py`) — the mechanisms in isolation,
checked against hand-computed answers: STDP weight curve, TD
fixed-point, beta conjugacy, spreading activation, edge-log fidelity,
language grounding, synapse persistence.

**Capability** (`probes/capability.py`) — the whole pipeline does
intelligent things: learn a taught fact, decline an untaught one,
verify a derived fact, retain across restart, and keep confidence
meaningful. Exact statistics; no observation dropped.

**Dynamics** (`probes/dynamics.py`) — the deeper test: **remove a
mechanism inside the live `think()` pipeline and see what changes.**
Same generated input, mechanism on vs off, diff the measured substrate
(synaptic weights, edges, dopamine signal, verdict metadata, episodes,
responses). What doesn't move when the mechanism is off is decorative;
what moves is load-bearing. Established so far:

- A taught fact is written by **two** independent mechanisms —
  `self_learner` (w≈0.7) + semantic consolidation (→0.958 combined).
  Redundant write paths, decomposed.
- STDP is live in-vivo: real turns write synapses at the exact
  closed-form weight `A₊·e^(−Δt/τ)·lr` — and user→response binding is
  *absent* under the current ≥40 ms gap vs 40 ms window.
- The verification verdict is load-bearing for yes/no answers (ablate
  it → grounded answer collapses to a decline).
- Episodic vs semantic memory are separable: episodes ablate to zero
  while semantic recall keeps working.
- The hedge gate is redundant on unanswerable items (composer already
  declines).

**Chemical map** (`tests/validation_*.rs` + `dose-response`) — all 18
chemicals at depleted / low / baseline / high, 200-tick holds, every
downstream channel recorded (valence, arousal, tone, plasticity gate,
memory enc/cons/retr weights, emergent phase). 144 declared couplings
verified by sign + proportionality. Cortisol closes plasticity
(BDNF·(1−cort·0.5) measured), ACh opens all three memory channels,
adenosine flips to NREM. Findings logged per-chemical.

**Breakthrough** (`probes/breakthrough.py`) — the "can a lookup do
this?" tier. Each probe is constructed so canned responses fail by
design; the report is an audit, not a gate.

*Current state — after the probe-driven refactors documented below.
The first-run column records what the probes measured before any fix;
the current column is what the suite measures now.*

| Probe | First run | Current | Mechanism added |
|---|---|---|---|
| reasoning_trace | verdicts 8/8 · explanations 0/8 | **PASS** 8/8 · 8/8 | `is_a_path` BFS returns the derivation; verification metadata carries `path`/`conflicts`; the composer articulates intermediates |
| contradiction_detection | flagged 0/8 · revised 0/8 | **PASS** 8/8 · 8/8 | taxonomic-disjointness check at the `IS_A` write site, across senses; `weaken_edge` revises through the canonical log; `belief_conflict`/`correction` events surface it |
| gap_interrogation | 16/32 questions | **PASS** 16/32, p=0.002 | unchanged — interrogates unbound referents ~half the time; zero fabricated referents |
| self_calibration | AUC 1.000 (fallback channel) | **PASS** AUC 1.000 on `answer_assessment` | `answer_assessment` field on `CognitiveState`, populated on early routes too; deliberative verdicts fuse into the assessment |
| sequence_induction | ~chance (3/8) | **PASS** 8/8, p=0.0001 | `predict_continuation`/`continuation_distribution` hooks on `LanguageEngine`; `question_sequence_continuation` restricts the learned transition distribution to the observed alphabet; the induction is emitted as a real `ReasoningType.INDUCTIVE` result so the decision engine scores its evidence |
| counterfactual | literals 3/8 | **PASS** 8/8 engaged · 0 literals | WHAT_IF for embedded "if … were/would" premises; `counterfactual_premise` parser shared by both routes; hypothetical clauses excluded from fact extraction; graph-walk suppressed so the literal relation can't outscore the honest answer |
| autonomous_initiation | 3/12 echoes | **PASS** 12/12, p≈6e-8 | contentless turns (no question, no topics) fall back to a `social_presence` question — the interlocutor is the one open unknown; backchannel interjections removed from topic extraction |
| self_monitor repair | — | fixed | a <2-word response triggered a repair that rendered the *diagnostic string* ("unable to respond meaningfully") as speech — now keeps the original when the recompose just echoes the diagnostic |

Earlier first-run measurements (pre-refactor) are preserved in
`results/probes-20260926-185627-offline.json` for comparison — the
diff between that file and the current run is the measured effect of
the refactors, not a claim about a fresh mind.

## Known limitations & honest disclosures

- Passing a breakthrough probe is *behavioral* evidence only. The
  dynamics tier (mechanism on/off, same input, different state) is
  where causal claims live — e.g., sequence induction passes because
  the statistical learner's transitions drive the prediction; ablating
  the learner should collapse it (not yet tested as an ablation).
- `answer_assessment` is now populated on all response routes
  including early/meta-cognitive ones (added during this audit).
- An earlier breakthrough run caught **nondeterministic web-lookup
  injection** — raw search-result spam (including explicit terms)
  spoken verbatim on novel-word input. Fixed: `question_handler`
  `_extract_relevant_summary` now filters fetched text by topicality
  and rejects boilerplate/nav-chrome/digit-heavy spam before any of it
  reaches speech; unfiltered evidence preserved in
  `results/probes-20260926-185627-offline.json`.
- `birds` vs `bird` are distinct concepts — no plural normalization.
- STDP user→response binding is absent under the current 40 ms window
  / ~50 ms spacing — recorded, not fixed.
- Offline backend uses the declared client stand-in; daemon-verified
  claims require `--backend live` against a scratch daemon.
- The parietal lobe does not exist — see `validation/PARIETAL.md`
  (research spec: basis functions, priority map, Bayesian causal
  inference, ATOM magnitude, bump attractor, DDM, reorienting; plus
  the lesion-profile ablation plan).

## Related docs

- `validation/PARIETAL.md` — parietal computational specification.
- `AGENTS.md` — repository conventions (no hardcoded responses;
  edge log is canonical; state-integrity rules).
