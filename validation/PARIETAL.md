# Parietal Cortex — Computational Specification

**Status:** design specification, pre-implementation. Every computation
below is the mathematically documented form used in the neuroscience
literature, not an invented abstraction. Each section ends with what
the mechanism must do *for Genesis*, measured — not what it should be
named.

## What the parietal lobe computes (established models)

The parietal cortex is not one computation. Six interacting
computations are consistently identified across the experimental and
modeling literature. A faithful implementation needs all six; the
lesion literature specifies what each contributes by what its absence
destroys.

### P1. Basis-function representation — reference-frame transforms

**Model:** Pouget & Sejnowski (1994); Zipser & Andersen (1988).

Parietal neurons do not encode position in any single reference frame.
Each unit's response is a *product* of a stimulus term and a
posture/context term — a "gain field":

```
b_i(x, c) = exp(-||x - μ_i||² / 2σ_i²) · σ(w_i·c + θ_i)
```

where `x` is the sensory variable (retinal position), `c` is the
context variable (eye position), `exp(·)` is Gaussian tuning, and
`σ(·)` is a sigmoid gain.

**Why it matters:** a population of such products forms a *basis set*
from which any smooth transform `T_k` is recoverable by linear
readout:

```
ŷ_k = Σ_j α_jk · b_j(x, c)
```

The parietal never commits to a frame — it holds the basis from which
*all* frames are computable simultaneously. (This is why parietal
lesion hemineglect erases multiple reference frames at once rather
than one.)

**For Genesis:** the transform substrate between her declared frames —
self-model / world-model / user-model. Currently these cannot be
converted into each other; they are separate stores. The basis-function
layer is how "what she knows about the user" and "what she knows about
the world" become commensurable.

### P2. Priority map — competitive attention allocation

**Model:** Bisley & Goldberg (2010); Itti & Koch (2001).

A spatial/feature map where each candidate's activity is a weighted
sum of bottom-up salience and top-down relevance:

```
a_i(t+1) = a_i(t) + η·( w_bu·S_i + w_td·R_i - a_i(t) )
a_i ← E · a_i² / (σ² + Σ_j a_j²)          # divisive normalization
winner = argmax a_i                        # winner-take-all
a_winner ← suppressed for refractory τ     # inhibition of return
```

**For Genesis:** attention is currently spreading activation — a walk,
not a competition. Nothing decides "this candidate, not that one" with
mutual suppression; the priority map is the stage where the workspace
gets *one* occupant.

**Chemistry coupling (documented):** ACh sharpens the normalization
denominator (top-down precision); NE raises `w_bu` (stimulus-driven
capture). The hooks exist in `NeurochemicalVector`.

### P3. Bayesian causal inference — the binding computation

**Model:** Körding et al. (2007); Rohe & Noppeney (2015).

Given two concurrent signals `x_1, x_2` with sensory reliabilities
`σ_1², σ_2²`, infer whether they share a cause. With common-cause prior
`p_C`:

```
p(C=1 | x_1, x_2) = sigmoid( log[p_C/(1-p_C)] + LLR )
Ŝ_fused = (x_1·σ_2² + x_2·σ_1²) / (σ_1² + σ_2²)      # min-variance fusion
Ŝ = p(C=1|x)·Ŝ_fused + (1-p(C=1|x))·Ŝ_reliable       # model averaging
```

`Ŝ_reliable` = the higher-reliability single-cue estimate (for C=2,
do not fuse).

**For Genesis:** the edge-formation gate. Currently co-activation →
edge, unconditionally. The parietal computation is "should these be
bound?" — `p(C=1)` computed from co-occurrence reliability and
congruence gates whether an edge is written, and at what weight. This
turns associative glue into causal inference.

### P4. ATOM — the shared magnitude manifold

**Model:** Walsh (2003); Bueti & Walsh (2009); Fechner/Weber scaling.

Time, space, and quantity are represented on **one log-compressed
metric** in the intraparietal sulcus:

```
m(q) = k · ln(q)
compare(q1, q2): |m(q1) - m(q2)| < Δ  →  "about equal"   (Weber fraction)
order(q1, q2):   sign(m(q1) - m(q2))
ratio(q1, q2):   exp(|m(q1) - m(q2)|)
```

Cross-domain interference (a bigger font making a number "feel
bigger") is not a bug — it is the mechanism: duration, distance, and
count share units on the same line.

**For Genesis:** no magnitude substrate exists. `12` and "a long wait"
have no common currency — she cannot answer "was that a long time?"
the way she answers "is that a big number?" The magnitude line is the
substrate of number sense, duration, and scalar comparison — and it is
what makes a conversational partner able to reason about *scale*.

### P5. Bump attractor — held working-memory state

**Model:** Amari (1977); Compte et al. (2000); Wu, Hamaguchi & Amari
(2008); certainty-encoding extension, Carroll et al.

A recurrent field with local excitation and lateral inhibition
supports a self-sustaining activity bump:

```
τ ∂u_i/∂t = -u_i + Σ_j W_ij·f(u_j) + I_i + noise
W_ij = J_e·exp(-d_ij² / 2σ²) - J_i          # Mexican-hat kernel
f(u) = max(0, u) or sigmoid
```

Bump *position* encodes the held value (location/magnitude); bump
*amplitude* encodes certainty — larger amplitude is more noise-robust.
Diffusion of the bump = working-memory error.

**For Genesis:** `state.thought` is overwritten every turn — there is
no persistent held value between turns, no certainty dimension on held
state. The attractor gives working memory *dynamics* rather than a
slot.

### P6. Evidence accumulation — decision dynamics

**Model:** Shadlen drift-diffusion; **correction:** Nature
Communications (2022) showed spatial *selection* and evidence
*accumulation* are dissociable in LIP — the parietal's primary role is
the priority map; accumulation is partly distributed.

```
dx = A·dt + c·dW          commit when |x| ≥ z
```

**For Genesis:** answers commit in a single pass. An evidence
accumulator across retrieval/verification steps — with bound `z`
modulated by cortisol (stress lowers it: faster, worse) — changes
decision dynamics measurably. Implement as *complementary* to P2, not
merged with it (per the 2022 dissociation).

### P7. Ventral reorienting interrupt

**Model:** Corbetta & Shulman (2002; 2008) — TPJ/ventral frontoparietal
network, LC-NE coupled.

A stimulus-driven circuit breaker: behaviorally-relevant unexpected
input preempts ongoing dorsal processing.

```
interrupt_t = ( novelty_t · priority_t > θ ) → preempt, re-perceive
```

NE burst from locus coeruleus gates the interrupt — she already has
norepinephrine; the hook exists.

**For Genesis:** a mid-pipeline check in `think()` — if a
high-priority incongruent signal arrives while a response is being
composed, processing can reroute. Currently `think()` runs its ~15
stages unconditionally regardless of what arrives mid-computation.

---

## Lesion-profile ablation spec (the test plan is already written)

Each subsystem's absence produces a named signature — these are the
ablation arms:

| Ablate | Signature | Probe measurement |
|---|---|---|
| P1 basis functions | multi-frame loss | transforms between self/world/user frames fail |
| P2 priority map | hemineglect analog | selection systematically misses one region of candidate space |
| P3 BCI binding | binding failure | edges form without causal gating (over-binding) or never (under-binding) |
| P4 magnitude line | acalculia analog | all "more/longer/bigger" comparisons fail, regardless of domain |
| P5 attractor | working-memory loss | held value lost after any distractor turn |
| P6 accumulator | premature commit | answers commit before evidence threshold; no "wait" behavior |
| P7 reorienting | no interrupt | incongruent mid-stream input never reroutes processing |

Each is measurable in the existing dynamics-probe harness: intact arm
vs ablated arm, same generated input, measure the downstream variable.

---

## Implementation seams (against current code)

- **P2 priority map** → the candidate-selection stage in `think()`
  (composer competition). ACh modulates `σ` (normalization), NE
  modulates `w_bu`.
- **P3 BCI binding** → gate on `add_edge` call sites in
  `self_learner.learn_from_input` and `_consolidate_to_network`:
  co-activation currently writes edges unconditionally.
- **P4 magnitude line** → new substrate `magnitude.py`; consumers:
  comparison questions, temporal references, confidence-as-magnitude.
- **P5 attractor** → `CognitiveState.thought` persistence between
  turns; amplitude couples to `thought.confidence`.
- **P6 accumulator** → verification path (relation_verification
  already produces verdicts; make verdict reach the response only at
  bound).
- **P7 reorienting** → `neuro_impulse` on incongruent mid-pipeline
  input; NE is already the mechanism the model uses for orienting.

The chemistry interfaces are already in place — ACh, NE, cortisol,
GABA each have documented effects on these computations. The parietal
doesn't need new chemicals; it needs the computations that consume
them.

## What this is NOT

- Not a `ParietalRegion` enum — a label without the six computations is
  decoration.
- Not a new "module" bolted on — each computation hooks an existing
  seam in `think()`, `add_edge`, `neuro_impulse`, or `CognitiveState`.
- Not simulated neurons — the substrate is the *computation* (priority
  map, basis functions, magnitude line, attractor field, DDM, interrupt),
  abstracted at the level the rest of the architecture operates at.

## Validation criterion

A parietal implementation is validated when the seven ablation arms
produce their lesion signatures and the intact arm does not — measured
through the real `think()` pipeline on generated stimuli, per the
dynamics-probe methodology. That is the level of evidence the rest of
this suite operates at.
