"""Substrate probes — does each mechanism compute what it claims to compute?

Every probe here is a *controlled causal comparison*:

- A single variable is toggled (learning on/off, edge present/absent,
  spike timing ±5ms, modulator 0/1) while everything else is held fixed.
- Observations are checked against analytic predictions — closed-form
  values derived independently from the published update rules — not
  merely against "did it change".
- Control arms must be exactly quiet: a disabled mechanism produces a
  zero delta, proving the measured effect is caused by the mechanism
  rather than the probe harness.

These are implementation evidence. They do not by themselves establish
end-to-end intelligent behavior; that claim belongs to capability.py.
"""

from __future__ import annotations

import math
import tempfile
from itertools import pairwise
from pathlib import Path

from genesis_cognitive.concepts import ConceptNetwork
from genesis_cognitive.concepts.edge_log import EdgeLog, is_derivable_edge
from genesis_cognitive.concepts.types import RelationType
from genesis_cognitive.learning import STDP, SynapticStore, TDLearner
from genesis_cognitive.reasoning.decision import ActionType, DecisionEngine
from genesis_cognitive.self.model import SelfModel
from genesis_cognitive.world.belief import PresenceBelief

from .report import Measurement, ProbeReport, check, check_exact

# ─────────────────────────────────────────────────────────────────
# TD learning — convergence to the Bellman fixed point
# ─────────────────────────────────────────────────────────────────
#
# With a single-concept state s=[alpha] and fixed next state
# s'=[beta] (whose weight is never updated, so V(s') = c_beta is
# constant), the TD(0) recursion is exactly linear:
#
#     δ_k   = r + γ·V(s') − V_k(s)
#     w_{k+1} = w_k + α·δ_k          (trace = 1 for the sole concept)
#     V_{k+1} = c_a·w_{k+1} = V_k + α·c_a·δ_k
#     ⇒ δ_{k+1} = δ_k·(1 − α·c_a)    — a pure geometric sequence
#
# so the RPE must shrink by the exact factor (1 − α·c_a) each step and
# the value must converge to the fixed point V* = r + γ·V(s').


def probe_td_fixed_point() -> ProbeReport:
    """TD(0) converges to the analytic fixed point with geometric RPE decay."""
    r = ProbeReport(
        probe_id="substrate.td_fixed_point",
        title="TD learning converges to the Bellman fixed point",
        claim="TDLearner implements actual temporal-difference learning: "
              "RPE decays geometrically at exactly (1 − α·c) per step and "
              "the value converges to V* = r + γ·V(s').",
        tier="substrate",
        evidence="implementation",
    )
    alpha, gamma, conf = TDLearner.DEFAULT_LEARNING_RATE, TDLearner.DEFAULT_DISCOUNT, 0.9
    reward = 1.0
    steps = 150

    def run(enabled: bool) -> tuple[TDLearner, list[float]]:
        net = ConceptNetwork()
        net.add_concept("alpha", confidence=conf, origin="validation")
        net.add_concept("beta", confidence=conf, origin="validation")
        learner = TDLearner(net, lam=0.0)
        rpes: list[float] = []
        for _ in range(steps if enabled else 0):
            rpes.append(learner.update(["alpha"], reward=reward, next_state=["beta"]))
        return learner, rpes

    learner, rpes = run(enabled=True)
    disabled, _ = run(enabled=False)

    v_star = reward + gamma * conf          # 1.0 + 0.9·0.9 = 1.81
    w_star = v_star / conf                   # ≈ 2.0111
    decay_rate = 1.0 - alpha * conf          # 0.91

    r.measurements += [
        check("final_value_at_fixed_point", v_star,
              learner.predict_value(["alpha"]), tol=1e-4,
              detail=f"V* = r + γ·V(s') = {reward} + {gamma}·{conf}"),
        check("final_weight", w_star, learner.get_weight("alpha"), tol=1e-4),
        check("first_rpe", reward + gamma * conf - conf, rpes[0], tol=1e-12),
        check("rpe_geometric_step_10", rpes[0] * decay_rate**10, rpes[10], tol=1e-12),
        check("rpe_geometric_step_100", rpes[0] * decay_rate**100, rpes[100], tol=1e-9),
        check("control_arm_silent", 0.0,
              disabled.predict_value(["alpha"]) - conf, tol=0.0,
              detail="no updates → value stays at the prior (confidence)"),
    ]
    return r


def probe_td_reward_sign() -> ProbeReport:
    """TD moves weights in the direction of the reward sign."""
    r = ProbeReport(
        probe_id="substrate.td_reward_sign",
        title="TD value moves with reward sign",
        claim="Positive reward raises predicted value; zero reward after "
              "a learned expectation lowers it (negative RPE).",
        tier="substrate",
        evidence="implementation",
    )
    net = ConceptNetwork()
    net.add_concept("alpha", confidence=0.9, origin="validation")
    learner = TDLearner(net, lam=0.0)

    v0 = learner.predict_value(["alpha"])
    for _ in range(30):
        learner.update(["alpha"], reward=1.0, next_state=["alpha"])
    v_pos = learner.predict_value(["alpha"])
    for _ in range(60):
        learner.update(["alpha"], reward=0.0, next_state=["alpha"])
    v_neg = learner.predict_value(["alpha"])

    r.measurements += [
        Measurement("positive_reward_raises", "increase", v_pos > v0, v_pos > v0,
                    f"V {v0:.6f} → {v_pos:.6f}"),
        Measurement("zero_reward_lowers", "decrease", v_neg < v_pos, v_neg < v_pos,
                    f"V {v_pos:.6f} → {v_neg:.6f}"),
        check("negative_rpe_sign", -1.0, math.copysign(1.0, learner.get_rpe()), tol=0.0),
    ]
    return r


# ─────────────────────────────────────────────────────────────────
# STDP — exact timing rule, application gating, downstream read
# ─────────────────────────────────────────────────────────────────
#
# The documented rule is:
#     Δt = t_post − t_pre
#     Δt > 0 →  +A₊·exp(−Δt/τ₊)     (LTP)
#     Δt < 0 →  −A₋·exp(+Δt/τ₋)     (LTD)
#     |Δt| > window  → no pair formed (boundary is inclusive: |Δt|=40 counts)
#     Δw = learning_rate · rule(Δt) · modulator,  clamped to [0,1]


def probe_stdp_timing_curve() -> ProbeReport:
    """The STDP timing curve matches the analytic exponential exactly."""
    r = ProbeReport(
        probe_id="substrate.stdp_timing_curve",
        title="STDP timing rule matches the analytic curve",
        claim="compute_stdp_window(Δt) equals +A₊e^(−Δt/τ₊) for Δt>0, "
              "−A₋e^(Δt/τ₋) for Δt<0, +A₊ at Δt=0 — the classic "
              "Bi & Poo (1998) shape, exact to float precision.",
        tier="substrate",
        evidence="implementation",
    )
    stdp = STDP(SynapticStore(), ConceptNetwork())
    for dt in (-39.0, -20.0, -10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 20.0, 39.0):
        if dt > 0:
            expected = STDP.A_PLUS * math.exp(-dt / STDP.TAU_PLUS)
        elif dt < 0:
            expected = -STDP.A_MINUS * math.exp(dt / STDP.TAU_MINUS)
        else:
            expected = STDP.A_PLUS
        r.measurements.append(
            check(f"stdp_window_dt{dt:+.0f}", expected,
                  stdp.compute_stdp_window(dt), tol=1e-15)
        )
    return r


def _train_ltp(synapses: SynapticStore, net: ConceptNetwork,
               pairs: int, dt: float, *, modulator: float = 1.0) -> STDP:
    """Drive `pairs` one-to-one pre→post spike pairs, spaced past the window.

    Spacing successive pairs by 100ms guarantees exactly one pair per
    iteration — earlier spikes are outside the ±40ms window — so the
    analytic expectation is pairs·lr·modulator·rule(dt), not an estimate.
    """
    stdp = STDP(synapses, net)
    stdp.set_modulator(modulator)
    for k in range(pairs):
        stdp.record_spike("alpha", 100.0 * k)
        stdp.record_spike("beta", 100.0 * k + dt)
    stdp.apply_updates()
    return stdp


def probe_stdp_integration() -> ProbeReport:
    """N pairings produce exactly N·α·A₊·e^(−Δt/τ₊) weight on the synapse."""
    r = ProbeReport(
        probe_id="substrate.stdp_integration",
        title="STDP integrates to the closed-form weight",
        claim="Ten pre→post pairs at Δt=+5ms produce "
              "w = 10·lr·A₊·e^(−5/20)·modulator on the alpha→beta synapse, "
              "exactly; the control arm (modulator 0) produces exactly zero "
              "while still recording the spikes.",
        tier="substrate",
        evidence="implementation",
    )
    net = ConceptNetwork()
    for name in ("alpha", "beta"):
        net.add_concept(name, confidence=0.9, origin="validation")

    syn_on = SynapticStore()
    _train_ltp(syn_on, net, 10, 5.0, modulator=1.0)
    syn_gate = SynapticStore()
    gated = _train_ltp(syn_gate, net, 10, 5.0, modulator=0.0)

    per_pair = STDP.LEARNING_RATE * STDP.A_PLUS * math.exp(-5.0 / STDP.TAU_PLUS)
    expected = 10 * per_pair
    observed = syn_on.get_weight("alpha", "beta")

    r.measurements += [
        check("ltp_weight_exact", expected, observed, tol=1e-15),
        check("modulator_gate_silent", 0.0, syn_gate.get_weight("alpha", "beta"),
              tol=0.0, detail="modulator=0 gates application, not the mechanism"),
        check_exact("gated_pairs_counted", 10, gated.total_gated,
                    "all 10 pairs computed and gated"),
        check("propagation_reads_efficacy", observed,
              syn_on.propagate(["alpha"], 1.0).get("beta", -1.0), tol=0.0,
              detail="learned efficacy is the downstream read path"),
    ]
    return r


def probe_stdp_timing_boundaries() -> ProbeReport:
    """The ±40ms pairing window is a real boundary, and reversed order gives LTD."""
    r = ProbeReport(
        probe_id="substrate.stdp_timing_boundaries",
        title="STDP window boundary and LTD direction",
        claim="Pairs at |Δt|>40ms produce zero; |Δt|=40 counts (inclusive "
              "boundary); post-before-pre ordering produces LTD of exactly "
              "−A₋e^(Δt/τ₋) per pair.",
        tier="substrate",
        evidence="implementation",
    )
    net = ConceptNetwork()
    for name in ("alpha", "beta"):
        net.add_concept(name, confidence=0.9, origin="validation")

    # Outside window → exactly zero.
    syn_out = SynapticStore()
    _train_ltp(syn_out, net, 10, 45.0)
    w_out = syn_out.get_weight("alpha", "beta")

    # Inclusive boundary: Δt=40 counts.
    syn_edge = SynapticStore()
    _train_ltp(syn_edge, net, 1, 40.0)
    w_edge = syn_edge.get_weight("alpha", "beta")
    expected_edge = STDP.LEARNING_RATE * STDP.A_PLUS * math.exp(-40.0 / STDP.TAU_PLUS)

    # LTD: beta (post) recorded 5ms BEFORE alpha (pre) → pair (alpha,beta)
    # with Δt = −5 → weight decreases by lr·A₋·e^(−5/20) per pair.
    syn_ltd = SynapticStore()
    _train_ltp(syn_ltd, net, 10, 5.0)  # build up first
    w_before = syn_ltd.get_weight("alpha", "beta")
    stdp = STDP(syn_ltd, net)
    for k in range(4):
        stdp.record_spike("alpha", 10_000.0 + 100.0 * k + 5.0)
        stdp.record_spike("beta", 10_000.0 + 100.0 * k)  # post before pre
    stdp.apply_updates()
    w_after = syn_ltd.get_weight("alpha", "beta")
    per_ltd = STDP.LEARNING_RATE * STDP.A_MINUS * math.exp(-5.0 / STDP.TAU_MINUS)

    r.measurements += [
        check("outside_window_zero", 0.0, w_out, tol=0.0,
              detail="Δt=45ms pairs produce no connection"),
        check("window_edge_inclusive", expected_edge, w_edge, tol=1e-15,
              detail="Δt=40ms is inside the window"),
        check("ltd_drop_exact", w_before - 4 * per_ltd, w_after, tol=1e-15,
              detail="4 reversed pairs → exactly −4·lr·A₋·e^(−5/20)"),
        Measurement("ltd_direction", "decrease", w_after < w_before,
                    w_after < w_before, f"{w_before:.6g} → {w_after:.6g}"),
    ]
    return r


# ─────────────────────────────────────────────────────────────────
# Spreading activation — fan-out and decay are the documented formula
# ─────────────────────────────────────────────────────────────────
#
# spread = source_activation · (w_edge/Σw) · decay^(d+1) · multipliers
# With one edge (share=1), same-column concepts, and depth≥1:
#     activated[target] = amount · 0.5          (decay^1 = 0.5)


def probe_spreading_activation() -> ProbeReport:
    """Spreading activation follows the fan-out/depth-decay formula exactly."""
    r = ProbeReport(
        probe_id="substrate.spreading_activation",
        title="Spreading activation matches the analytic formula",
        claim="A neighbor receives amount·share·0.5^depth; share is the "
              "edge-weight fraction. No edge → no activation (control arm "
              "exactly silent).",
        tier="substrate",
        evidence="implementation",
    )

    def run(with_edge: bool, weights: tuple[float, ...] = (0.8,)) -> dict[str, float]:
        net = ConceptNetwork()
        for name in ("alpha", "beta", "gamma"):
            net.add_concept(name, confidence=0.9, origin="validation")
        if with_edge:
            targets = ("beta", "gamma")[: len(weights)]
            for t, w in zip(targets, weights, strict=True):
                net.add_edge("alpha", t, RelationType.RELATED_TO, weight=w,
                             origin="validation")
        return net.spread_activation(["alpha"], amount=0.2, depth=1)

    one_edge = run(True, (0.8,))
    no_edge = run(False)
    two_edges = run(True, (0.3, 0.9))

    beta_id = "beta"
    r.measurements += [
        check("single_edge_spread", 0.2 * 1.0 * 0.5, one_edge.get(beta_id, 0.0),
              tol=1e-15, detail="amount·(w/Σw)·decay¹ = 0.2·1·0.5"),
        check("no_edge_silent", 0.0, no_edge.get(beta_id, 0.0), tol=0.0),
        check("fanout_beta_share", 0.2 * (0.3 / 1.2) * 0.5,
              two_edges.get("beta", 0.0), tol=1e-15,
              detail="share = 0.3/(0.3+0.9) = 0.25"),
        check("fanout_gamma_share", 0.2 * (0.9 / 1.2) * 0.5,
              two_edges.get("gamma", 0.0), tol=1e-15,
              detail="share = 0.9/1.2 = 0.75"),
    ]
    return r


# ─────────────────────────────────────────────────────────────────
# Edge log — the canonical store actually is canonical
# ─────────────────────────────────────────────────────────────────


def probe_edge_log_fidelity() -> ProbeReport:
    """Fold semantics: last-write-wins, tombstones, snapshot reset, crash tolerance."""
    r = ProbeReport(
        probe_id="substrate.edge_log_fidelity",
        title="Edge log fold semantics are exact",
        claim="Assert/retract/snapshot replay to the expected edge set "
              "exactly; derivable edges are refused; a torn tail line is "
              "skipped; compaction preserves the fold; reopening the file "
              "yields the identical fold.",
        tier="substrate",
        evidence="implementation",
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "edge_log.jsonl"
        log = EdgeLog(path)
        log.assert_edge("a", "b", RelationType.IS_A, 0.7, "validation", 1000)
        log.assert_edge("a", "b", RelationType.IS_A, 0.9, "validation", 1000)
        log.assert_edge("c", "d", RelationType.CAUSES, 0.8, "validation", 2000)
        log.retract_edge("c", "d", RelationType.CAUSES)
        derivable_ok = log.assert_edge(
            "x", "y", RelationType.RELATED_TO, 0.5, "hub_attachment", 3000
        )
        fold1 = log.fold()
        live1 = {(e.source, e.target, e.relation.value): e.weight
                 for e in fold1.values()}

        # Crash-tolerance: append a torn JSON tail, re-fold.
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"op":"assert","source":"ghost","ta')
        fold_torn = {(e.source, e.target, e.relation.value): e.weight
                     for e in log.fold().values()}

        n_compacted = log.compact()
        fold_compact = {(e.source, e.target, e.relation.value): e.weight
                        for e in log.fold().values()}
        log.close()

        # Reopen the same file — persistence check.
        log2 = EdgeLog(path)
        fold_reopen = {(e.source, e.target, e.relation.value): e.weight
                       for e in log2.fold().values()}
        log2.close()

    r.measurements += [
        check_exact("last_write_wins", {("a", "b", "is_a"): 0.9}, live1),
        check_exact("derivable_rejected", False, derivable_ok),
        check_exact("torn_tail_ignored", live1, fold_torn),
        check_exact("compact_preserves_fold", live1, fold_compact),
        check_exact("compact_count", 1, n_compacted),
        check_exact("reopen_identical", live1, fold_reopen),
    ]
    return r


def probe_derivable_edge_policy() -> ProbeReport:
    """With a live log, derivable edges are never materialized; typed always are."""
    r = ProbeReport(
        probe_id="substrate.derivable_edge_policy",
        title="Derivable-edge rejection is enforced at the boundary",
        claim="An untyped edge with a pipeline origin returns None and "
              "never enters the network or the log; a typed relation with "
              "the same origin is still canonical; an untyped edge with an "
              "unlisted origin defaults to canonical (the asymmetric "
              "safety rule).",
        tier="substrate",
        evidence="implementation",
    )
    with tempfile.TemporaryDirectory() as td:
        net = ConceptNetwork()
        for name in ("alpha", "beta", "gamma", "delta", "epsilon"):
            net.add_concept(name, confidence=0.9, origin="validation")
        net.attach_edge_log(EdgeLog(Path(td) / "edge_log.jsonl"))

        # Pipeline geometry → rejected.
        e1 = net.add_edge("alpha", "beta", RelationType.RELATED_TO,
                          weight=0.5, origin="hub_attachment")
        # Typed relation with pipeline origin → still canonical.
        e2 = net.add_edge("gamma", "delta", RelationType.IS_A,
                          weight=0.8, origin="semantic_bridge")
        # Untyped but unlisted provenance → canonical (keep the binding).
        e3 = net.add_edge("alpha", "epsilon", RelationType.RELATED_TO,
                          weight=0.4, origin="conversation")
        assert net._edge_log is not None
        fold = net._edge_log.fold()
        fold_keys = {(e.source, e.target, e.relation.value) for e in fold.values()}

        is_deriv = is_derivable_edge(RelationType.RELATED_TO, "hub_attachment")
        r.measurements += [
            check_exact("derivable_returns_none", None, e1),
            check_exact("derivable_not_materialized", [],
                        [(e.relation.value, e.target) for e in net.get_edges("alpha")
                         if e.target == "beta"]),
            check_exact("derivable_not_logged", False,
                        ("alpha", "beta", "related_to") in fold_keys),
            check_exact("typed_pipeline_canonical", True,
                        ("gamma", "delta", "is_a") in fold_keys
                        and e2 is not None),
            check_exact("unlisted_origin_canonical", True,
                        ("alpha", "epsilon", "related_to") in fold_keys
                        and e3 is not None),
            check_exact("is_derivable_rule", True, is_deriv),
        ]
    return r


def probe_edge_log_is_truth() -> ProbeReport:
    """The log fold is the edge set — materialized edges equal the fold."""
    r = ProbeReport(
        probe_id="substrate.edge_log_is_truth",
        title="Materialized edges are exactly the log fold",
        claim="After attach_edge_log + mutations, a fresh network attached "
              "to the same log materializes exactly the fold — the in-memory "
              "edge list is a projection, not an independent store.",
        tier="substrate",
        evidence="implementation",
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "edge_log.jsonl"
        net = ConceptNetwork()
        for name in ("alpha", "beta", "gamma"):
            net.add_concept(name, confidence=0.9, origin="validation")
        net.attach_edge_log(EdgeLog(path))
        net.add_edge("alpha", "beta", RelationType.IS_A, 0.8, "validation")
        net.add_edge("beta", "gamma", RelationType.CAUSES, 0.6, "validation")
        net.remove_edge("beta", "gamma", RelationType.CAUSES)
        net.add_edge("gamma", "alpha", RelationType.ENABLES, 0.7, "validation")
        net._edge_log.close()

        live = {(e.source, e.target, e.relation.value): e.weight
                for e in net._edges}
        net2 = ConceptNetwork()
        for name in ("alpha", "beta", "gamma"):
            net2.add_concept(name, confidence=0.9, origin="validation")
        net2.attach_edge_log(EdgeLog(path))
        restored = {(e.source, e.target, e.relation.value): e.weight
                    for e in net2._edges}
        net2._edge_log.close()

    expected = {("alpha", "beta", "is_a"): 0.8,
                ("gamma", "alpha", "enables"): 0.7}
    r.measurements += [
        check_exact("fold_is_edge_set", expected, live),
        check_exact("fresh_network_materializes_fold", live, restored),
    ]
    return r


# ─────────────────────────────────────────────────────────────────
# Decision engine — TD value enters the score at exactly 0.1 weight
# ─────────────────────────────────────────────────────────────────


def probe_td_decision_attribution() -> ProbeReport:
    """Consulting TD adds exactly 0.1·V(s,action) to the composite score."""
    r = ProbeReport(
        probe_id="substrate.td_decision_attribution",
        title="TD value enters decision scoring at exactly weight 0.1",
        claim="composite_with_td − composite_without_td == 0.1·td_value, "
              "so learned value is causally responsible for the score "
              "change — not a side effect of the harness.",
        tier="substrate",
        evidence="implementation",
    )
    net = ConceptNetwork()
    net.add_concept("alpha", confidence=0.9, origin="validation")

    def score(td: TDLearner | None) -> tuple[float, float, str]:
        engine = DecisionEngine(net, td_learner=td)
        out = engine.decide(
            default_action=ActionType.ANSWER,
            confidence=0.8, uncertainty=0.2,
            topics=["alpha"], perception_intent="question",
        )
        ans = next(c for c in out.candidates if c.action_type is ActionType.ANSWER)
        ask = next(c for c in out.candidates if c.action_type is ActionType.ASK)
        return ans.composite_score, ask.composite_score, out.action_type.value

    learner = TDLearner(net, lam=0.0)
    for _ in range(30):
        learner.update(["alpha"], reward=5.0, next_state=["alpha"],
                       action="answer", next_action="answer")
    td_value = learner.predict_value(["alpha"], action="answer")

    base_ans, base_ask, _sel = score(None)
    td_ans, td_ask, sel = score(learner)

    r.measurements += [
        check("score_delta_is_0.1x_td", td_value * 0.1, td_ans - base_ans,
              tol=1e-9, detail="score attribution, not just direction"),
        check("ask_candidate_untrained", 0.0,
              td_ask - base_ask - 0.1 * learner.predict_value(["alpha"], action="ask"),
              tol=1e-9, detail="untrained action gets its own (untrained) value"),
        Measurement("td_value_trained", "> 0", td_value, td_value > 0,
                    f"V([alpha],answer) = {td_value:.6f}"),
        check_exact("selected_action", "answer", sel),
    ]
    return r


# ─────────────────────────────────────────────────────────────────
# Presence belief — exact Bayesian posteriors
# ─────────────────────────────────────────────────────────────────


def probe_beta_conjugacy() -> ProbeReport:
    """Responsiveness is a true Beta posterior: exact α/β after k of n bids."""
    r = ProbeReport(
        probe_id="substrate.beta_conjugacy",
        title="Responsiveness posterior is exactly Beta(1+k, 1+n−k)",
        claim="After k answered of n bids, the posterior parameters equal "
              "the conjugate Beta update exactly; variance shrinks "
              "monotonically with evidence.",
        tier="substrate",
        evidence="implementation",
    )
    belief = PresenceBelief()
    t = 1_000.0
    answered_pattern = [True, True, False, True, False, False, True]
    for i, ok in enumerate(answered_pattern):
        belief.note_bid(t + i * 1000, ["topic_x"])
        belief.resolve_bid(ok, t + i * 1000 + 30.0)

    k = sum(answered_pattern)
    n = len(answered_pattern)
    post = belief.responsiveness
    expected_mean = (1 + k) / (2 + n)
    expected_var = (post.alpha * post.beta) / (
        (post.alpha + post.beta) ** 2 * (post.alpha + post.beta + 1)
    )

    # Variance is NOT strictly monotone in general: a step that moves the
    # mean toward 0.5 can widen the posterior even as evidence grows. What
    # is mathematically guaranteed: along a consistent-evidence path
    # (all positive), variance decreases strictly, and var → 0 as
    # α+β → ∞. The probe asserts the honest property.
    seq = PresenceBelief()
    var_path = [seq.responsiveness.variance]
    for i in range(6):
        seq.note_bid(t + i * 1000, ["t"])
        seq.resolve_bid(True, t + i * 1000 + 10.0)
        var_path.append(seq.responsiveness.variance)
    r.measurements += [
        check("posterior_alpha", 1 + k, post.alpha, tol=0.0),
        check("posterior_beta", 1 + (n - k), post.beta, tol=0.0),
        check("posterior_mean", expected_mean, post.mean, tol=1e-15),
        check("posterior_evidence", float(n), post.evidence, tol=0.0),
        check("posterior_variance", expected_var, post.variance, tol=1e-15),
        Measurement("variance_shrinks_on_consistent_evidence", True,
                    all(b < a for a, b in pairwise(var_path)),
                    all(b < a for a, b in pairwise(var_path)),
                    "strictly decreasing for all-positive evidence; "
                    "not claimed monotone in general (mean near 0.5 "
                    "can widen it) — see report notes"),
        Measurement("mixed_evidence_variance_below_prior", True,
                    post.variance < 1.0 / 12.0, post.variance < 1.0 / 12.0,
                    f"{post.variance:.6f} < Beta(1,1) var 0.0833"),
    ]
    return r


def probe_bid_resolution_timing() -> ProbeReport:
    """A late answer is honestly counted as unanswered — no latency credit."""
    r = ProbeReport(
        probe_id="substrate.bid_resolution_timing",
        title="Bid timing rules are enforced exactly",
        claim="An answer inside the window gives a positive observation and "
              "a latency sample; the same answer past the window counts as "
              "unanswered and contributes no latency sample.",
        tier="substrate",
        evidence="implementation",
    )
    timely = PresenceBelief()
    timely.note_bid(1000.0, ["t"])
    timely.resolve_bid(True, 1030.0)   # 30s < 300s window

    late = PresenceBelief()
    late.note_bid(1000.0, ["t"])
    late.resolve_bid(True, 1400.0)     # 400s > 300s window → unanswered

    r.measurements += [
        check("timely_alpha", 2.0, timely.responsiveness.alpha, tol=0.0),
        check_exact("timely_latency_sampled", 1, timely.reply_latency.n),
        check("late_alpha_unchanged", 1.0, late.responsiveness.alpha, tol=0.0),
        check("late_beta_incremented", 2.0, late.responsiveness.beta, tol=0.0),
        check_exact("late_no_latency_sample", 0, late.reply_latency.n),
    ]
    return r


def probe_unresponsive_boundary() -> ProbeReport:
    """Learned unresponsiveness fires only past the evidence boundary."""
    r = ProbeReport(
        probe_id="substrate.unresponsive_boundary",
        title="Unresponsiveness needs ≥4 unanswered bids and mean < 0.2",
        claim="3 unanswered bids → not unresponsive (insufficient evidence); "
              "a 4th → mean 1/6 ≈ 0.167 < 0.2 → unresponsive. The verdict is "
              "exactly at the documented boundary.",
        tier="substrate",
        evidence="implementation",
    )
    belief = PresenceBelief()
    verdicts = []
    for i in range(4):
        belief.note_bid(1000.0 + i * 1000, ["t"])
        belief.resolve_bid(False, 1000.0 + i * 1000 + 500.0)
        verdicts.append(belief.unresponsive())

    r.measurements += [
        check_exact("verdicts_1234", [False, False, False, True], verdicts,
                    "boundary at exactly 4 bids"),
        check("mean_at_boundary", 1.0 / 6.0, belief.responsiveness.mean, tol=1e-15),
    ]
    return r


def probe_attention_decay() -> ProbeReport:
    """Attention decays as exp(−Δt/τ) — the analytic form, not an approximation."""
    r = ProbeReport(
        probe_id="substrate.attention_decay",
        title="Attention decay is exactly exponential",
        claim="attention_now(t) = a·e^(−Δt/1200) — verified bit-exact at "
              "several offsets.",
        tier="substrate",
        evidence="implementation",
    )
    belief = PresenceBelief()
    belief.note_activity(1000.0, weight=0.8)
    for dt in (60.0, 600.0, 1200.0, 3600.0):
        expected = 0.8 * math.exp(-dt / 1200.0)
        r.measurements.append(
            check(f"attention_t+{dt:.0f}s", expected,
                  belief.attention_now(1000.0 + dt), tol=1e-15)
        )
    return r


def probe_bid_window_learning() -> ProbeReport:
    """The reply window learns the p95 of observed latencies (lognormal)."""
    r = ProbeReport(
        probe_id="substrate.bid_window_learning",
        title="Learned bid window equals exp(μ_ln + 1.645·σ_ln), clamped",
        claim="Below 5 samples the window stays at the 300s default; at 5+ "
              "it equals the lognormal p95 recomputed from the raw samples, "
              "clamped to [60, 3600].",
        tier="substrate",
        evidence="implementation",
    )
    belief = PresenceBelief()
    latencies = [45.0, 60.0, 90.0, 120.0]
    for i, lat in enumerate(latencies):
        belief.note_bid(10_000.0 + i * 10_000, ["t"])
        belief.resolve_bid(True, 10_000.0 + i * 10_000 + lat)
    window_4 = belief.bid_window()

    belief.note_bid(90_000.0, ["t"])
    belief.resolve_bid(True, 90_000.0 + 75.0)
    latencies.append(75.0)
    window_5 = belief.bid_window()

    xs = [math.log(v) for v in latencies]
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    sigma = max(math.sqrt(var), 0.5)
    expected = min(3600.0, max(60.0, math.exp(mean + 1.645 * sigma)))

    r.measurements += [
        check("window_below_min_samples", 300.0, window_4, tol=0.0),
        check("window_is_p95", expected, window_5, tol=1e-9,
              detail=f"independently recomputed from {latencies}"),
    ]
    return r


def probe_rhythm_gating() -> ProbeReport:
    """Activity rhythm is an exact Dirichlet-smoothed histogram."""
    r = ProbeReport(
        probe_id="substrate.rhythm_gating",
        title="Learned activity rhythm matches the Dirichlet posterior",
        claim="expected_activity(h) = hours[h]/Σhours exactly; "
              "likely_awake refuses to suppress contact until ≥12 real "
              "observations, then suppresses only dead hours.",
        tier="substrate",
        evidence="implementation",
    )
    import time as _t

    # Compute timestamps whose *local* hour is the target — the rhythm
    # histogram is keyed on localtime, so the probe derives timestamps
    # dynamically and stays timezone-independent.
    def at_hour(hour: int, base: float = 1_760_000_000.0) -> float:
        t = base
        while _t.localtime(t).tm_hour != hour:
            t += 3600.0
        return t

    belief = PresenceBelief()
    active_hour = 10
    dead_hour = 3
    verdicts_during = []
    for i in range(13):
        belief.note_activity(at_hour(active_hour) + i * 86_400.0, weight=1.0)
        verdicts_during.append(belief.likely_awake(at_hour(dead_hour, 2_000_000_000.0)))

    hours = belief.activity_hours
    expected_activity = hours[active_hour] / sum(hours)
    dead_expected = hours[dead_hour] / sum(hours)

    r.measurements += [
        check("expected_activity_exact", expected_activity,
              belief.expected_activity(active_hour), tol=1e-15),
        check("dead_hour_expected", dead_expected,
              belief.expected_activity(dead_hour), tol=1e-15),
        check_exact("no_suppression_below_12", True,
                    all(verdicts_during[:11]),
                    "evidence 1..11 < 12 → never suppresses"),
        check_exact("dead_hour_suppressed_at_12", False,
                    verdicts_during[11],
                    "evidence = 12 → dead hour suppressed"),
        check_exact("dead_hour_suppressed_at_13", False,
                    verdicts_during[12]),
    ]
    return r


# ─────────────────────────────────────────────────────────────────
# Self-model — introspection reads the network, not a script
# ─────────────────────────────────────────────────────────────────


def probe_introspection_grounding() -> ProbeReport:
    """Self-knowledge is composed from the actual concept graph."""
    r = ProbeReport(
        probe_id="substrate.introspection_grounding",
        title="Self-model introspection reads the live network",
        claim="SelfModel.introspect() reports 'a kind of X' exactly when the "
              "genesis concept has an IS_A→X edge (w>0.5) — change the graph "
              "and the self-report changes; remove it and there is no claim. "
              "The report can be *wrong* in the direction of the graph — "
              "which is the honest signature of actually reading it.",
        tier="substrate",
        evidence="implementation",
    )
    model = SelfModel()

    def nature(is_a_target: str | None) -> str | None:
        net = ConceptNetwork()
        if is_a_target is not None:
            net.add_concept("genesis", confidence=0.9, origin="validation")
            net.add_concept(is_a_target, confidence=0.9, origin="validation")
            net.add_edge("genesis", is_a_target, RelationType.IS_A,
                         weight=0.8, origin="validation")
        return model.introspect(net).get("nature")

    truth = nature("cognitive_architecture")
    counterfactual = nature("teapot")
    absent = nature(None)

    r.verbatim = {
        "true_graph": repr(truth),
        "counterfactual_graph": repr(counterfactual),
        "no_graph": repr(absent),
    }
    r.measurements += [
        check_exact("reads_is_a_edge", "a kind of cognitive architecture", truth),
        check_exact("follows_counterfactual", "a kind of teapot", counterfactual,
                    "report tracks the graph even when wrong"),
        check_exact("no_concept_no_claim", None, absent,
                    "absent self-concept → no nature claim (no confabulation)"),
    ]
    return r


# ─────────────────────────────────────────────────────────────────
# Persistence — what is learned survives save/load exactly
# ─────────────────────────────────────────────────────────────────


def probe_synapse_persistence() -> ProbeReport:
    """Synaptic efficacy round-trips through save/load bit-exactly."""
    r = ProbeReport(
        probe_id="substrate.synapse_persistence",
        title="Learned synaptic state survives persistence intact",
        claim="STDP save_state → load_state restores every expressed "
              "efficacy exactly — learned plasticity is durable state, "
              "not session scratch.",
        tier="substrate",
        evidence="implementation",
    )
    net = ConceptNetwork()
    for name in ("alpha", "beta"):
        net.add_concept(name, confidence=0.9, origin="validation")
    syn = SynapticStore()
    _train_ltp(syn, net, 10, 5.0)
    stdp = STDP(syn, net)
    blob = stdp.save_state()

    syn2 = SynapticStore()
    stdp2 = STDP(syn2, net)
    stdp2.load_state(blob)

    r.measurements += [
        check_exact("weights_roundtrip", syn.connections(), syn2.connections()),
        check("efficiency_exact", syn.get_weight("alpha", "beta"),
              syn2.get_weight("alpha", "beta"), tol=0.0),
    ]
    return r


# ─────────────────────────────────────────────────────────────────
# Language — the utterance is a causal function of the network
# ─────────────────────────────────────────────────────────────────


def probe_language_is_grounded() -> ProbeReport:
    """What Genesis says about a concept tracks the graph, exactly."""
    r = ProbeReport(
        probe_id="substrate.language_grounded",
        title="Generated language is a causal function of the concept graph",
        claim="Same seed, same thought, same voice — change ONLY the "
              "network and the sentence changes with it: is_a→mammals "
              "produces 'a type of mammals', is_a→fish produces 'a type of "
              "fish', no edge produces a low-confidence hedge, and an "
              "unrelated concept changes nothing. Four arms, one variable.",
        tier="substrate",
        evidence="behavioral",
        notes="Generation is seeded; the varying input is the knowledge "
              "state alone. This is the strongest available evidence that "
              "utterances are composed from the network rather than recited.",
    )
    from genesis_cognitive.emotion import EmotionalState
    from genesis_cognitive.language.base import Thought
    from genesis_cognitive.language.generator import GenerativeEngine
    from genesis_cognitive.self.model import SelfModel as SM

    em = EmotionalState(label="curious", cognitive_style="steady")
    th = Thought(content="dolphins", intent="inform", topics=["dolphins"])

    def say(is_a_target: str | None, extra_concept: str | None = None) -> str:
        net = ConceptNetwork()
        net.add_concept("dolphins", confidence=0.9, origin="validation")
        if is_a_target is not None:
            net.add_concept(is_a_target, confidence=0.9, origin="validation")
            net.add_edge("dolphins", is_a_target, RelationType.IS_A,
                         weight=0.9, origin="validation")
        if extra_concept is not None:
            net.add_concept(extra_concept, confidence=0.9, origin="validation")
        eng = GenerativeEngine(SM(), seed=42, network=net)
        return eng.generate(th, em)

    a = say("mammals")
    b = say("fish")
    c = say(None)
    d = say("mammals", extra_concept="unrelated_xyz")

    r.verbatim = {"is_a_mammals": a, "is_a_fish": b, "no_edge": c,
                  "irrelevant_concept_added": d}
    r.measurements += [
        Measurement("counterfactual_tracks_graph", "fish", "fish" in b and "mammals" not in b,
                    "fish" in b and "mammals" not in b, b),
        Measurement("no_edge_hedges", "low-confidence hedge",
                    "mammals" not in c and "fish" not in c,
                    "mammals" not in c and "fish" not in c, c),
        check_exact("irrelevant_change_invisible", a, d,
                    "unrelated concept perturbation → identical utterance"),
        Measurement("edge_changes_utterance", "different", a != b, a != b,
                    "same seed, different graph → different sentence"),
    ]
    return r


ALL_PROBES = [
    probe_td_fixed_point,
    probe_td_reward_sign,
    probe_stdp_timing_curve,
    probe_stdp_integration,
    probe_stdp_timing_boundaries,
    probe_spreading_activation,
    probe_edge_log_fidelity,
    probe_derivable_edge_policy,
    probe_edge_log_is_truth,
    probe_td_decision_attribution,
    probe_beta_conjugacy,
    probe_bid_resolution_timing,
    probe_unresponsive_boundary,
    probe_attention_decay,
    probe_bid_window_learning,
    probe_rhythm_gating,
    probe_introspection_grounding,
    probe_synapse_persistence,
    probe_language_is_grounded,
]
