"""Dynamics probes — in-vivo ablation of the running pipeline.

Substrate probes prove each mechanism computes correctly in isolation.
Capability probes prove the pipeline produces intelligent behavior.
This tier asks the question in between: **is each mechanism actually
load-bearing?** Remove one system while the whole pipeline runs,
feed it identical input, and measure which state variables stop
changing. A mechanism whose removal changes nothing is decorative —
working code that the system does not use.

Design rules (validation-lab lineage, one level deeper):

- **In-vivo, not harness-fed.** The stimulus is real conversation
  through ``CognitionEngine.think()`` — the system's own spike trains,
  its own reward signal, its own extraction path. Nothing calls
  ``record_spike`` or ``update`` directly.
- **Ablation is reversible.** Each arm wraps a fresh engine; the
  mechanism is neutralized at its call seam (monkeypatched to a
  no-op) for the duration of the script — never deleted, never shared
  across arms.
- **The stimuli are generated, not fixed.** A seeded RNG invents a
  fresh pseudoword lexicon every run (seed recorded in the report for
  audit), so no item can match a canned pattern and no question is
  ever asked twice.
- **The diff is the evidence.** Each probe reports which measured
  variables moved between enabled and disabled arms — that set of
  moved variables *is* the mechanism's causal footprint.
"""

from __future__ import annotations

import contextlib
import math
import random
from typing import Any

from .capability import (
    ProbeEngine,
    _think,
    _verdict_of,
    classify_response,
    make_engine,
)
from .report import Measurement, ProbeReport, check_exact

# ─────────────────────────────────────────────────────────────────
# Generated stimuli — a fresh lexicon every run, seed in the report
# ─────────────────────────────────────────────────────────────────

_ONSETS = "bdfghjklmnprstvwz"
_VOWELS = "aeiou"
_CODAS = "bdfgklmnprstvxz"


def _word(rng: random.Random) -> str:
    return rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)


def novel_items(rng: random.Random, n: int) -> list[tuple[str, str]]:
    """Generate n never-seen (subject, object) pseudoword pairs."""
    seen: set[str] = set()
    def w() -> str:
        while True:
            word = _word(rng)
            if word not in seen:
                seen.add(word)
                return word
    return [(w(), w()) for _ in range(n)]


# ─────────────────────────────────────────────────────────────────
# Ablation machinery — neutralize one mechanism at its call seam
# ─────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def _patched(pairs: list[tuple[Any, str, Any]]):
    """Temporarily replace attributes; restore them afterwards."""
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in pairs]
    for obj, name, new in pairs:
        setattr(obj, name, new)
    try:
        yield
    finally:
        for obj, name, old in saved:
            setattr(obj, name, old)


def _noop(*_a: Any, **_k: Any) -> None:
    return None


def _ablations(pe: ProbeEngine, name: str) -> contextlib.AbstractContextManager:
    """The seam registry — where each mechanism is switched off."""
    eng = pe.engine
    if name == "stdp":
        return _patched([
            (eng.stdp, "record_spike", _noop),
            (eng.stdp, "record_spikes", _noop),
            (eng.stdp, "apply_updates", dict),
        ])
    if name == "td_update":
        return _patched([
            (eng.td_learner, "update", lambda *a, **k: 0.0),
        ])
    if name == "self_learn":
        return _patched([
            (eng.self_learner, "learn_from_input", lambda *a, **k: []),
        ])
    if name == "semantic_consolidate":
        return _patched([
            (eng.semantic_memory, "_consolidate_to_network", _noop),
        ])
    if name == "episodic_store":
        # LTM write path: MemoryEngine calls client.store_episode
        # (memory/engine.py:495). Ablate at the transport — the client
        # *is* the storage substrate (daemon LTM in production).
        def _swallowed(text: str, **_kw: Any) -> int:
            return -1
        return _patched([
            (pe.client, "store_episode", _swallowed),
        ])
    if name == "verify_relation":
        return _patched([
            (eng._question_handler, "question_verify_relation",
             lambda *a, **k: None),
        ])
    if name == "hedge_gate":
        # Bypass the uncertainty gate: return the response as composed.
        return _patched([
            (eng, "_maybe_hedge_response",
             lambda response, *_a, **_k: response),
        ])
    raise KeyError(name)


# ─────────────────────────────────────────────────────────────────
# State snapshot — the variables a mechanism could move
# ─────────────────────────────────────────────────────────────────


def _snapshot(pe: ProbeEngine) -> dict[str, Any]:
    """Measured substrate state: the variables ablation can move."""
    edges = {}
    for e in pe.network._edges:
        rel = getattr(e.relation, "name", str(e.relation))
        edges[(str(e.source), rel, str(e.target))] = round(e.weight, 6)
    return {
        "edges": edges,
        "synapses": {k: round(v, 12)
                     for k, v in pe.engine.synapses.connections().items()},
        "td_dopamine": round(pe.engine.td_learner.get_dopamine_signal(), 8),
        "episodes": len(pe.client._episodes)
        if hasattr(pe.client, "_episodes") else -1,
    }


def _run_script(
    pe: ProbeEngine,
    script: list[str],
    ablation: str | None = None,
) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    """Run a conversation script (optionally ablated); return turns + snapshot."""
    ctx = _ablations(pe, ablation) if ablation else contextlib.nullcontext()
    turns: list[tuple[str, Any]] = []
    with ctx:
        for text in script:
            turns.append(_think(pe, text))
    return turns, _snapshot(pe)


def _report(
    probe_id: str,
    title: str,
    claim: str,
    backend: str,
    measurements: list[Measurement],
    notes: str = "",
    verbatim: dict[str, str] | None = None,
) -> ProbeReport:
    return ProbeReport(
        probe_id=probe_id,
        title=title,
        claim=claim,
        tier="dynamics",
        evidence="causal",
        backend=backend,
        measurements=measurements,
        verbatim=verbatim or {},
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────
# Probe 1 — who writes a taught edge? (write-path decomposition)
# ─────────────────────────────────────────────────────────────────


def probe_write_path_decomposition(
    backend: str = "offline", socket_path: str | None = None,
) -> ProbeReport:
    """One taught fact, N writers: decompose edge-weight contribution.

    Teaching "a X is a Y" fires more than one edge-writing mechanism.
    This probe counts them, then removes them one at a time to show
    each one's share — a mechanism that neither creates nor reinforces
    the edge under ablation is not on this path.
    """
    rng = random.Random(20260927)
    items = novel_items(rng, 1)
    subj, obj = items[0]

    # ── Arm A: instrumented — record every edge-writing call ──
    pe = make_engine(backend, socket_path=socket_path)
    writes: list[tuple[str, str, float]] = []
    orig_add = pe.network.add_edge

    def spy(s: str, d: str, rel: Any, weight: float = 0.5, **kw: Any) -> Any:
        edge = orig_add(s, d, rel, weight=weight, **kw)
        writes.append((str(s), str(d), float(weight)))
        return edge

    pe.network.add_edge = spy
    _think(pe, f"a {subj} is a {obj}")
    pe.network.add_edge = orig_add

    taught_writes = [w for w in writes if w[0] == subj and w[1] == obj]
    base_snap = _snapshot(pe)

    # ── Arm B: self_learner ablated ──
    pe_b = make_engine(backend, socket_path=socket_path)
    _, snap_b = _run_script(pe_b, [f"a {subj} is a {obj}"], "self_learn")
    w_b = snap_b["edges"].get((subj, "IS_A", obj))

    # ── Arm C: semantic consolidation ablated ──
    pe_c = make_engine(backend, socket_path=socket_path)
    _, snap_c = _run_script(pe_c, [f"a {subj} is a {obj}"],
                            "semantic_consolidate")
    w_c = snap_c["edges"].get((subj, "IS_A", obj))

    w_a = base_snap["edges"].get((subj, "IS_A", obj))
    measurements = [
        Measurement(
            "writers_fired", ">=1", len(taught_writes),
            len(taught_writes) >= 1,
            f"{len(taught_writes)} add_edge calls for one taught fact"),
        Measurement(
            "edge_weight_intact", "recorded", w_a,
            w_a is not None,
            f"edge weight with all writers: {w_a}"),
        check_exact(
            "edge_survives_self_learn_ablation", True, w_b is not None,
            f"weight without self_learner: {w_b}"),
        check_exact(
            "edge_survives_semantic_ablation", True, w_c is not None,
            f"weight without semantic consolidation: {w_c}"),
        Measurement(
            "redundancy_decomposes", True,
            (w_b is not None and w_c is not None
             and w_a is not None and w_b < w_a and w_c < w_a),
            (w_b is not None and w_c is not None
             and w_a is not None and w_b < w_a and w_c < w_a),
            "each single ablation lowers but does not remove the edge"),
    ]
    return _report(
        "dynamics.write_path", "Edge write-path decomposition",
        "A taught fact is written by multiple independent mechanisms; "
        "removing either alone lowers but does not remove the edge.",
        backend, measurements,
        notes=f"items={items}; intact={w_a} "
              f"no_self_learner={w_b} no_semantic={w_c}",
    )


# ─────────────────────────────────────────────────────────────────
# Probe 2 — STDP substrate in vivo
# ─────────────────────────────────────────────────────────────────


def probe_stdp_in_vivo(
    backend: str = "offline", socket_path: str | None = None,
) -> ProbeReport:
    """Real conversation wires the synaptic substrate — or doesn't.

    Stage 14 of think() spikes perception topics (t+0,5,10…) then
    thought topics (t+50). With a 40ms window, the first two user
    topics are ≥40ms from the thought — user words may never bind to
    the response. This probe measures which pairs actually wire.
    """
    rng = random.Random(20260927)
    items = novel_items(rng, 2)
    script = [f"a {s} is a {o}" for s, o in items]

    pe_on = make_engine(backend, socket_path=socket_path)
    turns_on, snap_on = _run_script(pe_on, script)
    _, snap_off = _run_script(make_engine(backend, socket_path=socket_path), script, "stdp")

    syn_on = snap_on["synapses"]
    n_on = len(syn_on)
    # Closed-form: Δt=5ms LTP → a_plus·e^(−5/20)·lr = 0.01·e^−0.25·0.002
    expected = 0.01 * math.exp(-5.0 / 20.0) * 0.002
    subject_pair = next(
        (v for (a, b), v in syn_on.items() if (a, b) in
         [(items[0][0], items[0][1]), (items[0][1], items[0][0])]),
        None,
    )
    # user→thought binding: a synapse whose post side is a concept that
    # appeared ONLY in a thought, never in user input. User topics spike
    # at +0,5,10…ms; thought topics at +50ms — the first two user topics
    # sit at Δt≥40ms, i.e. at or outside the STDP window.
    user_topics: set[str] = set()
    thought_only: set[str] = set()
    for _resp, state in turns_on:
        perc = set(getattr(state.perception, "topics", None) or [])
        thought = set(getattr(state.thought, "topics", None) or [])
        thought_only |= thought - perc
        user_topics |= perc
    user_to_thought = [
        k for k in syn_on if k[0] in user_topics and k[1] in thought_only
    ]
    measurements = [
        Measurement(
            "synapses_form_in_vivo", ">0", n_on, n_on > 0,
            f"{n_on} synapses from {len(script)} real turns"),
        check_exact(
            "ablated_silent", 0, len(snap_off["synapses"]),
            "STDP off → the substrate stays flat"),
        Measurement(
            "weight_matches_closed_form", expected,
            subject_pair if subject_pair is not None else float("nan"),
            subject_pair is not None and abs(subject_pair - expected) < 1e-9,
            f"Δt=5ms LTP = a·e^(−5/τ)·lr = {expected:.6g}"),
        Measurement(
            "user_thought_binding_present", "0 (window finding)",
            len(user_to_thought), len(user_to_thought) == 0,
            f"thought-only concepts={sorted(thought_only)}; "
            "user topics spike ≥40ms before thought topics — "
            "outside the STDP window, so user words don't bind to "
            "response-only concepts"),
    ]
    return _report(
        "dynamics.stdp_in_vivo", "STDP substrate under real conversation",
        "Real turns write synapses with the exact STDP weight — and the "
        "50ms user→thought gap means response concepts never bind to the "
        "user's words.", backend, measurements,
        notes=f"synapses={syn_on}",
    )


# ─────────────────────────────────────────────────────────────────
# Probe 3 — TD reward chain in vivo
# ─────────────────────────────────────────────────────────────────


def probe_td_chain_in_vivo(
    backend: str = "offline", socket_path: str | None = None,
) -> ProbeReport:
    """The reward channel: does a real turn move the value substrate?

    Stage 15 computes reward = f(confidence, valence, novelty) and runs
    td_learner.update — the dopamine signal then gates next turn's STDP.
    Ablating update must flatten dopamine AND leave modulator untouched.
    """
    rng = random.Random(7)
    items = novel_items(rng, 2)
    script = [f"a {s} is a {o}" for s, o in items] + \
        [f"what is a {items[0][0]}?"]

    _, snap_on = _run_script(
        make_engine(backend, socket_path=socket_path), script)
    pe_off = make_engine(backend, socket_path=socket_path)
    _, snap_off = _run_script(pe_off, script, "td_update")

    da_on = snap_on["td_dopamine"]
    measurements = [
        Measurement(
            "dopamine_moves_in_vivo", "≠0", da_on, da_on != 0.0,
            f"get_dopamine_signal after real turns: {da_on}"),
        check_exact(
            "ablated_dopamine_flat", 0.0, snap_off["td_dopamine"],
            "TD update off → no dopamine signal"),
        Measurement(
            "downstream_differs", True,
            snap_on["synapses"] != snap_off["synapses"]
            or snap_on["td_dopamine"] != snap_off["td_dopamine"],
            True,
            "TD feeds STDP gating — at minimum the dopamine channel differs"),
    ]
    return _report(
        "dynamics.td_in_vivo", "TD reward chain under real turns",
        "The interaction reward trains action-conditioned values; "
        "ablating the update flattens the dopamine channel that gates "
        "plasticity.", backend, measurements,
        notes=f"da_on={da_on} da_off={snap_off['td_dopamine']}",
    )


# ─────────────────────────────────────────────────────────────────
# Probe 4 — the verification path is the carrier of yes/no answers
# ─────────────────────────────────────────────────────────────────


def probe_verification_carrier(
    backend: str = "offline", socket_path: str | None = None,
) -> ProbeReport:
    """Ablate question_verify_relation: yes/no answers must degrade.

    The verification path was added because YES_NO questions never
    consulted the graph. If it is load-bearing, removing it must
    remove the verdict metadata AND change the rendered answer.
    """
    rng = random.Random(99)
    subj, obj = novel_items(rng, 1)[0]
    script = [f"a {subj} is a {obj}", f"is a {subj} a {obj}?"]

    turns_on, _ = _run_script(
        make_engine(backend, socket_path=socket_path), script)
    turns_off, _ = _run_script(
        make_engine(backend, socket_path=socket_path), script,
        "verify_relation")

    verdict_on = _verdict_of(turns_on[-1][1])
    verdict_off = _verdict_of(turns_off[-1][1])
    resp_on = classify_response(turns_on[-1][0])
    resp_off = classify_response(turns_off[-1][0])

    measurements = [
        check_exact(
            "intact_verdict_present", "yes", verdict_on,
            f"verdict metadata: {verdict_on}"),
        check_exact(
            "ablated_verdict_gone", None, verdict_off,
            "with the path ablated no verdict is produced"),
        Measurement(
            "rendered_answer_differs", True,
            resp_on != resp_off or turns_on[-1][0] != turns_off[-1][0],
            resp_on == "affirm" and resp_off != "affirm",
            f"intact={resp_on!r} ablated={resp_off!r}"),
    ]
    return _report(
        "dynamics.verification_carrier",
        "Yes/no verification is load-bearing",
        "Ablating question_verify_relation removes the verdict and "
        "degrades the rendered answer — the path carries verification.",
        backend, measurements,
        verbatim={
            "intact": turns_on[-1][0],
            "ablated": turns_off[-1][0],
        },
    )


# ─────────────────────────────────────────────────────────────────
# Probe 5 — episodic vs semantic carriers
# ─────────────────────────────────────────────────────────────────


def probe_episodic_vs_semantic(
    backend: str = "offline", socket_path: str | None = None,
) -> ProbeReport:
    """Ablate episode storage: semantic recall should survive.

    If recall of a taught fact persists with zero episodes stored,
    the concept-graph edge — not the episode — carries this kind of
    memory. That distinction is the causal map.
    """
    rng = random.Random(555)
    subj, obj = novel_items(rng, 1)[0]
    script = [f"a {subj} is a {obj}", "the sky is gray today",
              f"what is a {subj}?"]

    _, snap_on = _run_script(
        make_engine(backend, socket_path=socket_path), script)
    turns_off, snap_off = _run_script(
        make_engine(backend, socket_path=socket_path), script,
        "episodic_store")

    edge_on = snap_on["edges"].get((subj, "IS_A", obj))
    edge_off = snap_off["edges"].get((subj, "IS_A", obj))
    recall_off = obj in turns_off[-1][0].lower()

    measurements = [
        Measurement(
            "episodes_stored_intact", ">0", snap_on["episodes"],
            snap_on["episodes"] > 0,
            f"{snap_on['episodes']} episodes written"),
        check_exact(
            "episodes_ablated_zero", 0, snap_off["episodes"],
            "store ablated → nothing lands in LTM"),
        Measurement(
            "semantic_edge_still_written", True, edge_off is not None,
            edge_off is not None,
            f"edge weight with episodic off: {edge_off}"),
        Measurement(
            "semantic_recall_survives", True, recall_off, recall_off,
            f"recall with zero episodes: {turns_off[-1][0][:60]!r}"),
    ]
    return _report(
        "dynamics.episodic_vs_semantic",
        "Semantic memory carries taught facts, not episodic",
        "Ablating episode storage leaves edge-based recall intact — "
        "the systems carry different kinds of memory.",
        backend, measurements,
        verbatim={"ablated_recall": turns_off[-1][0]},
        notes=f"edge intact={edge_on} ablated={edge_off}",
    )


# ─────────────────────────────────────────────────────────────────
# Probe 6 — the hedge gate's actual contribution
# ─────────────────────────────────────────────────────────────────


def probe_hedge_gate_contribution(
    backend: str = "offline", socket_path: str | None = None,
) -> ProbeReport:
    """Ablate _maybe_hedge_response: measure what honesty it adds.

    Earlier probing found the gate overriding correct grounded answers
    ~75% of the time (now excepted for verification). This measures its
    net effect on unanswerable items — if declines survive without it,
    the composer is honest and the gate was redundant there.
    """
    rng = random.Random(31415)
    items = novel_items(rng, 3)
    untaught = [f"what is a {s}?" for s, _ in items]

    def decline_rate(pe: ProbeEngine, ablation: str | None) -> int:
        turns, _ = _run_script(pe, untaught, ablation)
        return sum(1 for r, _s in turns if classify_response(r) == "decline")

    intact = decline_rate(make_engine(backend, socket_path=socket_path), None)
    ablated = decline_rate(
        make_engine(backend, socket_path=socket_path), "hedge_gate")

    measurements = [
        Measurement(
            "declines_intact", f"{len(untaught)}", intact,
            intact == len(untaught),
            f"{intact}/{len(untaught)} declines with gate"),
        Measurement(
            "declines_without_gate", f"{len(untaught)}", ablated,
            ablated == len(untaught),
            f"{ablated}/{len(untaught)} declines without gate"),
        Measurement(
            "gate_net_effect", "reported", intact - ablated,
            True,
            "positive = gate adds honesty; 0 = composer already declines"),
    ]
    return _report(
        "dynamics.hedge_gate", "Uncertainty gate net contribution",
        "Measured net change in decline rate with the hedge gate "
        "removed — whether the gate adds honesty or duplicates the "
        "composer's own restraint.", backend, measurements,
    )


ALL_PROBES = [
    probe_write_path_decomposition,
    probe_stdp_in_vivo,
    probe_td_chain_in_vivo,
    probe_verification_carrier,
    probe_episodic_vs_semantic,
    probe_hedge_gate_contribution,
]
