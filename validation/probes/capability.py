"""Capability probes — does Genesis *behave* intelligently?

Substrate probes (``substrate.py``) prove each mechanism computes its
documented math. That is necessary but not sufficient for the claim the
public actually cares about. These probes instead measure
*capabilities*: observable behaviors that require the machinery to work
together end to end, through the real conversation pipeline
(``CognitionEngine.think``), on items the system was never explicitly
given as answers.

Design rules:

- Every stimulus is a *novel* pseudoword ("wug", "ferren") so no item
  can be answered from seeded knowledge — any correct answer can only
  have come through the taught relation.
- Taught and untaught items are matched: an intelligent system must
  answer what it learned AND decline what it was never taught, rather
  than confabulate.
- Responses are scored semantically, not string-matched — the language
  engine is intentionally non-deterministic, so each observation
  records both the *internal verdict* (the structured
  ``relation_verification`` metadata the question handler produces)
  and the *expressed polarity* of the rendered sentence. Disagreement
  between the two channels is itself a reported finding.
- Statistics are exact: binomial tests against a stated chance level,
  Wilson 95% intervals. No observation is dropped silently.

The engine is constructed exactly as ``Mind`` wires it (shared concept
network, canonical edge log attached, memory + language + cognition on
the same network) — except the daemon client, which is ``offline`` (the
declared in-memory stand-in) or ``live`` (a real ``GenesisClient`` over
a socket the caller provides, intended for a dedicated scratch daemon —
never the user's live instance, since teaching writes into real LTM).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any

from genesis_cognitive.cognition.engine import CognitionEngine
from genesis_cognitive.concepts import ConceptNetwork
from genesis_cognitive.concepts.edge_log import open_edge_log
from genesis_cognitive.language import GenerativeEngine
from genesis_cognitive.memory import MemoryEngine
from genesis_cognitive.self import SelfModel
from genesis_cognitive.user_profile import UserProfile

from ..offline_client import OfflineProbeClient
from .report import Measurement, ProbeReport, check_exact, check_rate

# ─────────────────────────────────────────────────────────────────
# Stimulus set — invented words only, so no answer can come from seeds
# ─────────────────────────────────────────────────────────────────
#
# (subject, object) pairs taught through natural-language statements.
# Both sides are pseudowords: "a wug is a ferren" carries zero
# pre-existing meaning; the only route to a correct answer is the edge
# the statement created.
TEACH_ITEMS: list[tuple[str, str]] = [
    ("wug", "ferren"),
    ("glorp", "sorrel"),
    ("tufa", "vessal"),
    ("kep", "mordel"),
]
# Pseudowords never presented to the system — matched controls.
UNTAUGHT_ITEMS: list[str] = ["zav", "lum", "narb", "pilf"]

# Surface markers for expressed-polarity classification. These classify
# the *rendered* sentence — the same judgment a human reader would make.
_AFFIRM_MARKERS = ("yes —", "yes,", "yes.", "yes ")
_DENY_MARKERS = ("no —", "no,", "no.", "no ", "not that i know")
_DECLINE_MARKERS = (
    "i'm not sure",
    "i don't quite follow",
    "i don't know",
    "tell me more",
    "could you say more",
    "what do you mean",
    "am i on the right track",
)


def classify_response(text: str) -> str:
    """Classify a rendered response: affirm / deny / decline / assert_other.

    Polarity markers can follow an opener ("My thought: yes — …"), so
    the whole text is scanned; "no" as a bare word is riskier (it appears
    inside "nothing"/"don't"), so denial requires the em-dash or
    punctuation form the verification composer actually emits.
    """
    low = text.lower().strip()
    if any(m in low for m in _AFFIRM_MARKERS):
        return "affirm"
    if "not that i know" in low or "no —" in low or low.startswith(("no.", "no,")):
        return "deny"
    if any(m in low for m in _DECLINE_MARKERS):
        return "decline"
    return "assert_other"


def _verdict_of(state: Any) -> str | None:
    """The internal verdict, if the question handler produced one."""
    meta = getattr(getattr(state, "thought", None), "metadata", {}) or {}
    rv = meta.get("relation_verification")
    return rv.get("verdict") if isinstance(rv, dict) else None


@dataclass
class ProbeEngine:
    """A cognitive engine wired the way Mind wires it, for probing."""

    engine: CognitionEngine
    network: ConceptNetwork
    client: Any
    data_dir: str
    backend: str


def make_engine(
    backend: str = "offline",
    *,
    data_dir: str | None = None,
    socket_path: str | None = None,
    lang_seed: int = 1,
) -> ProbeEngine:
    """Build a CognitionEngine for capability probing.

    ``backend="offline"`` uses the declared in-memory client stand-in.
    ``backend="live"`` uses a real ``GenesisClient`` over ``socket_path``
    — intended for a dedicated scratch daemon (teaching writes real
    episodes into whatever daemon the socket belongs to).
    """
    data_dir = data_dir or tempfile.mkdtemp(prefix="genesis_probe_")

    if backend == "live":
        if not socket_path:
            raise ValueError("live backend requires socket_path")
        from genesis_client.client import GenesisClient

        client: Any = GenesisClient(socket_path)
        client.connect()
    else:
        client = OfflineProbeClient()

    network = ConceptNetwork()
    # Canonical edge log on this data dir — same wiring as Mind. Edge
    # mutations write through to edge_log.jsonl, which is what makes
    # the restart probe honest: a fresh network on the same dir folds
    # the same edges back.
    network.attach_edge_log(open_edge_log(data_dir))

    self_model = SelfModel()
    memory = MemoryEngine(client, network=network, get_emotion=lambda: None)
    language = GenerativeEngine(self_model, seed=lang_seed, network=network)
    engine = CognitionEngine(
        client=client,
        self_model=self_model,
        memory=memory,
        language=language,
        network=network,
        data_dir=data_dir,
        user_profile=UserProfile(),
    )
    return ProbeEngine(engine, network, client, data_dir, backend)


def _think(pe: ProbeEngine, text: str) -> tuple[str, Any]:
    return pe.engine.think(text)


def _report(
    probe_id: str,
    title: str,
    claim: str,
    backend: str,
    notes: str = "",
) -> ProbeReport:
    return ProbeReport(
        probe_id=probe_id,
        title=title,
        claim=claim,
        tier="capability",
        evidence="behavioral",
        backend=backend,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────
# 1. Learning from experience
# ─────────────────────────────────────────────────────────────────


def probe_learn_recall(backend: str = "offline", socket_path: str | None = None,
                      replicates: int = 3) -> ProbeReport:
    """Taught novel facts are recalled; untaught items are declined.

    Teaches N pseudoword facts through natural-language statements
    ("a wug is a ferren"), then asks "what is a <subject>?" — a correct
    answer must name the taught object. Untaught matched controls must
    be declined (no invented answer). Chance of naming a specific
    arbitrary object word by luck is conservatively set at 0.25.
    """
    r = _report(
        "capability.learn_recall",
        "Learns novel facts from dialogue and recalls them",
        "Telling Genesis a novel fact in plain language creates usable "
        "knowledge: 'what is a wug?' answers with the taught object, "
        "while matched untaught items are declined rather than "
        "confabulated.",
        backend,
        notes="All four terms per item are invented words; correct "
              "answers cannot come from seeded knowledge. Scored on the "
              "rendered response containing the taught object.",
    )

    hits = misses = declines = confabulations = 0
    for _ in range(replicates):
        pe = make_engine(backend, socket_path=socket_path)
        for subj, obj in TEACH_ITEMS:
            _think(pe, f"a {subj} is a {obj}")
        for subj, obj in TEACH_ITEMS:
            resp, _state = _think(pe, f"what is a {subj}?")
            r.verbatim[f"recall:{subj}"] = resp
            if obj in resp.lower():
                hits += 1
            else:
                misses += 1
        for subj in UNTAUGHT_ITEMS:
            resp, _state = _think(pe, f"what is a {subj}?")
            r.verbatim[f"untaught:{subj}"] = resp
            cls = classify_response(resp)
            # A decline names no answer; asserting a specific object
            # for a never-taught item would be confabulation.
            if cls in ("decline",) or not any(
                w in resp.lower() for w in ("is a", "is an", "kind of")
            ):
                declines += 1
            else:
                confabulations += 1

    n_taught = replicates * len(TEACH_ITEMS)
    n_untaught = replicates * len(UNTAUGHT_ITEMS)
    r.measurements += [
        check_rate("taught_recall", hits, n_taught, 0.25, min_rate=0.5,
                   detail="response names the taught object"),
        check_rate("untaught_declined", declines, n_untaught, 0.5,
                   min_rate=0.5,
                   detail="no invented answer for never-taught items"),
        check_exact("confabulations", 0, confabulations,
                    "asserted a specific object for an untaught item"),
    ]
    return r


# ─────────────────────────────────────────────────────────────────
# 2. Verification — yes/no questions checked against the graph
# ─────────────────────────────────────────────────────────────────


def probe_verification(backend: str = "offline", socket_path: str | None = None,
                       trials: int = 4) -> ProbeReport:
    """Yes/no questions are verified against the graph, not guessed.

    Teaches a chain (wug→ferren→lorp) plus an unrelated fact
    (kep→mordel), then asks polar questions in four conditions:

    - affirmed: "is a wug a ferren?" → yes (direct edge)
    - inferred: "is a wug a lorp?" → yes (never stated; reachable
      through the is_a chain — pure generalization)
    - contradicted: "is a wug a mordel?" → no (both concepts known,
      subject carries is_a to other objects)
    - unknowable: "is a wug a brillax?" → decline (object absent)
    - unknown subject: "is a zav a ferren?" → decline

    Each trial records BOTH the internal verdict and the polarity of the
    rendered sentence; the two must agree.
    """
    r = _report(
        "capability.verification",
        "Verifies propositions against its own knowledge graph",
        "A yes/no question is answered by checking the concept network: "
        "taught relations affirm, transitively-implied relations affirm "
        "(never stated outright), contradicted relations deny with the "
        "actual alternative, and unresolvable terms decline honestly.",
        backend,
        notes="Two evidence channels per trial: the internal verdict "
              "(relation_verification metadata) and the expressed "
              "polarity of the rendered sentence.",
    )

    conditions: list[tuple[str, str, str]] = [
        # (question, expected_verdict, label)
        ("is a wug a ferren?", "yes", "affirmed_direct"),
        ("is a wug a lorp?", "yes", "affirmed_inferred"),
        ("is a wug a mordel?", "no", "contradicted"),
        ("is a wug a brillax?", None, "unknowable_object"),
        ("is a zav a ferren?", None, "unknowable_subject"),
    ]

    verdict_correct = 0
    surface_consistent = 0
    total = 0
    declines_correct = 0
    declines_total = 0

    pe = make_engine(backend, socket_path=socket_path)
    for stmt in ("a wug is a ferren", "a ferren is a lorp", "a kep is a mordel"):
        _think(pe, stmt)

    for question, expected, label in conditions:
        for _ in range(trials):
            resp, state = _think(pe, question)
            verdict = _verdict_of(state)
            surface = classify_response(resp)
            r.verbatim[f"{label}:{question}"] = resp
            total += 1

            if expected is None:
                declines_total += 1
                # Correct = no grounded verdict produced, and the
                # surface declines rather than asserting.
                if verdict is None and surface == "decline":
                    declines_correct += 1
                    verdict_correct += 1
                    surface_consistent += 1
            else:
                if verdict == expected:
                    verdict_correct += 1
                want_surface = "affirm" if expected == "yes" else "deny"
                if surface == want_surface:
                    surface_consistent += 1

    r.measurements += [
        check_rate("verdict_correct", verdict_correct, total, 0.4,
                   min_rate=0.6,
                   detail="internal verdict matches ground truth "
                          "(yes/no/decline as designed)"),
        check_rate("surface_consistent", surface_consistent, total, 0.4,
                   min_rate=0.6,
                   detail="rendered sentence expresses the same verdict"),
        check_rate("unknowable_declined", declines_correct, declines_total,
                   0.25, min_rate=0.5,
                   detail="unresolvable terms produce honest declines"),
    ]
    return r


# ─────────────────────────────────────────────────────────────────
# 3. Retention under interference
# ─────────────────────────────────────────────────────────────────


def probe_retention(backend: str = "offline", socket_path: str | None = None,
                    distractors: int = 6, replicates: int = 5) -> ProbeReport:
    """A taught fact survives intervening unrelated conversation.

    Teaches one fact, inserts K distractor turns on unrelated topics,
    then asks the recall question. Compares against a no-distractor
    arm — retention must not collapse under interference.
    """
    r = _report(
        "capability.retention",
        "Knowledge survives intervening unrelated turns",
        "A fact taught, then separated from the query by unrelated "
        "conversation, is still recalled — memory is an actual store, "
        "not an echo of the last turn.",
        backend,
    )

    distractor_lines = [
        "a kep is a mordel",
        "the weather is grey today",
        "i like listening to music",
        "what is a kep?",
        "a zundle is a provit",
        "tell me about rivers",
    ]

    hits_clean = hits_dirty = 0
    for _ in range(replicates):
        # Clean arm: teach then ask immediately.
        pe_clean = make_engine(backend, socket_path=socket_path)
        _think(pe_clean, "a wug is a ferren")
        resp, _ = _think(pe_clean, "what is a wug?")
        r.verbatim.setdefault("clean", resp)
        if "ferren" in resp.lower():
            hits_clean += 1

        # Distractor arm: teach, K unrelated turns, then ask.
        pe_dirty = make_engine(backend, socket_path=socket_path)
        _think(pe_dirty, "a wug is a ferren")
        for i in range(distractors):
            _think(pe_dirty, distractor_lines[i % len(distractor_lines)])
        resp, _ = _think(pe_dirty, "what is a wug?")
        r.verbatim[f"after_{distractors}_distractors"] = resp
        if "ferren" in resp.lower():
            hits_dirty += 1

    r.measurements += [
        check_rate("recall_immediate", hits_clean, replicates, 0.25,
                   min_rate=0.5, detail="baseline: teach → ask"),
        check_rate("recall_after_distractors", hits_dirty, replicates, 0.25,
                   min_rate=0.5,
                   detail=f"teach → {distractors} unrelated turns → ask"),
    ]
    return r


# ─────────────────────────────────────────────────────────────────
# 4. Persistence across restart
# ─────────────────────────────────────────────────────────────────


def probe_restart_persistence(
    backend: str = "offline", socket_path: str | None = None
) -> ProbeReport:
    """Learned knowledge survives a full restart of the cognitive layer.

    Teaches a fact on engine A, saves cognitive state, builds a fresh
    engine B on the same data dir (new Python objects — the only
    continuity is what persistence actually wrote), and asks engine B
    the recall question. A second fresh engine on an empty dir is the
    negative control.
    """
    r = _report(
        "capability.restart_persistence",
        "Learned knowledge survives process restart",
        "Knowledge acquired through dialogue persists through the "
        "canonical stores (edge log + saved network state): a fresh "
        "engine on the same data dir recalls the fact; a fresh dir "
        "does not.",
        backend,
        notes="Restart is genuine: new ConceptNetwork, new engine — "
              "only the on-disk state is shared.",
    )
    from genesis_cognitive import persistence

    # Arm A: teach + save.
    pe_a = make_engine(backend, socket_path=socket_path)
    _think(pe_a, "a wug is a ferren")
    persistence.save_state(
        pe_a.data_dir,
        network=pe_a.network,
        reflection=pe_a.engine.reflection,
        narrative=pe_a.engine.narrative,
        self_model=pe_a.engine.self_model,
    )

    # Arm B: fresh engine, same data dir; restore network state.
    pe_b = make_engine(backend, socket_path=socket_path, data_dir=pe_a.data_dir)
    saved = persistence.load_state(pe_a.data_dir)
    if saved and "concept_network" in saved:
        persistence.restore_network(pe_b.network, saved["concept_network"])
    resp_b, _ = _think(pe_b, "what is a wug?")
    r.verbatim["after_restart"] = resp_b

    # Negative control: fresh dir, never taught.
    pe_c = make_engine(backend, socket_path=socket_path)
    resp_c, _ = _think(pe_c, "what is a wug?")
    r.verbatim["fresh_control"] = resp_c

    r.measurements += [
        Measurement(
            "recalled_after_restart", "contains 'ferren'",
            "ferren" in resp_b.lower(), "ferren" in resp_b.lower(),
            resp_b),
        Measurement(
            "control_declines", "no 'ferren'",
            "ferren" not in resp_c.lower(), "ferren" not in resp_c.lower(),
            resp_c),
    ]
    return r


# ─────────────────────────────────────────────────────────────────
# 5. Calibration — knowing what it knows
# ─────────────────────────────────────────────────────────────────


def probe_calibration(backend: str = "offline", socket_path: str | None = None,
                      trials: int = 4) -> ProbeReport:
    """Confidence and behavior track actual knowledge.

    Asks answerable questions (verified relations) and unanswerable
    ones (unknown subjects/objects), then measures whether the system
    behaves differently in the two conditions: grounded answers where
    it knows, declines where it doesn't, plus the reported confidence
    distribution for honest disclosure.
    """
    r = _report(
        "capability.calibration",
        "Confidence and behavior track actual knowledge",
        "Genesis behaves differently when it knows vs when it doesn't: "
        "answerable questions get grounded verdicts, unanswerable ones "
        "get declines — and the confidence channel is reported as "
        "measured, whatever it shows.",
        backend,
    )
    pe = make_engine(backend, socket_path=socket_path)
    _think(pe, "a wug is a ferren")

    answerable = [
        "is a wug a ferren?",
        "what is a wug?",
    ]
    unanswerable = [
        "is a zav a ferren?",
        "what is a zav?",
        "is a wug a brillax?",
        "what is a brillax?",
    ]

    conf_known: list[float] = []
    conf_unknown: list[float] = []
    grounded_known = declined_unknown = 0
    n_known = n_unknown = 0

    for q in answerable:
        for _ in range(trials):
            resp, state = _think(pe, q)
            n_known += 1
            conf_known.append(float(getattr(state.thought, "confidence", 0.0)))
            if _verdict_of(state) == "yes" or "ferren" in resp.lower():
                grounded_known += 1
            r.verbatim.setdefault(f"known:{q}", resp)

    for q in unanswerable:
        for _ in range(trials):
            resp, state = _think(pe, q)
            n_unknown += 1
            conf_unknown.append(float(getattr(state.thought, "confidence", 0.0)))
            if classify_response(resp) == "decline" and _verdict_of(state) != "yes":
                declined_unknown += 1
            r.verbatim.setdefault(f"unknown:{q}", resp)

    mean_known = sum(conf_known) / len(conf_known) if conf_known else 0.0
    mean_unknown = sum(conf_unknown) / len(conf_unknown) if conf_unknown else 0.0

    r.measurements += [
        check_rate("answerable_grounded", grounded_known, n_known, 0.4,
                   min_rate=0.5,
                   detail="answerable → affirmed or names the object"),
        check_rate("unanswerable_declined", declined_unknown, n_unknown, 0.4,
                   min_rate=0.5,
                   detail="unanswerable → declines, never affirms"),
        # Reported for transparency, not gated — the confidence scalar
        # is a diagnostic, not a pass criterion.
        Measurement(
            "confidence_known_mean", "reported",
            mean_known, True, "mean thought.confidence on answerable"),
        Measurement(
            "confidence_unknown_mean", "reported",
            mean_unknown, True, "mean thought.confidence on unanswerable"),
        Measurement(
            "confidence_separates", "higher when known",
            mean_known > mean_unknown, mean_known > mean_unknown,
            f"known={mean_known:.3f} vs unknown={mean_unknown:.3f} — "
            "descriptive; a flat channel is an honest finding"),
    ]
    return r


ALL_PROBES = [
    probe_learn_recall,
    probe_verification,
    probe_retention,
    probe_restart_persistence,
    probe_calibration,
]
