"""Tests for the validation suite itself.

These verify the *measurement machinery* — that Wilson intervals and
exact binomial p-values are right, that reports serialize cleanly
(regardless of what a probe recorded), that the offline client returns
production-typed objects, and that fast deterministic probes are
stable. They do not re-run the whole suite — that is what
``run_probes.py`` is for.

Run from the repo root:

    PYTHONPATH=python python3 -m pytest validation/test_probes.py -q
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_REPO_ROOT / "python"), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from validation.probes import substrate
from validation.probes.report import (
    Measurement,
    ProbeReport,
    binom_p_ge,
    check_rate,
    wilson_ci,
)

# ─── statistics ──────────────────────────────────────────────────


def test_wilson_ci_endpoints_and_monotonicity():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and 0.0 < lo < 1.0
    # More evidence at the same rate must tighten the interval.
    w1 = wilson_ci(5, 10)
    w2 = wilson_ci(50, 100)
    assert (w2[1] - w2[0]) < (w1[1] - w1[0])
    # Interval must contain the point estimate.
    for k, n in ((0, 5), (3, 10), (12, 12), (7, 40)):
        lo, hi = wilson_ci(k, n)
        assert lo <= k / n <= hi


def test_binom_p_ge_known_values():
    # P(X >= k | n, 0.5) has closed-form values for small n.
    assert math.isclose(binom_p_ge(4, 4, 0.5), 1 / 16)
    assert math.isclose(binom_p_ge(0, 4, 0.5), 1.0)
    assert math.isclose(binom_p_ge(8, 8, 0.5), 1 / 256)
    # Two flips of a fair coin: P(X>=1) = 3/4.
    assert math.isclose(binom_p_ge(1, 2, 0.5), 0.75)
    # Monotonic in k.
    assert binom_p_ge(9, 10, 0.5) < binom_p_ge(6, 10, 0.5)


def test_check_rate_power_floor():
    # 4/4 vs p0=0.5 cannot reach p<0.05 — the suite must not call a
    # small perfect run significant.
    m = check_rate("small_n", 4, 4, 0.5)
    assert not m.passed
    # 8/8 clears it: p = 1/256 ≈ 0.0039.
    m = check_rate("enough_n", 8, 8, 0.5)
    assert m.passed
    # A mid rate must fail even with large n.
    m = check_rate("half", 50, 100, 0.9)
    assert not m.passed


# ─── report serialization ────────────────────────────────────────


def test_report_serializes_real_python_objects():
    """Substrate probes record dicts keyed by tuples, sets, raw values
    — the JSON report must not choke on them."""
    rep = ProbeReport(
        probe_id="test.serialize",
        title="serialization",
        claim="x",
        tier="substrate",
        evidence="implementation",
        measurements=[
            Measurement("edge_fold", "x", {("a", "b", "is_a"): 0.9}, True),
            Measurement("a_set", "x", {"b", "a"}, True),
            Measurement("a_tuple", "x", (1, 2), True),
        ],
        verbatim={("k",): "v"},
    )
    text = json.dumps(rep.to_dict())
    parsed = json.loads(text)
    assert parsed["measurements"][0]["observed"] == {"('a', 'b', 'is_a')": 0.9}
    assert parsed["measurements"][1]["observed"] == ["a", "b"]
    assert parsed["measurements"][2]["observed"] == [1, 2]
    assert rep.passed


def test_report_passed_requires_measurements():
    rep = ProbeReport(
        probe_id="x", title="t", claim="c",
        tier="substrate", evidence="implementation",
    )
    assert rep.passed is False
    rep.measurements.append(Measurement("m", 1.0, 1.0, True))
    assert rep.passed is True
    rep.measurements.append(Measurement("m2", 1.0, 2.0, False))
    assert rep.passed is False


# ─── offline client typing ───────────────────────────────────────


def test_offline_client_returns_production_types():
    """The stand-in must return the same types the daemon client would
    — the engine calls ``.episode_id`` etc. on them."""
    from genesis_client.types import Episode, SimilarEpisode
    from validation.offline_client import OfflineProbeClient

    client = OfflineProbeClient()
    client.connect()
    eid = client.store_episode(
        timestamp=1_700_000_000_000,
        event_type=1,
        source_module=1,
        salience=0.5,
        emotional_tag=[0.5, 0.5, 0.5],
        text="test turn",
    )
    ep = client.retrieve_episode(eid)
    assert isinstance(ep, Episode)
    recent = client.get_recent_episodes(5)
    assert all(hasattr(e, "episode_id") for e in recent)
    sims = client.find_similar("test", 5)
    assert all(isinstance(s, SimilarEpisode) for s in sims)
    summary = client.get_inference_summary()
    assert hasattr(summary, "free_energy")
    # Audit trail exists.
    assert "store_episode" in client.calls
    assert "find_similar" in client.calls


# ─── substrate probe stability ───────────────────────────────────
#
# Substrate probes must be deterministic: same code, same inputs →
# same measured values. Run the cheap ones twice and require identical
# measurement tuples.


def test_substrate_probes_deterministic():
    for fn in (
        substrate.probe_td_fixed_point,
        substrate.probe_stdp_timing_curve,
        substrate.probe_attention_decay,
        substrate.probe_beta_conjugacy,
    ):
        r1 = fn()
        r2 = fn()
        assert r1.passed, f"{r1.probe_id} failed"
        assert [(m.name, m.observed, m.passed) for m in r1.measurements] == [
            (m.name, m.observed, m.passed) for m in r2.measurements
        ], f"{r1.probe_id} is not deterministic"


def test_every_probe_reports_required_fields():
    """Every probe result must carry the disclosure fields the public
    report depends on — tier, evidence, backend."""
    for fn in substrate.ALL_PROBES[:5]:
        rep = fn()
        assert rep.tier in ("substrate", "dynamics", "capability")
        assert rep.evidence in ("implementation", "behavioral", "causal")
        assert rep.backend in ("offline", "live")
        d = rep.to_dict()
        for key in ("probe_id", "claim", "measurements", "passed"):
            assert key in d
