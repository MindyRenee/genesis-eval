"""Evidence records for the Genesis validation suite.

Every probe returns a ``ProbeReport`` carrying typed ``Measurement``
records: the analytic expectation, the observed value, and whether the
observation matched within tolerance. Reports are what the public sees —
they record *measured* values, not just pass/fail, so a reader can check
the arithmetic themselves.

Evidence tiers (recorded on every report):

- ``implementation`` — component-level causal evidence: a mechanism,
  toggled or isolated, changes its own substrate exactly as its math
  predicts.
- ``behavioral`` — capability evidence: the whole cognitive pipeline
  produces an observable behavior change (learn, recall, generalize,
  abstain) on items it was never explicitly given.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Measurement:
    """One observed-vs-expected quantity.

    ``expected`` is the analytic prediction (closed-form or reference
    model), ``observed`` what the system produced. ``tolerance`` is the
    absolute error bound applied; ``exact`` means bit-for-bit.
    """

    name: str
    expected: float | str | bool
    observed: float | str | bool
    passed: bool
    detail: str = ""


def _jsonable(value: Any) -> Any:
    """Make a measured value JSON-serializable without losing information.

    Measurements record real Python objects — edge-fold dicts keyed by
    ``(src, dst, rel)`` tuples, sets, nested structures. Public JSON must
    preserve them readably: tuples become strings when dict keys and lists
    elsewhere; sets become sorted lists; everything else passes through.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class ProbeReport:
    """The auditable record of one probe."""

    probe_id: str
    title: str
    claim: str
    tier: str  # "substrate" | "capability"
    evidence: str  # "implementation" | "behavioral"
    backend: str = "offline"  # "offline" (in-memory client) | "live" (real daemon)
    measurements: list[Measurement] = field(default_factory=list)
    verbatim: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    @property
    def passed(self) -> bool:
        return all(m.passed for m in self.measurements) if self.measurements else False

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "title": self.title,
            "claim": self.claim,
            "tier": self.tier,
            "evidence": self.evidence,
            "backend": self.backend,
            "passed": self.passed,
            "measurements": [
                {
                    "name": m.name,
                    "expected": _jsonable(m.expected),
                    "observed": _jsonable(m.observed),
                    "passed": m.passed,
                    "detail": m.detail,
                }
                for m in self.measurements
            ],
            "verbatim": {str(k): str(v) for k, v in self.verbatim.items()},
            "notes": self.notes,
        }


def check(name: str, expected: float, observed: float, tol: float = 1e-12,
          detail: str = "") -> Measurement:
    """Numeric measurement: pass when |observed - expected| <= tol."""
    ok = math.isfinite(observed) and abs(observed - expected) <= tol
    return Measurement(name, expected, observed, ok, detail)


def check_exact(name: str, expected: Any, observed: Any, detail: str = "") -> Measurement:
    """Exact measurement (strings, booleans, bit-identical floats)."""
    return Measurement(name, expected, observed, expected == observed, detail)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% at z=1.96).

    Preferred over the normal approximation at small n — the interval
    a public reader should see next to "k out of n".
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n))
    # The Wilson interval contains p̂ by construction (it inverts the
    # score test, and p̂ trivially passes) — clamp to that invariant so
    # float undershoot at the edges can't produce hi < p or lo > p.
    return (max(0.0, min(p, (centre - half) / denom)),
            min(1.0, max(p, (centre + half) / denom)))


def binom_p_ge(k: int, n: int, p0: float) -> float:
    """Exact one-sided binomial p-value: P(X >= k) under H0: p = p0.

    Computed directly from the PMF — no scipy dependency in the report
    path, so the published numbers are reproducible anywhere.
    """
    if n <= 0:
        return 1.0
    return min(
        1.0,
        sum(math.comb(n, i) * p0**i * (1 - p0) ** (n - i) for i in range(k, n + 1)),
    )


def check_rate(
    name: str,
    k: int,
    n: int,
    p0: float,
    *,
    min_rate: float = 0.0,
    detail: str = "",
) -> Measurement:
    """Binomial-rate measurement: pass when the exact one-sided test
    rejects the chance baseline at alpha=0.05 AND the observed rate
    clears ``min_rate``. Records the rate, CI, and p in ``detail``."""
    lo, hi = wilson_ci(k, n)
    p = binom_p_ge(k, n, p0)
    rate = k / n if n else 0.0
    ok = (p < 0.05) and (rate >= min_rate)
    return Measurement(
        name, f"> {p0} (chance)", f"{k}/{n} = {rate:.3f}",
        ok, f"95% CI [{lo:.3f}, {hi:.3f}], p={p:.4g}. {detail}".strip(),
    )
