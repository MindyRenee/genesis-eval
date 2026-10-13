"""Transfer-compounding eval — does Genesis's learning accelerate?

The question behind the benchmark argument: not "can it solve task
X" but "does each solved task make the next one cheaper." This eval
measures that directly through the competence substrate.

Protocol — two arms share one task plan per replicate:

- ``trained``  probes(r0) → sorter ladder → probes(r1) →
               sequence ladder → probes(r2)
- ``control``  probes(r0) → probes(r1) → probes(r2), no training

Probes are fixed-difficulty spec *templates* instantiated with fresh
seeds each round; the same spec objects feed both arms, so a cost
difference between arms is substrate, not task luck. Probes in the
trained families measure near transfer; probes in families never
trained measure far transfer. Every ``recognize`` call is logged, so
"did a skill from another domain get retrieved?" is telemetry, not
inference.

The control arm is *not* cold: every solved probe consolidates a
skill, so by round 1 it carries the priors its own probes compiled.
The difference-in-differences therefore measures the marginal value
of *more* same-domain experience — dose 1 vs dose ~16 — while the
first-contact table measures the step from zero skills to one.

Usage:
    PYTHONPATH=python PYTHONHASHSEED=0 \
        python3 eval_transfer/run_eval.py --replicates 5

Writes ``eval_transfer/results.tsv`` (one row per attempt) and
``eval_transfer/telemetry/<stamp>.jsonl``; prints the analysis.

Honest caveats baked into the numbers:
- Agent seeds derive from ``hash((name, attempts))`` — reproducible
  only under a fixed PYTHONHASHSEED (the sweep script pins it).
- Salience is unmodulated here (getter defaults to 1.0): every
  verified episode consolidates. That is the *most favorable*
  condition for transfer — it measures the substrate's capacity,
  not the daemon's neuromodulatory gating.
- "Same difficulty" across rounds means same generator parameters,
  different seeds. Instance hardness still varies — that is what
  the control arm and replicates are for.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_genesis_dir = os.environ.get(
    "GENESIS_DIR",
    str(Path(__file__).resolve().parent.parent.parent / "genesis"),
)
sys.path.insert(0, str(Path(_genesis_dir) / "python"))

from genesis_cognitive.concepts import ConceptNetwork
from genesis_cognitive.reasoning.competence import TaskCompetence
from genesis_cognitive.spatial.practice import (
    SpatialPractice,
    normalize_offered,
)
from genesis_cognitive.spatial.solver import SpatialReasoner

# ── Task spec construction ──────────────────────────────────────
# Everything below emits *raw* offered-puzzle specs and normalizes
# them through the real intake path (normalize_offered) — the same
# validation a dropped JSON file would face.


def _sorter(rng: random.Random, name: str, difficulty: int,
            slots: int, decoys: int, lid: bool) -> dict:
    return {
        "name": name,
        "family": "sorter",
        "max_steps": 400,
        "generate": {
            "slots": slots,
            "difficulty": difficulty,
            "decoys": decoys,
            "lid": lid,
            "seed": rng.randrange(2**30),
        },
    }


def _sequence(rng: random.Random, name: str, period: int,
              length: int, decoys: int) -> dict:
    return {
        "name": name,
        "family": "sequence",
        "max_steps": 400,
        "generate": {
            "period": period,
            "length": length,
            "decoys": decoys,
            "seed": rng.randrange(2**30),
        },
    }


def _relations(rng: random.Random, name: str) -> dict:
    return {
        "name": name,
        "family": "relations",
        "max_steps": 600,
        "generate": {"objects": 6, "goals": 6, "seed": rng.randrange(2**30)},
    }


def _quantities(rng: random.Random, name: str) -> dict:
    return {
        "name": name,
        "family": "quantities",
        "max_steps": 400,
        "generate": {"groups": 7, "target": 18, "seed": rng.randrange(2**30)},
    }


def _assembly(rng: random.Random, name: str) -> dict:
    return {
        "name": name,
        "family": "assembly",
        "max_steps": 700,
        "generate": {"rows": 4, "cols": 4, "seed": rng.randrange(2**30)},
    }


def _classification(rng: random.Random, name: str) -> dict:
    """An induce-mode worksheet: hidden at_least(2 of 3 attrs) rule,
    12 items, 5 labeled examples. Regenerated until both labels appear
    in items and examples."""
    vocab = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta",
             "eta", "theta")

    from genesis_cognitive.classification import eval_predicate

    for _ in range(300):
        targets = rng.sample(vocab, 3)
        predicate = {"op": "at_least", "of": targets, "count": 2}
        seen: set[frozenset[str]] = set()
        items: list[dict[str, Any]] = []
        while len(items) < 12:
            attrs = frozenset(a for a in vocab if rng.random() < 0.4)
            if not attrs or attrs in seen:
                continue
            seen.add(attrs)
            items.append({
                "name": f"item_{len(items)}",
                "has": sorted(attrs),
                "label": bool(eval_predicate(predicate, attrs)),
            })
        positives = [i for i in items if i["label"]]
        negatives = [i for i in items if not i["label"]]
        if len(positives) < 3 or len(negatives) < 3:
            continue
        examples = [
            {"name": f"ex_{j}", "has": i["has"], "label": i["label"]}
            for j, i in enumerate(
                rng.sample(positives, 2) + rng.sample(negatives, 3)
            )
        ]
        return {
            "name": name,
            "family": "classification",
            "max_steps": 400,
            "kind": "induce-at_least-2of3",
            "items": items,
            "examples": examples,
        }
    raise RuntimeError("could not build a balanced classification spec")


def probe_battery(rng: random.Random, tag: str) -> list[dict]:
    """Two attempts-level probes per family at reference difficulty.

    Difficulty is set high enough that unconstrained exploration
    wastes moves — near-floor tasks leave no room for transferred
    priors to show. Sorter and sequence are the trained families
    (near transfer); the rest are never trained (far transfer)."""
    return [
        _sorter(rng, f"probe_sorter_{tag}_a", 3, 6, 4, True),
        _sorter(rng, f"probe_sorter_{tag}_b", 3, 6, 4, True),
        _sequence(rng, f"probe_sequence_{tag}_a", 4, 12, 3),
        _sequence(rng, f"probe_sequence_{tag}_b", 4, 12, 3),
        _relations(rng, f"probe_relations_{tag}_a"),
        _relations(rng, f"probe_relations_{tag}_b"),
        _quantities(rng, f"probe_quantities_{tag}_a"),
        _quantities(rng, f"probe_quantities_{tag}_b"),
        _assembly(rng, f"probe_assembly_{tag}_a"),
        _assembly(rng, f"probe_assembly_{tag}_b"),
        _classification(rng, f"probe_classification_{tag}_a"),
        _classification(rng, f"probe_classification_{tag}_b"),
    ]


def sorter_ladder(rng: random.Random) -> list[dict]:
    """Ascending constraint arity — the match-everything lesson."""
    return [
        _sorter(rng, "train_sorter_d1a", 1, 4, 0, True),
        _sorter(rng, "train_sorter_d1b", 1, 4, 0, True),
        _sorter(rng, "train_sorter_d2a", 2, 5, 2, True),
        _sorter(rng, "train_sorter_d2b", 2, 5, 2, True),
        _sorter(rng, "train_sorter_d3a", 3, 5, 3, True),
        _sorter(rng, "train_sorter_d3b", 3, 5, 3, True),
        _sorter(rng, "train_sorter_d3c", 3, 6, 4, True),
        _sorter(rng, "train_sorter_d3d", 3, 6, 4, True),
    ]


def sequence_ladder(rng: random.Random) -> list[dict]:
    return [
        _sequence(rng, "train_sequence_p2a", 2, 8, 0),
        _sequence(rng, "train_sequence_p2b", 2, 8, 0),
        _sequence(rng, "train_sequence_p3a", 3, 9, 1),
        _sequence(rng, "train_sequence_p3b", 3, 9, 1),
        _sequence(rng, "train_sequence_p4a", 4, 12, 2),
        _sequence(rng, "train_sequence_p4b", 4, 12, 2),
        _sequence(rng, "train_sequence_p4c", 4, 12, 3),
        _sequence(rng, "train_sequence_p4d", 4, 12, 3),
    ]


# ── Instrumentation ─────────────────────────────────────────────


class TelemetryCompetence(TaskCompetence):
    """A TaskCompetence that logs every retrieval.

    Each entry records the task's domain, whether the schema was
    novel, and which previously-consolidated skills were retrieved —
    split into same-domain (near transfer) and cross-domain (far
    transfer) matches with their similarity scores.
    """

    def __init__(self) -> None:
        super().__init__()
        self.retrieval_log: list[dict[str, Any]] = []

    def recognize(self, **kwargs: Any) -> Any:
        context = super().recognize(**kwargs)
        domain = kwargs.get("domain", "")
        same: list[dict[str, Any]] = []
        cross: list[dict[str, Any]] = []
        for match in context.skills:
            entry = {
                "skill_domain": match.skill.signature.domain,
                "similarity": round(match.similarity, 3),
                "score": round(match.score, 3),
            }
            (same if entry["skill_domain"] == domain else cross).append(entry)
        self.retrieval_log.append({
            "domain": domain,
            "novel_schema": context.novel,
            "schema_similarity": round(context.similarity, 3),
            "same_domain_skills": same,
            "cross_domain_skills": cross,
        })
        return context


# ── Runner ──────────────────────────────────────────────────────

TRAIN_CAP = 6  # attempts per training task before moving on


def _run_task(
    practice: SpatialPractice,
    reasoner: SpatialReasoner,
    competence: TelemetryCompetence,
    raw_spec: dict,
    *,
    replicate: int,
    arm: str,
    phase: str,
    cap: int,
) -> list[dict]:
    """Attempt one task up to ``cap`` times (once for probes).

    Training tasks retry until solved or the cap; probe tasks get a
    single shot — first-encounter cost is the measurement."""
    name = str(raw_spec["name"])
    spec = normalize_offered(raw_spec, name)
    if spec is None:
        raise ValueError(f"spec rejected by normalize_offered: {name}")
    rows = []
    for attempt_idx in range(cap):
        skills_before = len(competence.skills)
        retrieval_before = len(competence.retrieval_log)
        result = practice.attempt(reasoner, task=spec)
        if result is None:
            break
        retrievals = competence.retrieval_log[retrieval_before:]
        retrieval = retrievals[-1] if retrievals else {}
        rows.append({
            "replicate": replicate,
            "arm": arm,
            "phase": phase,
            "task": name,
            "family": spec["family"],
            "attempt": attempt_idx,
            "steps": result.nodes,
            "score": result.score,
            "solved": result.solved,
            "rule": result.rule,
            "failure": result.failure,
            "schemas": competence.schema_count,
            "skills": len(competence.skills),
            "skills_new": len(competence.skills) - skills_before,
            "novel_schema": retrieval.get("novel_schema"),
            "same_domain_retrievals": len(
                retrieval.get("same_domain_skills", [])
            ),
            "cross_domain_retrievals": len(
                retrieval.get("cross_domain_skills", [])
            ),
            "cross_domain_best": max(
                (m["similarity"]
                 for m in retrieval.get("cross_domain_skills", [])),
                default=None,
            ),
        })
        if result.solved or cap == 1:
            break
    return rows


def run_arm(replicate: int, arm: str, rng: random.Random) -> list[dict]:
    """Run one arm to completion on a fresh substrate.

    Probe batteries are drawn up front so both arms face identical
    specs — seed-equivalent instances, different minds."""
    probes = [probe_battery(rng, f"r{i}") for i in range(3)]
    trains = {
        "sorter": sorter_ladder(rng),
        "sequence": sequence_ladder(rng),
    }
    with tempfile.TemporaryDirectory(
        prefix=f"transfer_{arm}_{replicate}_"
    ) as data_dir:
        competence = TelemetryCompetence()
        reasoner = SpatialReasoner(
            ConceptNetwork(), task_competence=competence
        )
        practice = SpatialPractice(data_dir)
        rows: list[dict] = []

        def do_phase(phase: str, specs: list[dict], cap: int) -> None:
            for spec in specs:
                rows.extend(
                    _run_task(
                        practice, reasoner, competence, spec,
                        replicate=replicate, arm=arm, phase=phase,
                        cap=cap,
                    )
                )

        do_phase("probe_r0", probes[0], cap=1)
        if arm == "trained":
            do_phase("train_sorter", trains["sorter"], cap=TRAIN_CAP)
        do_phase("probe_r1", probes[1], cap=1)
        if arm == "trained":
            do_phase("train_sequence", trains["sequence"], cap=TRAIN_CAP)
        do_phase("probe_r2", probes[2], cap=1)
    return rows


# ── Analysis ────────────────────────────────────────────────────


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def _sem(xs: list[float]) -> float:
    return statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0


def _summarize(rows: list[dict]) -> list[dict]:
    """Per (arm, phase, family): steps, solve rate, retrievals."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault(
            (row["arm"], row["phase"], row["family"]), []
        ).append(row)
    out = []
    for (arm, phase, family), g in sorted(groups.items()):
        steps = [r["steps"] for r in g]
        solved = [r["solved"] for r in g]
        cross = sum(r["cross_domain_retrievals"] for r in g)
        out.append({
            "arm": arm,
            "phase": phase,
            "family": family,
            "n": len(g),
            "mean_steps": _mean(steps),
            "sem_steps": _sem(steps),
            "solve_rate": _mean([float(s) for s in solved]),
            "mean_score": _mean([r["score"] for r in g]),
            "cross_retrievals": cross,
        })
    return out


def _probe_deltas(rows: list[dict]) -> dict[str, Any]:
    """Cold-vs-warm probe comparison, per family per arm.

    Paired across replicates: for each family, the per-replicate mean
    probe steps in round k vs round 0. Positive delta = got cheaper.
    """
    probes = [
        r for r in rows
        if r["phase"].startswith("probe_") and r["attempt"] == 0
    ]
    cells: dict[tuple, list[tuple[float, float]]] = {}
    for r in probes:
        key = (r["arm"], r["phase"], r["family"], r["replicate"])
        cells.setdefault(key, []).append((r["steps"], float(r["solved"])))
    means = {k: _mean([s for s, _ in v]) for k, v in cells.items()}
    solves = {k: _mean([sv for _, sv in v]) for k, v in cells.items()}
    families = sorted({r["family"] for r in probes})
    report: dict[str, Any] = {}
    for family in families:
        for arm in ("trained", "control"):
            for rnd in ("probe_r1", "probe_r2"):
                pairs: list[tuple[float, float]] = []
                solve_pairs: list[tuple[float, float]] = []
                for rep in {r["replicate"] for r in probes}:
                    cold = means.get((arm, "probe_r0", family, rep))
                    warm = means.get((arm, rnd, family, rep))
                    if cold is not None and warm is not None:
                        pairs.append((cold, warm))
                    c_solve = solves.get((arm, "probe_r0", family, rep))
                    w_solve = solves.get((arm, rnd, family, rep))
                    if c_solve is not None and w_solve is not None:
                        solve_pairs.append((c_solve, w_solve))
                if not pairs:
                    continue
                deltas = [a - b for a, b in pairs]
                report[f"{arm}/{family}/{rnd}"] = {
                    "n": len(deltas),
                    "mean_cold_steps": _mean([a for a, _ in pairs]),
                    "mean_warm_steps": _mean([b for _, b in pairs]),
                    "mean_step_delta": _mean(deltas),
                    "sem_delta": _sem(deltas),
                    "cold_solve": _mean([a for a, _ in solve_pairs]),
                    "warm_solve": _mean([b for _, b in solve_pairs]),
                    "replicates_improved": sum(1 for d in deltas if d > 0),
                }
    # Difference-in-differences: the trained arm's warm gain minus the
    # control arm's — the causal estimate of transferred learning,
    # with seed-difficulty drift differenced out.
    for family in families:
        for rnd in ("probe_r1", "probe_r2"):
            per_rep: dict[int, dict[str, float]] = {}
            for arm in ("trained", "control"):
                for rep in {r["replicate"] for r in probes}:
                    cold = means.get((arm, "probe_r0", family, rep))
                    warm = means.get((arm, rnd, family, rep))
                    if cold is not None and warm is not None:
                        per_rep.setdefault(rep, {})[arm] = cold - warm
            did = [
                d["trained"] - d["control"]
                for d in per_rep.values()
                if "trained" in d and "control" in d
            ]
            solve_did: list[float] = []
            for rep in per_rep:
                t0 = solves.get(("trained", "probe_r0", family, rep))
                t1 = solves.get(("trained", rnd, family, rep))
                c0 = solves.get(("control", "probe_r0", family, rep))
                c1 = solves.get(("control", rnd, family, rep))
                if (
                    t0 is not None and t1 is not None
                    and c0 is not None and c1 is not None
                ):
                    solve_did.append((t1 - t0) - (c1 - c0))
            if did:
                report[f"DiD/{family}/{rnd}"] = {
                    "n": len(did),
                    "mean_step_delta": _mean(did),
                    "sem_delta": _sem(did),
                    "mean_solve_delta": _mean(solve_did),
                    "replicates_improved": sum(1 for d in did if d > 0),
                }
    return report


def _print_summary(rows: list[dict]) -> str:
    lines = [
        "",
        "=== Per-phase summary (steps = agent actions to terminal) ===",
        f"{'arm':<9}{'phase':<16}{'family':<15}{'n':>4}"
        f"{'mean_steps':>11}{'±sem':>7}{'solve%':>8}{'xdom':>6}",
    ]
    for s in _summarize(rows):
        lines.append(
            f"{s['arm']:<9}{s['phase']:<16}{s['family']:<15}{s['n']:>4}"
            f"{s['mean_steps']:>11.1f}{s['sem_steps']:>7.1f}"
            f"{s['solve_rate'] * 100:>7.0f}%{s['cross_retrievals']:>6}"
        )
    deltas = _probe_deltas(rows)
    lines += [
        "",
        "=== Probe transfer deltas (cold steps − warm steps, paired "
        "across replicates) ===",
        f"{'arm/family/round':<40}{'cold':>7}{'warm':>7}{'Δ':>8}"
        f"{'±sem':>7}{'solve Δ':>9}{'improved':>9}",
    ]
    for key, d in deltas.items():
        if key.startswith("DiD/"):
            continue
        lines.append(
            f"{key:<40}{d['mean_cold_steps']:>7.1f}"
            f"{d['mean_warm_steps']:>7.1f}{d['mean_step_delta']:>8.1f}"
            f"{d['sem_delta']:>7.1f}"
            f"{d['warm_solve'] - d['cold_solve']:>8.0%}"
            f"  {d['replicates_improved']}/{d['n']}"
        )
    lines += [
        "",
        "=== Difference-in-differences: trained Δ − control Δ ===",
        "Positive = the trained substrate gained more than seed luck "
        "explains.",
        f"{'family/round':<40}{'Δ steps':>9}{'±sem':>7}"
        f"{'Δ solve':>9}{'improved':>9}",
    ]
    for key, d in deltas.items():
        if not key.startswith("DiD/"):
            continue
        lines.append(
            f"{key[4:]:<40}{d['mean_step_delta']:>9.1f}"
            f"{d['sem_delta']:>7.1f}"
            f"{d['mean_solve_delta']:>8.0%}"
            f"  {d['replicates_improved']}/{d['n']}"
        )
    lines += [
        "",
        "=== First-contact cost vs. skill-backed cost ===",
        "Cold = novel schema, nothing to retrieve. Warm = at least one "
        "same-domain skill on offer. Within each arm, all reps pooled.",
        f"{'arm/family':<34}{'cold n':>7}{'cold steps':>11}"
        f"{'warm n':>7}{'warm steps':>11}{'Δ':>8}",
    ]
    first: dict[tuple[str, str], dict[str, list[float]]] = {}
    for r in rows:
        cell = first.setdefault(
            (r["arm"], r["family"]), {"cold": [], "warm": []}
        )
        if r["same_domain_retrievals"]:
            cell["warm"].append(r["steps"])
        elif r["novel_schema"]:
            cell["cold"].append(r["steps"])
    for (arm, family), cell in sorted(first.items()):
        cold, warm = cell["cold"], cell["warm"]
        lines.append(
            f"{arm}/{family:<28}{len(cold):>7}{_mean(cold):>11.1f}"
            f"{len(warm):>7}{_mean(warm):>11.1f}"
            f"{_mean(cold) - _mean(warm):>8.1f}"
        )
    cross = [
        r for r in rows if r["cross_domain_retrievals"]
    ]
    lines += [
        "",
        f"Cross-domain skill retrievals fired in {len(cross)} of "
        f"{len(rows)} attempts.",
    ]
    for r in cross[:12]:
        lines.append(
            f"  r{r['replicate']} {r['arm']:<8} {r['task']:<28}"
            f"best cross-domain similarity "
            f"{r['cross_domain_best']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent),
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    telemetry = outdir / "telemetry"
    telemetry.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl = telemetry / f"attempts-{stamp}.jsonl"

    rows: list[dict] = []
    for rep in range(args.replicates):
        # One spec plan per replicate; both arms replay the same rng
        # stream so probes (and unused training draws) are identical —
        # an arm difference is substrate, not task luck.
        plan_state = random.Random(args.seed_base + rep).getstate()
        for arm in ("trained", "control"):
            arm_rng = random.Random()
            arm_rng.setstate(plan_state)
            rows.extend(run_arm(rep, arm, arm_rng))

    with jsonl.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    with (outdir / "results.tsv").open("w") as fh:
        cols = [
            "replicate", "arm", "phase", "task", "family", "attempt",
            "steps", "score", "solved", "schemas", "skills",
            "skills_new", "cross_domain_retrievals", "cross_domain_best",
        ]
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write(
                "\t".join("" if row.get(c) is None else str(row.get(c))
                          for c in cols)
                + "\n"
            )
    summary = _print_summary(rows)
    print(summary)
    (outdir / "summary.txt").write_text(summary + "\n")
    print(f"\nrows: {len(rows)}  telemetry: {jsonl}")


if __name__ == "__main__":
    main()
