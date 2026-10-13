"""Run the Genesis validation suite and write an auditable report.

This suite is an EXTERNAL experiment — it lives outside the genesis
repo (in genesis-eval/) so nothing here ships with her code. Probe
*findings* drive deliberate refactors in her code; the probes
themselves are disposable measurement rigs.

Usage (from anywhere — PYTHONPATH needs her python/ + this dir's parent):

    PYTHONPATH=/path/to/genesis/python:/path/to/genesis-eval \
        python3 -m validation.run_probes
    PYTHONPATH=... python3 -m validation.run_probes --tier breakthrough
    PYTHONPATH=... python3 -m validation.run_probes --backend both \
        --socket /path/to/scratch-daemon.sock

Tiers:
    substrate   — mechanism-level exact-math probes (implementation evidence)
    capability  — end-to-end behavioral probes (behavioral evidence)
    dynamics    — in-vivo ablation (mechanism on/off, same input, diff state)
    breakthrough— behaviors a lookup table cannot produce (intelligence audit)

Backends:
    offline — declared in-memory client stand-in (default; reproducible)
    live    — real GenesisClient over --socket or $GENESIS_PROBE_SOCKET.
              Point it at a DEDICATED SCRATCH DAEMON (fresh data dir):
              teaching probes write real episodes into whatever daemon
              the socket belongs to. Never run --live against a live
              Genesis.
    both    — run the suite on each backend; every report is tagged.

Output:
    validation/results/probes-<timestamp>-<backend>.json — one
    ProbeReport per probe: claim, measured values, exact statistics,
    verbatim responses. Failed probes are recorded, not hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT))  # so `validation.probes` resolves as a package

from validation.probes.report import ProbeReport


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _run_probe(fn, backend: str, socket_path: str | None) -> ProbeReport:
    """Run one probe; catch crashes so a failure is recorded, not lost."""
    try:
        if "backend" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
            report = fn(backend=backend, socket_path=socket_path)
        else:
            report = fn()
        return report
    except Exception as exc:  # noqa: BLE001 — a crashed probe is a result too
        return ProbeReport(
            probe_id=getattr(fn, "__name__", "unknown"),
            title=f"(crashed) {fn.__name__}",
            claim="probe raised an exception — recorded as failure",
            tier="capability",
            evidence="behavioral",
            backend=backend,
            notes=f"{type(exc).__name__}: {exc}",
        )


def run_tier(tier: str, backend: str, socket_path: str | None) -> list[ProbeReport]:
    reports: list[ProbeReport] = []
    if tier in ("substrate", "all"):
        from validation.probes import substrate

        for fn in substrate.ALL_PROBES:
            reports.append(_run_probe(fn, backend, socket_path))
    if tier in ("dynamics", "all"):
        from validation.probes import dynamics

        for fn in dynamics.ALL_PROBES:
            reports.append(_run_probe(fn, backend, socket_path))
    if tier in ("capability", "all"):
        from validation.probes import capability

        for fn in capability.ALL_PROBES:
            reports.append(_run_probe(fn, backend, socket_path))
    if tier in ("breakthrough", "all"):
        from validation.probes import breakthrough

        for fn in breakthrough.ALL_PROBES:
            reports.append(_run_probe(fn, backend, socket_path))
    return reports


def main() -> int:
    ap = argparse.ArgumentParser(description="Genesis validation suite")
    ap.add_argument("--tier", choices=["substrate", "dynamics", "capability",
                                       "breakthrough", "all"],
                    default="all")
    ap.add_argument("--backend", choices=["offline", "live", "both"],
                    default="offline")
    ap.add_argument("--socket", default=os.environ.get("GENESIS_PROBE_SOCKET"),
                    help="daemon socket for --backend live (scratch daemon only)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    backends = ["offline", "live"] if args.backend == "both" else [args.backend]
    if "live" in backends and not args.socket:
        print(
            "ERROR: --backend live requires --socket (or $GENESIS_PROBE_SOCKET) "
            "pointing at a dedicated scratch daemon — never a live Genesis.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_reports: list[ProbeReport] = []
    for backend in backends:
        print(f"── backend: {backend} " + "─" * 40)
        reports = run_tier(args.tier, backend, args.socket)
        for rep in reports:
            status = "PASS" if rep.passed else "FAIL"
            print(f"{status} {rep.probe_id} [{rep.backend}]")
            for m in rep.measurements:
                mark = "ok" if m.passed else "XX"
                print(f"     {mark} {m.name}: {m.observed}"
                      + (f" — {m.detail}" if m.detail else ""))
        all_reports.extend(reports)

        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"probes-{ts}-{backend}.json"
        out_path.write_text(json.dumps({
            "suite": "genesis-validation",
            "backend": backend,
            "git_sha": _git_sha(),
            "timestamp": ts,
            "probes": [rep.to_dict() for rep in reports],
        }, indent=2))
        print(f"  → wrote {out_path}")

    n_pass = sum(1 for rep in all_reports if rep.passed)
    n_fail = len(all_reports) - n_pass
    print(f"\n{n_pass} passed, {n_fail} failed, {len(all_reports)} total")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
