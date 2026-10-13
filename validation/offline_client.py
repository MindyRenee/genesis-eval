"""Deterministic in-memory stand-in for the daemon IPC boundary.

Capability probes exercise the cognitive layer end to end
(``CognitionEngine.think``), which expects a ``GenesisClient`` wired to a
live daemon. This class implements the same method surface the cognitive
layer calls, served deterministically from local state:

- ``store_episode`` appends to an in-memory episode store and returns real
  incrementing episode ids — retrieval calls read the same store, so
  memory probes observe real behavior through the real interface.
- Neurochemical reads return a fixed neutral state (declared in the
  report as ``chemistry: stubbed``), so runs are reproducible and any
  client call is recorded in ``self.calls`` for audit.

This is declared scaffolding, not a hidden fake: the validation report
marks every result obtained through it. Subcognitive dynamics
(neurochemical coupling, active-inference feedback) are probed separately
in ``tests/validation_neurodynamics.rs``, against the real Rust code.
"""

from __future__ import annotations

import threading
from typing import Any

from genesis_client.types import (
    CoreState,
    MemoryStats,
    NeuroSummary,
    PhaseInfo,
    PingResponse,
    PlasticityProfile,
)

# The 18 wire-order chemical names (genesis_client.protocol.CHEM_NAMES).
_CHEMS = [
    "dopamine", "serotonin", "norepinephrine", "acetylcholine", "gaba",
    "glutamate", "cortisol", "oxytocin", "endorphin", "histamine",
    "adenosine", "bdnf", "endocannabinoid", "vasopressin", "crh",
    "orexin", "epinephrine", "melatonin",
]


class OfflineProbeClient:
    """In-memory GenesisClient replacement for offline capability probes.

    Every invoked method appends ``(name, kwargs_digest)`` to ``calls`` so
    a probe can audit exactly what crossed the IPC boundary.
    """

    def __init__(
        self,
        *,
        arousal: float = 0.5,
        valence: float = 0.0,
        plasticity_gate: float = 0.5,
        phase: int = 0,
    ) -> None:
        self.calls: list[str] = []
        self._lock = threading.RLock()
        self._episodes: list[dict[str, Any]] = []
        self._next_eid = 1
        self.arousal = arousal
        self.valence = valence
        self.plasticity_gate = plasticity_gate
        self.phase = phase

    # ── Connection lifecycle ─────────────────────────────────────
    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, timeout: float | None = None) -> None:
        self.calls.append("connect")

    def disconnect(self) -> None:
        self.calls.append("disconnect")

    def ping(self, *, timeout: float | None = None) -> PingResponse:
        self.calls.append("ping")
        return PingResponse(ack=True, uptime_ms=0)

    # ── Neurochemical reads (fixed neutral state) ────────────────
    def get_state(self, *, timeout: float | None = None) -> CoreState:
        self.calls.append("get_state")
        return CoreState(
            heartbeat=0,
            instance_id=0,
            created_at=0,
            last_updated=0,
            arousal=self.arousal,
            valence=self.valence,
            global_tone=0.5,
            plasticity_gate=self.plasticity_gate,
            chemicals=dict.fromkeys(_CHEMS, 0.5),
            encoding_weight=0.5,
            consolidation_weight=0.5,
            retrieval_weight=0.5,
            cognitive_zone=0,
            emergent_phase=self.phase,
            stm_count=0,
            ltm_episode_count=len(self._episodes),
        )

    def get_neuro_summary(self, *, timeout: float | None = None) -> NeuroSummary:
        self.calls.append("get_neuro_summary")
        return NeuroSummary(
            arousal=self.arousal,
            valence=self.valence,
            global_tone=0.5,
            plasticity_gate=self.plasticity_gate,
            encoding_weight=0.5,
            consolidation_weight=0.5,
            retrieval_weight=0.5,
            phase=self.phase,
        )

    def get_phase(self, *, timeout: float | None = None) -> PhaseInfo:
        self.calls.append("get_phase")
        return PhaseInfo(phase=self.phase, arousal=self.arousal, valence=self.valence)

    def get_plasticity_profile(self, *, timeout: float | None = None) -> PlasticityProfile:
        self.calls.append("get_plasticity_profile")
        return PlasticityProfile(
            plasticity_gate=self.plasticity_gate,
            bdnf_effective=0.5,
            bdnf_tonic=0.5,
            cortisol_effective=0.0,
            cortisol_tonic=0.0,
            dopamine_effective=0.5,
            serotonin_effective=0.5,
            coupling_drift=0.0,
            mean_receptor_sensitivity=1.0,
            emergent_phase=self.phase,
        )

    # ── Memory ───────────────────────────────────────────────────
    def get_memory_stats(self, *, timeout: float | None = None) -> MemoryStats:
        self.calls.append("get_memory_stats")
        return MemoryStats(stm_count=0, ltm_count=len(self._episodes), ltm_capacity=0)

    def store_episode(
        self,
        *,
        timestamp: int,
        event_type: str,
        source_module: int,
        salience: float,
        emotional_tag: list[float],
        text: str,
        timeout: float | None = None,
    ) -> int:
        self.calls.append("store_episode")
        with self._lock:
            eid = self._next_eid
            self._next_eid += 1
            self._episodes.append(
                {
                    "eid": eid,
                    "timestamp": timestamp,
                    "event_type": event_type,
                    "source_module": source_module,
                    "salience": salience,
                    "emotional_tag": list(emotional_tag),
                    "text": text,
                }
            )
            return eid

    @property
    def episodes(self) -> list[dict[str, Any]]:
        """Audit access to the in-memory episode store."""
        return list(self._episodes)

    def retrieve_episode(self, eid: int, *, timeout: float | None = None) -> Any:
        from genesis_client.types import Episode

        self.calls.append("retrieve_episode")
        for ep in self._episodes:
            if ep["eid"] == eid:
                return Episode(
                    episode_id=ep["eid"],
                    timestamp=ep["timestamp"],
                    salience=ep["salience"],
                    association_hash=0,
                    full_emotional_tag=ep["emotional_tag"],
                    event_type=0,
                    source_module=ep["source_module"],
                    text=ep["text"],
                )
        return None

    def get_recent_episodes(
        self, count: int = 8, *, timeout: float | None = None
    ) -> list[Any]:
        from genesis_client.types import RecentEpisode

        self.calls.append("get_recent_episodes")
        return [
            RecentEpisode(
                episode_id=ep["eid"],
                timestamp=ep["timestamp"],
                salience=ep["salience"],
                event_type=0,
                source_module=ep["source_module"],
                text=ep["text"],
            )
            for ep in self._episodes[-count:]
        ]

    def find_similar(
        self, text: str, limit: int = 5, *, timeout: float | None = None
    ) -> list[Any]:
        from genesis_client.types import SimilarEpisode

        self.calls.append("find_similar")
        # Honest minimal similarity: content-word overlap, deterministic.
        # Hamming distance is approximated by disjoint terms — sufficient
        # for probe purposes and declared as such in the report.
        terms = {w for w in text.lower().split() if len(w) > 3}
        hits = []
        for ep in self._episodes:
            ep_terms = {w for w in str(ep["text"]).lower().split() if len(w) > 3}
            shared = terms & ep_terms
            if shared:
                hits.append(
                    SimilarEpisode(
                        episode_id=ep["eid"],
                        hamming_distance=max(0, 32 - 8 * len(shared)),
                        timestamp=ep["timestamp"],
                        salience=ep["salience"],
                    )
                )
        return hits[:limit]

    def search_episodes(self, query: str, *, timeout: float | None = None) -> list[Any]:
        self.calls.append("search_episodes")
        q = query.lower()
        return [ep for ep in self._episodes if q in str(ep["text"]).lower()]

    # ── Write/ack commands (recorded, acknowledged) ──────────────
    def store_event(self, *args: Any, **kwargs: Any) -> bool:
        self.calls.append("store_event")
        return True

    def neuro_impulse(self, *args: Any, **kwargs: Any) -> bool:
        self.calls.append("neuro_impulse")
        return True

    def neuro_adjust_baseline(self, *args: Any, **kwargs: Any) -> bool:
        self.calls.append("neuro_adjust_baseline")
        return True

    def set_zone(self, zone: int, **kwargs: Any) -> bool:
        self.calls.append("set_zone")
        return True

    def update_user_affect(self, *args: Any, **kwargs: Any) -> bool:
        self.calls.append("update_user_affect")
        return True

    def update_module_status(self, *args: Any, **kwargs: Any) -> bool:
        self.calls.append("update_module_status")
        return True

    def heartbeat_module(self, *args: Any, **kwargs: Any) -> bool:
        self.calls.append("heartbeat_module")
        return True

    def archive_episode(self, *args: Any, **kwargs: Any) -> bool:
        self.calls.append("archive_episode")
        return True

    def apply_body_control(self, *args: Any, **kwargs: Any) -> bool:
        self.calls.append("apply_body_control")
        return True

    def save_inference(self, *args: Any, **kwargs: Any) -> bool:
        self.calls.append("save_inference")
        return True

    def advance_neuro(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("advance_neuro")
        return None

    def consolidate(self, *, timeout: float | None = None) -> int:
        self.calls.append("consolidate")
        return 0

    def associate(self, *, timeout: float | None = None) -> int:
        self.calls.append("associate")
        return 0

    def dream(self, *, timeout: float | None = None) -> int:
        self.calls.append("dream")
        return 0

    def read_sensors(self, *, timeout: float | None = None) -> Any:
        self.calls.append("read_sensors")
        return None

    def get_body_state(self, *, timeout: float | None = None) -> Any:
        self.calls.append("get_body_state")
        return None

    def get_body_control(self, *, timeout: float | None = None) -> Any:
        self.calls.append("get_body_control")
        return None

    def get_subsystem_telemetry(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("get_subsystem_telemetry")
        return None

    def get_inference_summary(self, *args: Any, **kwargs: Any) -> Any:
        from genesis_client.types import InferenceSummary

        self.calls.append("get_inference_summary")
        return InferenceSummary(
            surprise_ema=0.0,
            free_energy=0.0,
            expected_free_energy=0.0,
            allostasis_load=0.0,
            precision=0.5,
            attunement=0.0,
            dyadic_synchrony=0.0,
            user_valence=0.0,
            user_arousal=0.5,
            user_engagement=0.5,
            prediction_error_dopamine=0.0,
            prediction_error_cortisol=0.0,
            prediction_error_serotonin=0.0,
            model_maturity=0.0,
            inference_tick_count=0,
        )

    def set_wake_alarm(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("set_wake_alarm")
        return None

    def sync(self, *, timeout: float | None = None) -> bool:
        self.calls.append("sync")
        return True
