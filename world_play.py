"""World play — interactive worlds Genesis chooses to inhabit.

This is the world analogue of practice.py: an *engagement* it owns.
A world (e.g. an ARC-3 environment) is installed — a spec lands in
its drop-box, or a host attaches one directly — and when its volition
engine raises the ``play`` urge, it plays a session: a bounded burst
of steps inside the world.

Protocol — no teaching, no correction:

  1. A world exposes the same minimal protocol SpatialAgent drives:
     ``actions``, ``frame()``, ``step(name, data, reasoning)``,
     ``reset()``, ``state``, ``levels_completed``, ``win_levels``,
     ``score``, ``is_complex(name)``, ``frame_shape``. The adapter
     is pure I/O — action names stay neutral, frames pass through
     unmodified, and the only progress signals are the world's own.
  2. The ``SpatialAgent`` is the skill substrate — it perceives,
     chooses, and learns affordances. The Mind owns everything else:
     the decision to engage, the felt consequences of each episode
     (through its emotional regulator), and the memories that persist.
  3. Progress persists in ``<data_dir>/world_play.json`` so sessions
     compound across restarts: episodes, deaths, wins, best level
     reached per world.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from genesis_cognitive.cognitive_journal import record_error
from genesis_cognitive.spatial.agent import SpatialAgent, frame_to_grid

_PROGRESS_FILE = "world_play.json"
logger = logging.getLogger(__name__)

# Terminal states where the only legal action is RESET.
_TERMINAL_STATES = frozenset({"NOT_PLAYED", "GAME_OVER"})


@dataclass
class EpisodeOutcome:
    """One life inside a world — ended by WIN, GAME_OVER, or burst end."""

    state: str
    levels_completed: int
    win_levels: int
    score: int


@dataclass
class WorldReport:
    """What a session did — steps, episode boundaries, win flag."""

    world_id: str
    steps: int = 0
    resets: int = 0
    episodes: list[EpisodeOutcome] = field(default_factory=list)
    won: bool = False
    levels_completed: int = 0
    win_levels: int = 0
    final_score: int = 0
    final_state: str = "UNKNOWN"

    @property
    def progress(self) -> float:
        """Fraction of the world's levels cleared, 0.0 → 1.0."""
        if self.win_levels <= 0:
            return 0.0
        return min(1.0, self.levels_completed / self.win_levels)


@dataclass
class WorldPlay:
    """Genesis's persistent world-engagement state.

    Lives inside its Mind: owns installed worlds and their play
    history, drives sessions through a SpatialAgent skill substrate
    when its volition raises the play urge.
    """

    data_dir: str
    _progress_path: Path = field(init=False)
    # world_id -> cumulative stats (persisted)
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    # world_id -> live (env, agent) pair — the substrate persists
    # across sessions in-process; task_competence carries what it
    # learned across restarts.
    _worlds: dict[str, tuple[Any, SpatialAgent]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        self._progress_path = Path(self.data_dir) / _PROGRESS_FILE
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._progress_path.read_text())
            self.stats = {
                str(k): dict(v)
                for k, v in data.get("worlds", {}).items()
            }
        except (OSError, json.JSONDecodeError, ValueError):
            self.stats = {}

    def _save(self) -> None:
        tmp = self._progress_path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps({"worlds": self.stats}, indent=2)
            )
            tmp.replace(self._progress_path)
        except OSError as e:
            # Same discipline as spatial_practice: a failed progress
            # save must leave a trace, not vanish silently.
            record_error("world_play.save", e)

    def attach(self, world: Any, world_id: str, seed: int = 0) -> None:
        """Install a world — it appears in Genesis's environment."""
        self._worlds[world_id] = (world, SpatialAgent(seed=seed))
        self.stats.setdefault(
            world_id,
            {
                "sessions": 0,
                "steps": 0,
                "episodes": 0,
                "deaths": 0,
                "wins": 0,
                "best_progress": 0.0,
            },
        )
        self._save()

    def detach(self, world_id: str) -> None:
        self._worlds.pop(world_id, None)

    def installed(self) -> list[str]:
        """Worlds currently installed."""
        return sorted(self._worlds)

    def has_pending(self) -> bool:
        """A world is installed and not yet won."""
        return any(
            not self.stats.get(wid, {}).get("wins")
            for wid in self._worlds
        )

    def best_progress(self, world_id: str) -> float:
        return float(self.stats.get(world_id, {}).get("best_progress", 0.0))

    def play_burst(
        self,
        world_id: str,
        max_steps: int = 80,
        on_episode: Callable[[EpisodeOutcome], None] | None = None,
        on_step: Callable[[dict], None] | None = None,
    ) -> WorldReport | None:
        """Play one session: up to *max_steps* actions in the world.

        Episode boundaries (WIN/GAME_OVER) close the skill substrate's
        episode and are reported via *on_episode* so the Mind can feel
        and remember each one. Resets are a session concern, owned
        here — the substrate only ever sees playable states.
        """
        pair = self._worlds.get(world_id)
        if pair is None:
            return None
        env, agent = pair
        report = WorldReport(world_id=world_id)
        t0 = time.monotonic()

        obs = env.reset()
        report.resets += 1
        grid = (
            frame_to_grid(env.frame()) if env.frame() is not None else None
        )

        for _ in range(max_steps):
            state = env.state
            if state in _TERMINAL_STATES:
                outcome = EpisodeOutcome(
                    state=state,
                    levels_completed=env.levels_completed,
                    win_levels=env.win_levels,
                    score=env.score,
                )
                report.episodes.append(outcome)
                if on_episode is not None:
                    on_episode(outcome)
                agent.on_episode_end(state)
                env.reset()
                report.resets += 1
                if env.frame() is not None:
                    grid = frame_to_grid(env.frame())
                continue

            actions = env.actions
            if not actions:
                break  # nothing legal and not terminal — stuck

            if grid is not None:
                agent.observe(grid)

            action = agent.choose_action(actions)
            data = (
                agent.action_data(action, env.frame_shape)
                if env.is_complex(action)
                else None
            )

            prev_levels = env.levels_completed
            reasoning = {
                "agent": "genesis-mind",
                "avatar_color": agent.avatar_color,
            }
            obs = env.step(action, data=data, reasoning=reasoning)
            if obs is None:
                break

            grid = (
                frame_to_grid(env.frame())
                if env.frame() is not None
                else None
            )
            report.steps += 1
            if on_step is not None:
                on_step(
                    {
                        "step": report.steps,
                        "action": action,
                        "data": data,
                        "state": env.state,
                        "score": env.score,
                        "levels_completed": env.levels_completed,
                        "avatar_color": agent.avatar_color,
                        "elapsed_ms": int(
                            (time.monotonic() - t0) * 1000
                        ),
                    }
                )

            if env.levels_completed > prev_levels:
                agent.mark_goal_reached()

            if env.state == "WIN":
                outcome = EpisodeOutcome(
                    state="WIN",
                    levels_completed=env.levels_completed,
                    win_levels=env.win_levels,
                    score=env.score,
                )
                report.episodes.append(outcome)
                if on_episode is not None:
                    on_episode(outcome)
                agent.on_episode_end("WIN")
                report.won = True
                break

        report.levels_completed = env.levels_completed
        report.win_levels = env.win_levels
        report.final_score = env.score
        report.final_state = env.state

        # Persist the world's running history — sessions compound.
        st = self.stats.setdefault(world_id, {})
        st["sessions"] = int(st.get("sessions", 0)) + 1
        st["steps"] = int(st.get("steps", 0)) + report.steps
        st["episodes"] = (
            int(st.get("episodes", 0)) + len(report.episodes)
        )
        st["deaths"] = int(st.get("deaths", 0)) + sum(
            1 for e in report.episodes if e.state == "GAME_OVER"
        )
        if report.won:
            st["wins"] = int(st.get("wins", 0)) + 1
        st["best_progress"] = max(
            float(st.get("best_progress", 0.0)), report.progress
        )
        self._save()
        return report
