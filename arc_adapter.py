"""ARC-AGI-3 adapter — exposes an ARC-3 environment to SpatialAgent.

Honesty by construction: the adapter injects no game knowledge.
Action names are neutral (the agent learns each action's affordance
from frame diffs, not from names like "up"), frames pass through
unmodified, and progress signals (score, levels, WIN) come only from
the environment's own reported state.

The wrapped object is duck-typed — it needs the ``EnvironmentWrapper``
surface (``action_space``, ``step``, ``reset``, ``observation_space``)
but not the ``arc_agi`` package, so tests can drive the adapter with a
mock environment and the real toolkit plugs in unchanged.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from genesis_cognitive.spatial.agent import (
    SpatialAgent,
    frame_diff,
    frame_to_grid,
)

# Terminal states where the only legal action is RESET.
_TERMINAL_STATES = frozenset({"NOT_PLAYED", "GAME_OVER"})


def _action_name(game_action: Any) -> str:
    """Neutral name for a toolkit action: 'action1', 'reset', …

    Lowercased so the agent's affordance model learns from
    consequences rather than from a hint baked into the name.
    """
    return str(getattr(game_action, "name", game_action)).lower()


def _is_complex(game_action: Any) -> bool:
    """True for coordinate-parameterized actions (ACTION6: click)."""
    is_complex = getattr(game_action, "is_complex", None)
    if callable(is_complex):
        return bool(is_complex())
    # Duck-typed fallback: ACTION6 is the only complex action in the
    # ARC-3 spec.
    return _action_name(game_action) == "action6"


class Arc3Environment:
    """Adapts an ARC-AGI-3 EnvironmentWrapper to the SpatialAgent env.

    Exposes the minimal protocol the agent drives: ``actions``
    (currently-available action names, excluding RESET — the runner
    owns resets), ``frame()`` (raw layers; ``frame_to_grid`` takes the
    last), ``step(name, data)``, ``reset()``, ``state``, ``score``.
    """

    def __init__(self, wrapper: Any) -> None:
        self._env = wrapper
        self._last_obs: Any = None

    def _obs(self) -> Any:
        obs = self._env.observation_space
        return obs if obs is not None else self._last_obs

    # ── SpatialAgent protocol ─────────────────────────────────

    @property
    def actions(self) -> list[str]:
        """Action names the game currently permits (RESET excluded).

        Prefers the per-frame ``available_actions`` — games can
        disable actions mid-play, and the response metadata is the
        authoritative live option set. Falls back to the declared
        action space for wrappers that don't report it.
        """
        space = self._env.action_space or []
        obs = self._obs()
        avail = getattr(obs, "available_actions", None)
        if avail:
            by_value = {
                int(getattr(a, "value", -1)): a for a in space
            }
            return [
                _action_name(by_value[v])
                for v in avail
                if v in by_value and _action_name(by_value[v]) != "reset"
            ]
        return [
            _action_name(a)
            for a in space
            if _action_name(a) != "reset"
        ]

    def is_complex(self, action: str) -> bool:
        """Whether *action* needs x,y data (ACTION6)."""
        for a in self._env.action_space or []:
            if _action_name(a) == action:
                return _is_complex(a)
        return False

    def frame(self) -> Any:
        """Raw frame layers; frame_to_grid takes the composed last."""
        obs = self._obs()
        return getattr(obs, "frame", None)

    def step(self, action: str, data: dict | None = None,
             reasoning: dict | None = None) -> Any:
        """Apply a named action; returns the new raw observation."""
        for a in self._env.action_space or []:
            if _action_name(a) == action:
                obs = self._env.step(a, data=data, reasoning=reasoning)
                if obs is not None:
                    self._last_obs = obs
                return obs
        return self._obs()

    def reset(self) -> Any:
        """Issue RESET; returns the new raw observation."""
        # The toolkit exposes reset() for initial start and accepts
        # the RESET action for post-GAME_OVER restarts.
        for a in self._env.action_space or []:
            if _action_name(a) == "reset":
                obs = self._env.step(a)
                if obs is not None:
                    self._last_obs = obs
                    return obs
        obs = self._env.reset()
        if obs is not None:
            self._last_obs = obs
        return obs

    # ── Telemetry accessors ───────────────────────────────────

    @property
    def state(self) -> str:
        obs = self._obs()
        state = getattr(obs, "state", None)
        return getattr(state, "name", str(state)) if state is not None else "UNKNOWN"

    @property
    def score(self) -> int:
        obs = self._obs()
        return int(getattr(obs, "score", 0) or 0)

    @property
    def levels_completed(self) -> int:
        obs = self._obs()
        return int(getattr(obs, "levels_completed", 0) or 0)

    @property
    def win_levels(self) -> int:
        obs = self._obs()
        return int(getattr(obs, "win_levels", 0) or 0)

    @property
    def frame_shape(self) -> tuple[int, int]:
        grid = frame_to_grid(self.frame())
        return (grid.height, grid.width)


@dataclass
class StepRecord:
    """One action's audit trail — what an examiner reviews."""

    step: int
    action: str
    data: dict | None
    state: str
    score: int
    levels_completed: int
    frame_change: float
    avatar_color: int | None
    elapsed_ms: int

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


@dataclass
class GameResult:
    """Outcome of one game: steps taken, levels cleared, win flag."""

    game_id: str
    steps: int = 0
    resets: int = 0
    won: bool = False
    levels_completed: int = 0
    win_levels: int = 0
    final_score: int = 0
    final_state: str = "UNKNOWN"
    records: list[StepRecord] = field(default_factory=list)


def run_game(
    env: Arc3Environment,
    agent: SpatialAgent,
    *,
    max_actions: int = 5000,
    game_id: str = "unknown",
    on_step: Callable[[StepRecord], None] | None = None,
) -> GameResult:
    """Drive the agent through one ARC-3 game.

    Loop: observe → choose → act → observe. The environment is the
    only source of progress signals — ``levels_completed`` rising is a
    sub-goal (``mark_goal_reached``), ``WIN`` ends the run
    (``on_episode_end("WIN")`` consolidates the learned control map).

    RESET is a meta-action owned by this loop, not the agent: it is
    issued whenever the game sits in NOT_PLAYED or GAME_OVER.
    """
    result = GameResult(game_id=game_id)
    t0 = time.monotonic()
    obs = env.reset()
    result.resets += 1
    grid = frame_to_grid(env.frame()) if env.frame() is not None else None

    for step in range(max_actions):
        state = env.state
        if state in _TERMINAL_STATES:
            # Level/game over — the only legal move is RESET.
            agent.on_episode_end(state)
            obs = env.reset()
            result.resets += 1
            if env.frame() is not None:
                grid = frame_to_grid(env.frame())
            continue

        actions = env.actions
        if not actions:
            break  # nothing legal to do and not terminal — stuck

        if grid is not None:
            agent.observe(grid)

        action = agent.choose_action(actions)
        data = (
            agent.action_data(action, env.frame_shape)
            if env.is_complex(action)
            else None
        )

        prev_grid = grid
        prev_levels = env.levels_completed
        reasoning = {
            "agent": "genesis-spatial",
            "avatar_color": agent.avatar_color,
        }
        obs = env.step(action, data=data, reasoning=reasoning)
        if obs is None:
            break

        new_grid = frame_to_grid(env.frame()) if env.frame() is not None else None
        change = (
            frame_diff(prev_grid, new_grid)
            if prev_grid is not None and new_grid is not None
            else 0.0
        )
        grid = new_grid

        record = StepRecord(
            step=step,
            action=action,
            data=data,
            state=env.state,
            score=env.score,
            levels_completed=env.levels_completed,
            frame_change=round(change, 4),
            avatar_color=agent.avatar_color,
            elapsed_ms=int((time.monotonic() - t0) * 1000),
        )
        result.records.append(record)
        if on_step is not None:
            on_step(record)

        if env.levels_completed > prev_levels:
            agent.mark_goal_reached()

        if env.state == "WIN":
            agent.on_episode_end("WIN")
            result.won = True
            break

    result.steps = len(result.records)
    result.won = result.won or env.state == "WIN"
    result.levels_completed = env.levels_completed
    result.win_levels = env.win_levels
    result.final_score = env.score
    result.final_state = env.state
    return result


def main(argv: list[str] | None = None) -> int:
    """Run SpatialAgent on an ARC-AGI-3 game.

    ``arc_agi`` is an optional dependency, imported here so the
    adapter and its tests stay toolkit-free.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Genesis on ARC-AGI-3")
    parser.add_argument("--game", required=True, help="game id, e.g. ls20")
    parser.add_argument("--actions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument(
        "--online",
        action="store_true",
        help="use the ARC API instead of local environment files",
    )
    parser.add_argument("--telemetry", help="JSONL path for per-step audit log")
    parser.add_argument("--render", action="store_true", help="terminal rendering")
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="disable the toolkit's official replay recording",
    )
    parser.add_argument(
        "--env-dir",
        default="environment_files",
        help="directory of local game files (default: environment_files)",
    )
    args = parser.parse_args(argv)

    try:
        import arc_agi
    except ImportError:
        print("arc-agi toolkit not installed: pip install arc-agi")
        return 2

    mode = (
        arc_agi.OperationMode.ONLINE
        if args.online
        else arc_agi.OperationMode.OFFLINE
    )
    arcade = arc_agi.Arcade(
        operation_mode=mode, environments_dir=args.env_dir
    )
    wrapper = arcade.make(
        args.game,
        seed=args.seed,
        save_recording=not args.no_record,
        render_mode="terminal" if args.render else None,
    )
    if wrapper is None:
        print(f"could not create environment for {args.game}")
        return 2

    env = Arc3Environment(wrapper)
    agent = SpatialAgent(seed=args.seed, epsilon=args.epsilon)

    log = open(args.telemetry, "a") if args.telemetry else None
    try:
        def _on_step(rec: StepRecord) -> None:
            line = rec.to_json()
            if log is not None:
                log.write(line + "\n")
                log.flush()

        result = run_game(
            env, agent,
            max_actions=args.actions, game_id=args.game, on_step=_on_step,
        )
    finally:
        if log is not None:
            log.close()

    print(
        f"{args.game}: {'WON' if result.won else 'not won'} — "
        f"{result.steps} actions, {result.levels_completed}/"
        f"{result.win_levels or '?'} levels, score {result.final_score}, "
        f"{result.resets} resets, final state {result.final_state}"
    )
    return 0 if result.won else 1


if __name__ == "__main__":
    raise SystemExit(main())
