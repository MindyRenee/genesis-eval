"""Tests for the ARC-AGI-3 adapter.

The adapter is duck-typed against the toolkit's EnvironmentWrapper,
so tests drive it with a mock env — a maze where ACTION1-4 move an
avatar and touching the goal wins. No arc_agi dependency required.
"""

from __future__ import annotations

from arc_adapter import (
    Arc3Environment,
    run_game,
)


class _FakeAction:
    """Stand-in for arcengine.GameAction."""

    def __init__(self, name: str, complex_: bool = False) -> None:
        self.name = name
        self._complex = complex_

    def is_complex(self) -> bool:
        return self._complex


class _FakeObs:
    """Stand-in for FrameDataRaw."""

    def __init__(
        self, state, score, levels_completed, win_levels, frame
    ) -> None:
        self.state = state
        self.score = score
        self.levels_completed = levels_completed
        self.win_levels = win_levels
        self.frame = frame


class _FakeState:
    """Enum-like state with a .name attribute, like GameState."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other) -> bool:
        return getattr(other, "name", other) == self.name

    def __hash__(self) -> int:
        return hash(self.name)


class _MazeEnv:
    """Minimal game: avatar (3) moves with ACTION1-4, goal (7) wins.

    Mirrors the wrapper surface: action_space (GameAction objects),
    observation_space, step(action, data, reasoning), reset().
    """

    def __init__(self, size: int = 16, goal=(1, 1), start=(5, 5)) -> None:
        self.size = size
        self.goal = goal
        self.pos = list(start)
        self.start = list(start)
        self._state = _FakeState("NOT_PLAYED")
        self._score = 0
        self._levels = 0
        self._win_levels = 1
        self.action_space = [
            _FakeAction("RESET"),
            _FakeAction("ACTION1"),   # up
            _FakeAction("ACTION2"),   # down
            _FakeAction("ACTION3"),   # left
            _FakeAction("ACTION4"),   # right
            _FakeAction("ACTION5"),   # interact (no-op here)
            _FakeAction("ACTION6", complex_=True),  # click
            _FakeAction("ACTION7"),   # undo (no-op here)
        ]
        self.observation_space = None
        self.seen_reasoning: list[dict] = []

    def _frame(self):
        g = [[0] * self.size for _ in range(self.size)]
        g[self.goal[0]][self.goal[1]] = 7
        g[self.pos[0]][self.pos[1]] = 3
        return [g]  # list-of-layers, like the real API

    def _obs(self):
        self.observation_space = _FakeObs(
            self._state, self._score, self._levels,
            self._win_levels, self._frame(),
        )
        return self.observation_space

    def reset(self):
        self.pos = list(self.start)
        self._state = _FakeState("PLAYING")
        return self._obs()

    def step(self, action, data=None, reasoning=None):
        if reasoning:
            self.seen_reasoning.append(reasoning)
        name = action.name if hasattr(action, "name") else str(action)
        if name == "RESET":
            return self.reset()
        if self._state.name in ("GAME_OVER", "WIN"):
            return self._obs()
        dr, dc = {
            "ACTION1": (-1, 0), "ACTION2": (1, 0),
            "ACTION3": (0, -1), "ACTION4": (0, 1),
        }.get(name, (0, 0))
        if name == "ACTION6" and data:
            # Click teleports the avatar onto the clicked cell.
            self.pos = [int(data["y"]), int(data["x"])]
        else:
            self.pos[0] = min(self.size - 1, max(0, self.pos[0] + dr))
            self.pos[1] = min(self.size - 1, max(0, self.pos[1] + dc))
        if [self.pos[0], self.pos[1]] == [self.goal[0], self.goal[1]]:
            self._state = _FakeState("WIN")
            self._levels = 1
            self._score = 1
        return self._obs()


def test_adapter_exposes_neutral_actions() -> None:
    """Action names are neutral; RESET is excluded from agent choice."""
    env = Arc3Environment(_MazeEnv())
    env.reset()
    actions = env.actions
    assert "reset" not in actions
    assert set(actions) == {
        "action1", "action2", "action3", "action4",
        "action5", "action6", "action7",
    }
    assert env.is_complex("action6")
    assert not env.is_complex("action1")


def test_adapter_frame_and_state() -> None:
    """Frames pass through; state/score mirror the observation."""
    env = Arc3Environment(_MazeEnv())
    env.reset()
    frame = env.frame()
    assert isinstance(frame, list) and len(frame) == 1
    assert frame[0][5][5] == 3  # avatar cell
    assert frame[0][1][1] == 7  # goal cell
    assert env.state == "PLAYING"
    assert env.score == 0
    assert env.frame_shape == (16, 16)


def test_adapter_step_moves_avatar() -> None:
    env = Arc3Environment(_MazeEnv())
    env.reset()
    env.step("action3")  # left
    frame = env.frame()[0]
    assert frame[5][4] == 3


def test_run_game_wins_maze() -> None:
    """The agent navigates to the goal; WIN flows through honestly."""
    from genesis_cognitive.spatial import SpatialAgent

    mock = _MazeEnv()
    env = Arc3Environment(mock)
    agent = SpatialAgent(seed=0, epsilon=0.0)
    # Teach the effect model as the existing tests do, then let the
    # learned navigation close the distance.
    env.reset()
    agent.observe(env.frame())
    for name, (dr, dc) in {
        "action1": (-1, 0), "action2": (1, 0),
        "action3": (0, -1), "action4": (0, 1),
    }.items():
        st = agent._stat(name)
        st.attempts, st.dr, st.dc = 1, dr, dc
    agent.avatar_color = 3

    from arc_adapter import StepRecord

    records: list[StepRecord] = []
    result = run_game(
        env, agent, max_actions=200, game_id="mock-maze",
        on_step=records.append,
    )
    assert result.won
    assert result.final_state == "WIN"
    assert result.levels_completed == 1
    assert result.steps < 200
    # Audit trail captured every action with state + telemetry.
    assert len(records) == result.steps
    assert all(r.action.startswith("action") for r in records)
    # Reasoning metadata reached the environment (examiner-visible).
    assert mock.seen_reasoning


class _DoomEnv(_MazeEnv):
    """Every non-RESET action is fatal — the runner must keep resetting."""

    def step(self, action, data=None, reasoning=None):
        name = action.name if hasattr(action, "name") else str(action)
        if name == "RESET":
            return self.reset()
        self._state = _FakeState("GAME_OVER")
        return self._obs()


def test_run_game_resets_after_game_over() -> None:
    """Terminal states trigger a runner-owned RESET, not an agent pick."""
    from genesis_cognitive.spatial import SpatialAgent

    env = Arc3Environment(_DoomEnv())
    agent = SpatialAgent(seed=0, epsilon=1.0)
    result = run_game(env, agent, max_actions=10, game_id="mock-doom")
    assert result.resets >= 2  # initial + post-GAME_OVER resets
    assert not result.won
    # Action records exist (it tried things) but never a "reset" pick —
    # resets are owned by the loop, not the agent's action space.
    assert all(r.action != "reset" for r in result.records)
