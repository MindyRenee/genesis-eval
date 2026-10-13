"""World-play tests — the mind's session layer over interactive worlds.

The worlds here are abstract mock environments — no game content.
They exist to prove the session contract: the Mind chooses to engage,
the substrate steps, episode boundaries surface as outcomes the Mind
feels and remembers, and progress persists across sessions.
"""

from __future__ import annotations

from world_play import WorldPlay


class _WorldEnv:
    """Minimal duck-typed world: avatar (3) moves on a grid,
    reaching the goal (7) wins; 'doom' kills instead."""

    def __init__(self, doom: bool = False, size: int = 8) -> None:
        self.size = size
        self.doom = doom
        self.pos = [size - 2, size - 2]
        self.goal = [0, 0]
        self._state = "NOT_PLAYED"
        self._levels = 0
        self._win_levels = 1
        self._score = 0

    # ── the world protocol WorldPlay drives ────────────────────
    @property
    def actions(self) -> list[str]:
        return ["action1", "action2", "action3", "action4"]

    def is_complex(self, action: str) -> bool:
        return False

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self.size, self.size)

    def frame(self):
        g = [[0] * self.size for _ in range(self.size)]
        g[self.goal[0]][self.goal[1]] = 7
        g[self.pos[0]][self.pos[1]] = 3
        return [g]

    @property
    def state(self) -> str:
        return self._state

    @property
    def levels_completed(self) -> int:
        return self._levels

    @property
    def win_levels(self) -> int:
        return self._win_levels

    @property
    def score(self) -> int:
        return self._score

    def reset(self):
        self.pos = [self.size - 2, self.size - 2]
        self._state = "PLAYING"
        return self.frame()

    def step(self, action, data=None, reasoning=None):
        if self.doom:
            self._state = "GAME_OVER"
            return self.frame()
        dr, dc = {
            "action1": (-1, 0), "action2": (1, 0),
            "action3": (0, -1), "action4": (0, 1),
        }.get(action, (0, 0))
        self.pos[0] = min(self.size - 1, max(0, self.pos[0] + dr))
        self.pos[1] = min(self.size - 1, max(0, self.pos[1] + dc))
        if self.pos == self.goal:
            self._state = "WIN"
            self._levels = 1
            self._score = 1
        return self.frame()


def test_burst_reports_episodes_and_wins(tmp_path) -> None:
    """A session steps the world; WIN ends it as a reported episode."""
    play = WorldPlay(str(tmp_path))
    play.attach(_WorldEnv(), "world-a", seed=0)
    seen: list[str] = []
    report = play.play_burst(
        "world-a", max_steps=100,
        on_episode=lambda o: seen.append(o.state),
    )
    assert report is not None
    assert report.steps > 0
    # Whether or not it found the goal this burst, every terminal
    # episode surfaced through the callback.
    if report.won:
        assert seen == ["WIN"]
        assert report.progress == 1.0


def test_doom_world_records_deaths(tmp_path) -> None:
    """Every GAME_OVER surfaces as an episode outcome and a death."""
    play = WorldPlay(str(tmp_path))
    play.attach(_WorldEnv(doom=True), "doom", seed=0)
    deaths: list[str] = []
    report = play.play_burst(
        "doom", max_steps=20,
        on_episode=lambda o: deaths.append(o.state),
    )
    assert report is not None
    assert not report.won
    assert deaths and all(s == "GAME_OVER" for s in deaths)
    # One death per step — each action ends a life, then reset.
    assert report.episodes == [
        e for e in report.episodes if e.state == "GAME_OVER"
    ]
    st = play.stats["doom"]
    assert st["deaths"] == len(deaths)
    assert st["episodes"] == len(deaths)


def test_pending_and_progress_persist(tmp_path) -> None:
    """has_pending tracks unbeaten worlds; stats survive reload."""
    play = WorldPlay(str(tmp_path))
    assert not play.has_pending()  # nothing installed
    play.attach(_WorldEnv(), "world-b", seed=1)
    assert play.has_pending()

    play.play_burst("world-b", max_steps=5)
    reloaded = WorldPlay(str(tmp_path))
    st = reloaded.stats["world-b"]
    assert st["sessions"] == 1
    assert st["steps"] == 5

    # A world with a recorded win is no longer pending.
    reloaded.attach(_WorldEnv(), "world-b", seed=1)
    reloaded.stats["world-b"]["wins"] = 1
    assert not reloaded.has_pending()


def test_unknown_world_returns_none(tmp_path) -> None:
    play = WorldPlay(str(tmp_path))
    assert play.play_burst("ghost") is None


def test_burst_callbacks_fire(tmp_path) -> None:
    """The engagement contract: on_episode fires per episode outcome
    and on_step per action — a driver can feel, log, and remember
    without reaching inside the session."""
    play = WorldPlay(str(tmp_path))
    play.attach(_WorldEnv(doom=True), "doom", seed=0)

    episodes: list = []
    steps: list = []
    report = play.play_burst(
        "doom", max_steps=400,
        on_episode=episodes.append,
        on_step=steps.append,
    )

    assert report is not None and report.steps == len(steps)
    assert episodes and all(o.state == "GAME_OVER" for o in episodes)
    assert report.episodes == episodes
