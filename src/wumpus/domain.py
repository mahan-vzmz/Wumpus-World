"""Immutable domain objects and mutable game-state types.

Coordinates are zero-based internally: Position(0, 0) is the external (1, 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Tile(str, Enum):
    EMPTY = "*"
    PIT = "P"
    WUMPUS = "W"
    WALL = "D"
    GOLD = "G"


class Action(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class Status(str, Enum):
    RUNNING = "RUNNING"
    WON = "WON"
    DEAD_WUMPUS = "DEAD_WUMPUS"
    DEAD_HEALTH = "DEAD_HEALTH"
    NO_SOLUTION = "NO_SOLUTION"
    STEP_LIMIT = "STEP_LIMIT"


_ACTION_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass(frozen=True, slots=True)
class Position:
    """A zero-based row/column coordinate."""

    row: int
    col: int

    def __post_init__(self) -> None:
        if not isinstance(self.row, int) or not isinstance(self.col, int):
            raise TypeError("row and col must be integers")

    def moved(self, action: Action) -> "Position":
        d_row, d_col = _ACTION_DELTAS[action]
        return Position(self.row + d_row, self.col + d_col)

    def neighbors(self) -> tuple["Position", ...]:
        return tuple(self.moved(action) for action in Action)

    def is_inside(self, size: int = 8) -> bool:
        return 0 <= self.row < size and 0 <= self.col < size


@dataclass(frozen=True, slots=True)
class GameMap:
    """The immutable 8x8 environment layout."""

    rows: tuple[tuple[Tile, ...], ...]

    def __post_init__(self) -> None:
        normalized_rows = tuple(tuple(row) for row in self.rows)
        object.__setattr__(self, "rows", normalized_rows)
        if len(normalized_rows) != 8 or any(
            len(row) != 8 for row in normalized_rows
        ):
            raise ValueError("GameMap must contain exactly 8 rows of 8 tiles")

    @classmethod
    def from_rows(cls, rows: tuple[tuple[Tile, ...], ...]) -> "GameMap":
        return cls(rows=rows)

    def tile_at(self, position: Position) -> Tile:
        if not position.is_inside(8):
            raise IndexError(f"position outside map: {position}")
        return self.rows[position.row][position.col]

    def is_inside(self, position: Position) -> bool:
        return position.is_inside(8)


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Static parameters controlling one episode."""

    initial_health: int
    gold_value: int
    pit_score_delta: int
    exit_position: Position
    grid_size: int = 8
    pit_rounding: str = "floor_with_min_one"
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.grid_size != 8:
            raise ValueError("this project currently supports only an 8x8 grid")
        if self.initial_health <= 0:
            raise ValueError("initial_health must be positive")
        if self.gold_value < 0:
            raise ValueError("gold_value cannot be negative")
        if self.pit_score_delta > 0:
            raise ValueError("pit_score_delta must be non-positive")
        if not self.exit_position.is_inside(self.grid_size):
            raise ValueError("exit_position must be inside the grid")
        if self.pit_rounding != "floor_with_min_one":
            raise ValueError("unsupported pit rounding policy")
        if self.max_steps is None:
            object.__setattr__(
                self,
                "max_steps",
                min(self.initial_health, 4 * self.grid_size * self.grid_size),
            )
        elif self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


@dataclass
class GameState:
    """Mutable state for one running episode."""

    position: Position
    health: int
    collected_gold: int = 0
    remaining_gold: frozenset[Position] = frozenset()
    pit_entries: int = 0
    steps: int = 0
    status: Status = Status.RUNNING
    event_log: list[str] = field(default_factory=list)