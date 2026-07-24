"""Parser and validation for the official text input format."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .domain import GameConfig, GameMap, Position, Tile


class InputFormatError(ValueError):
    """Raised when an input file violates the project specification."""


@dataclass(frozen=True, slots=True)
class ParsedInput:
    game_map: GameMap
    config: GameConfig
    warnings: tuple[str, ...] = ()


_EXIT_RE = re.compile(r"^\(?\s*(\d+)\s*(?:,|\s)\s*(\d+)\s*\)?$")
_VALID_TILES = {tile.value: tile for tile in Tile}


def _parse_int(value: str, field_name: str) -> int:
    try:
        return int(value.strip())
    except ValueError as exc:
        raise InputFormatError(f"{field_name} must be an integer") from exc


def _parse_exit(value: str) -> Position:
    match = _EXIT_RE.fullmatch(value.strip())
    if match is None:
        raise InputFormatError(
            "exit must use row col, row,col, or (row,col) format"
        )
    external_row, external_col = (int(part) for part in match.groups())
    if external_row < 1 or external_col < 1:
        raise InputFormatError("exit coordinates are one-based and positive")
    return Position(external_row - 1, external_col - 1)


def parse_input(text: str) -> ParsedInput:
    """Parse the 8 map rows followed by the four parameter lines."""
    raw_lines = text.splitlines()
    while raw_lines and not raw_lines[-1].strip():
        raw_lines.pop()
    if len(raw_lines) != 12:
        raise InputFormatError("input must contain exactly 12 non-empty lines")

    map_rows: list[tuple[Tile, ...]] = []
    for row_index, raw_row in enumerate(raw_lines[:8], start=1):
        row = raw_row.strip()
        if len(row) != 8:
            raise InputFormatError(f"map row {row_index} must contain 8 symbols")
        unknown = sorted(set(row) - set(_VALID_TILES))
        if unknown:
            raise InputFormatError(
                f"map row {row_index} contains invalid symbol(s): {unknown}"
            )
        map_rows.append(tuple(_VALID_TILES[symbol] for symbol in row))

    game_map = GameMap.from_rows(tuple(map_rows))
    initial_health = _parse_int(raw_lines[8], "initial_health")
    gold_value = _parse_int(raw_lines[9], "gold_value")
    pit_score_delta = _parse_int(raw_lines[10], "pit_score_delta")
    exit_position = _parse_exit(raw_lines[11])

    warnings: list[str] = []
    if pit_score_delta > 0:
        pit_score_delta = -pit_score_delta
        warnings.append("positive pit score was normalized to a negative delta")

    start = Position(0, 0)
    if game_map.tile_at(start) in {Tile.WALL, Tile.WUMPUS, Tile.PIT}:
        raise InputFormatError("start (1,1) cannot contain a wall, Wumpus, or pit")
    if not exit_position.is_inside(8):
        raise InputFormatError("exit must be inside the 8x8 grid")
    exit_tile = game_map.tile_at(exit_position)
    if exit_tile in {Tile.WALL, Tile.WUMPUS}:
        raise InputFormatError("exit cannot contain a wall or Wumpus")

    try:
        config = GameConfig(
            initial_health=initial_health,
            gold_value=gold_value,
            pit_score_delta=pit_score_delta,
            exit_position=exit_position,
        )
    except ValueError as exc:
        raise InputFormatError(str(exc)) from exc
    return ParsedInput(game_map=game_map, config=config, warnings=tuple(warnings))