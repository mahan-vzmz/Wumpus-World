"""Random map generator for dataset creation and benchmarking.

Per SPEC §10:
- Grid size: 8x8
- Start (1,1) -> internal Position(0, 0): no wall, wumpus, or pit.
- Exit (row, col): no wall or wumpus (pit and gold on exit permitted).
- No tile symbol overlaps.
- Must be solvable by A* search (solvability constraint).
- Seed-controlled for 100% reproducibility.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from wumpus.domain import GameConfig, GameMap, Position, Tile
from wumpus.search import solve_astar


@dataclass(frozen=True)
class MapGenerationConfig:
    num_pits: int = 3
    num_wumpus: int = 1
    num_walls: int = 4
    num_golds: int = 2
    initial_health: int = 50
    gold_value: int = 10
    pit_score_delta: int = -15
    exit_position: Position = Position(7, 7)


def generate_map(
    gen_config: MapGenerationConfig = MapGenerationConfig(),
    seed: int = 42,
    max_attempts: int = 1000,
) -> tuple[GameMap, GameConfig]:
    """Generate a valid, solvable Wumpus World map with the specified seed.

    Returns (GameMap, GameConfig).
    Raises RuntimeError if no solvable map could be generated within max_attempts.
    """
    rng = random.Random(seed)
    start_pos = Position(0, 0)
    exit_pos = gen_config.exit_position

    for attempt in range(max_attempts):
        # Create empty grid
        grid: list[list[Tile]] = [
            [Tile.EMPTY for _ in range(8)] for _ in range(8)
        ]

        # Available positions for hazards/items (excluding start)
        available: list[Position] = [
            Position(r, c)
            for r in range(8)
            for c in range(8)
            if Position(r, c) != start_pos
        ]
        rng.shuffle(available)

        # Place Wumpus (cannot be on exit or start)
        wumpus_placed = 0
        wumpus_candidates = [p for p in available if p != exit_pos]
        if len(wumpus_candidates) < gen_config.num_wumpus:
            continue
        for _ in range(gen_config.num_wumpus):
            w_pos = wumpus_candidates.pop()
            available.remove(w_pos)
            grid[w_pos.row][w_pos.col] = Tile.WUMPUS
            wumpus_placed += 1

        # Place Walls (cannot be on exit or start)
        wall_candidates = [p for p in available if p != exit_pos]
        if len(wall_candidates) < gen_config.num_walls:
            continue
        for _ in range(gen_config.num_walls):
            w_pos = wall_candidates.pop()
            available.remove(w_pos)
            grid[w_pos.row][w_pos.col] = Tile.WALL

        # Place Pits (can be on exit, not on start)
        pit_candidates = [p for p in available]
        if len(pit_candidates) < gen_config.num_pits:
            continue
        for _ in range(gen_config.num_pits):
            p_pos = pit_candidates.pop()
            if p_pos in available:
                available.remove(p_pos)
            grid[p_pos.row][p_pos.col] = Tile.PIT

        # Place Golds (can be on exit, not on start)
        gold_candidates = [p for p in available]
        if len(gold_candidates) < gen_config.num_golds:
            continue
        for _ in range(gen_config.num_golds):
            g_pos = gold_candidates.pop()
            if g_pos in available:
                available.remove(g_pos)
            grid[g_pos.row][g_pos.col] = Tile.GOLD

        rows = tuple(tuple(row) for row in grid)
        game_map = GameMap.from_rows(rows)

        config = GameConfig(
            initial_health=gen_config.initial_health,
            gold_value=gen_config.gold_value,
            pit_score_delta=gen_config.pit_score_delta,
            exit_position=exit_pos,
        )

        # Check solvability using A*
        result = solve_astar(game_map, config)
        if result.solved:
            return game_map, config

    raise RuntimeError(
        f"Failed to generate a solvable map after {max_attempts} attempts with seed {seed}"
    )
