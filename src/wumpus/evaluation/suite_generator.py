"""Generate a rich test suite of diverse maps across 5 categories (T600).

Categories:
1. Easy: straight paths, high health, few obstacles
2. Pit-Heavy: 3-5 pits testing pit detection and avoidance
3. Wumpus-Hazard: 1-2 Wumpuses blocking direct paths
4. Gold-Hunter: 2-3 Gold tiles rewarding risk/reward detours
5. Hard-Complex: Low health, walls, pits, wumpus, and gold combined
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from wumpus.domain import Position
from wumpus.generator import MapGenerationConfig, generate_map
from wumpus.parser import parse_input


@dataclass
class SuiteCategoryConfig:
    name: str
    num_maps: int
    map_gen_config: MapGenerationConfig


SUITE_CATEGORIES = [
    SuiteCategoryConfig(
        name="01_easy",
        num_maps=4,
        map_gen_config=MapGenerationConfig(
            num_pits=0, num_wumpus=0, num_walls=2, num_golds=1, initial_health=50
        ),
    ),
    SuiteCategoryConfig(
        name="02_pit_heavy",
        num_maps=4,
        map_gen_config=MapGenerationConfig(
            num_pits=4, num_wumpus=0, num_walls=3, num_golds=1, initial_health=40
        ),
    ),
    SuiteCategoryConfig(
        name="03_wumpus_hazard",
        num_maps=4,
        map_gen_config=MapGenerationConfig(
            num_pits=1, num_wumpus=2, num_walls=3, num_golds=1, initial_health=40
        ),
    ),
    SuiteCategoryConfig(
        name="04_gold_hunter",
        num_maps=4,
        map_gen_config=MapGenerationConfig(
            num_pits=2, num_wumpus=1, num_walls=2, num_golds=3, initial_health=45
        ),
    ),
    SuiteCategoryConfig(
        name="05_hard_complex",
        num_maps=4,
        map_gen_config=MapGenerationConfig(
            num_pits=3, num_wumpus=1, num_walls=4, num_golds=2, initial_health=30
        ),
    ),
]


def _format_map_file(game_map, config) -> str:
    """Format GameMap and GameConfig back to the official 12-line text format."""
    lines = []
    for row in game_map.rows:
        lines.append("".join(tile.value for tile in row))
    lines.append(str(config.initial_health))
    lines.append(str(config.gold_value))
    lines.append(str(config.pit_score_delta))
    ext_row = config.exit_position.row + 1
    ext_col = config.exit_position.col + 1
    lines.append(f"{ext_row} {ext_col}")
    return "\n".join(lines) + "\n"


def generate_map_suite(output_dir: Path, base_seed: int = 500) -> list[Path]:
    """Generate 20 test maps across 5 categories and write them to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []
    current_seed = base_seed

    for cat in SUITE_CATEGORIES:
        for idx in range(1, cat.num_maps + 1):
            map_name = f"{cat.name}_map_{idx:02d}.txt"
            file_path = output_dir / map_name

            game_map, config = generate_map(cat.map_gen_config, seed=current_seed)
            current_seed += 1

            content = _format_map_file(game_map, config)
            file_path.write_text(content, encoding="utf-8")
            generated_files.append(file_path)

    return generated_files
