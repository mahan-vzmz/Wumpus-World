"""Tests for the map generator: config validation and injectable solvability."""

import pytest

from wumpus.core.domain import Position
from wumpus.core.generator import MapGenerationConfig, generate_map


class TestMapGenerationConfigValidation:

    def test_negative_count_is_rejected(self):
        with pytest.raises(ValueError):
            MapGenerationConfig(num_pits=-1)

    def test_too_many_objects_is_rejected(self):
        with pytest.raises(ValueError):
            MapGenerationConfig(num_pits=30, num_walls=30, num_golds=10)

    def test_non_positive_health_is_rejected(self):
        with pytest.raises(ValueError):
            MapGenerationConfig(initial_health=0)

    def test_positive_pit_penalty_is_rejected(self):
        with pytest.raises(ValueError):
            MapGenerationConfig(pit_score_delta=5)

    def test_default_config_is_valid(self):
        cfg = MapGenerationConfig()
        assert cfg.num_pits == 3


class TestInjectableSolvability:

    def test_injected_checker_is_used(self):
        """generate_map consults the injected predicate instead of the default."""
        calls: list[int] = []

        def always_solvable(_game_map, _config) -> bool:
            calls.append(1)
            return True

        game_map, config = generate_map(
            MapGenerationConfig(), seed=1, is_solvable=always_solvable
        )
        assert calls  # the injected checker was consulted
        assert config.exit_position == Position(7, 7)

    def test_unsatisfiable_checker_raises_runtime_error(self):
        with pytest.raises(RuntimeError):
            generate_map(
                MapGenerationConfig(),
                seed=1,
                max_attempts=3,
                is_solvable=lambda _gm, _cfg: False,
            )

    def test_default_generates_a_solvable_map(self):
        from wumpus.ai.search import solve_astar

        game_map, config = generate_map(MapGenerationConfig(), seed=42)
        assert solve_astar(game_map, config).solved
