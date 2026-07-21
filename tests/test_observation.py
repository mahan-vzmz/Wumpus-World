"""Tests for Observation / percepts (T104).

Covers: breeze, stench, glitter, at_exit, legal_actions,
corner/edge cases, diagonal non-effect, and no hidden info leak.
"""

import pytest

from wumpus.domain import Action, GameConfig, GameMap, GameState, Position, Status, Tile
from wumpus.engine import init_state, step
from wumpus.observation import Observation, make_observation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _map_from_string(text: str) -> GameMap:
    tiles = {t.value: t for t in Tile}
    rows: list[tuple[Tile, ...]] = []
    for line in text.strip().splitlines():
        rows.append(tuple(tiles[ch] for ch in line.strip()))
    return GameMap.from_rows(tuple(rows))


def _cfg(exit_pos: Position = Position(7, 7), health: int = 50) -> GameConfig:
    return GameConfig(
        initial_health=health, gold_value=10,
        pit_score_delta=-15, exit_position=exit_pos,
    )


# ---------------------------------------------------------------------------
# Breeze tests
# ---------------------------------------------------------------------------

class TestBreeze:

    def test_breeze_adjacent_to_pit(self) -> None:
        """Breeze on all 4 orthogonal neighbors of a pit."""
        gm = _map_from_string(
            "********\n"
            "********\n"
            "********\n"
            "***P****\n"   # pit at (4,4) = 0-indexed (3,3)
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _cfg()
        pit = Position(3, 3)

        # Check all 4 neighbors of pit
        for neighbor in pit.neighbors():
            if neighbor.is_inside(8):
                state = GameState(
                    position=neighbor, health=50,
                    remaining_gold=frozenset(), status=Status.RUNNING,
                )
                obs = make_observation(gm, cfg, state)
                assert obs.breeze is True, f"expected breeze at {neighbor}"

    def test_no_breeze_away_from_pit(self) -> None:
        gm = _map_from_string(
            "********\n"
            "********\n"
            "********\n"
            "***P****\n"
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _cfg()
        # (0,0) is far from pit at (3,3)
        state = GameState(
            position=Position(0, 0), health=50,
            remaining_gold=frozenset(), status=Status.RUNNING,
        )
        obs = make_observation(gm, cfg, state)
        assert obs.breeze is False

    def test_no_breeze_diagonal_to_pit(self) -> None:
        """Diagonal neighbors of pit should NOT have breeze."""
        gm = _map_from_string(
            "********\n"
            "********\n"
            "********\n"
            "***P****\n"   # pit at (3,3)
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _cfg()
        # (2,2) is diagonal to (3,3)
        state = GameState(
            position=Position(2, 2), health=50,
            remaining_gold=frozenset(), status=Status.RUNNING,
        )
        obs = make_observation(gm, cfg, state)
        assert obs.breeze is False


# ---------------------------------------------------------------------------
# Stench tests
# ---------------------------------------------------------------------------

class TestStench:

    def test_stench_adjacent_to_wumpus(self) -> None:
        gm = _map_from_string(
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "****W***\n"   # wumpus at (4,4)
            "********\n"
            "********\n"
            "********"
        )
        cfg = _cfg()
        wumpus = Position(4, 4)

        for neighbor in wumpus.neighbors():
            if neighbor.is_inside(8):
                state = GameState(
                    position=neighbor, health=50,
                    remaining_gold=frozenset(), status=Status.RUNNING,
                )
                obs = make_observation(gm, cfg, state)
                assert obs.stench is True, f"expected stench at {neighbor}"

    def test_no_stench_diagonal_to_wumpus(self) -> None:
        gm = _map_from_string(
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "****W***\n"   # wumpus at (4,4)
            "********\n"
            "********\n"
            "********"
        )
        cfg = _cfg()
        state = GameState(
            position=Position(3, 3), health=50,
            remaining_gold=frozenset(), status=Status.RUNNING,
        )
        obs = make_observation(gm, cfg, state)
        assert obs.stench is False


# ---------------------------------------------------------------------------
# Glitter tests
# ---------------------------------------------------------------------------

class TestGlitter:

    def test_glitter_on_gold_cell(self) -> None:
        gm = _map_from_string(
            "********\n"
            "*G******\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _cfg()
        gold_pos = Position(1, 1)
        state = GameState(
            position=gold_pos, health=50,
            remaining_gold=frozenset({gold_pos}), status=Status.RUNNING,
        )
        obs = make_observation(gm, cfg, state)
        assert obs.glitter is True

    def test_no_glitter_after_collection(self) -> None:
        """Gold already collected → no glitter."""
        gm = _map_from_string(
            "********\n"
            "*G******\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _cfg()
        gold_pos = Position(1, 1)
        state = GameState(
            position=gold_pos, health=50, collected_gold=1,
            remaining_gold=frozenset(), status=Status.RUNNING,  # already collected
        )
        obs = make_observation(gm, cfg, state)
        assert obs.glitter is False

    def test_no_glitter_on_adjacent_cell(self) -> None:
        """Glitter only on the exact gold cell, not neighbors."""
        gm = _map_from_string(
            "********\n"
            "*G******\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _cfg()
        state = GameState(
            position=Position(0, 1), health=50,  # above gold
            remaining_gold=frozenset({Position(1, 1)}), status=Status.RUNNING,
        )
        obs = make_observation(gm, cfg, state)
        assert obs.glitter is False


# ---------------------------------------------------------------------------
# At-exit tests
# ---------------------------------------------------------------------------

class TestAtExit:

    def test_at_exit_on_exit_cell(self) -> None:
        gm = _map_from_string(
            "********\n" * 8
        )
        cfg = _cfg(exit_pos=Position(7, 7))
        state = GameState(
            position=Position(7, 7), health=50,
            remaining_gold=frozenset(), status=Status.RUNNING,
        )
        obs = make_observation(gm, cfg, state)
        assert obs.at_exit is True

    def test_not_at_exit_on_other_cell(self) -> None:
        gm = _map_from_string(
            "********\n" * 8
        )
        cfg = _cfg(exit_pos=Position(7, 7))
        state = GameState(
            position=Position(0, 0), health=50,
            remaining_gold=frozenset(), status=Status.RUNNING,
        )
        obs = make_observation(gm, cfg, state)
        assert obs.at_exit is False


# ---------------------------------------------------------------------------
# Legal actions
# ---------------------------------------------------------------------------

class TestLegalActions:

    def test_corner_has_two_legal_actions(self) -> None:
        """(0,0) corner: only DOWN and RIGHT are valid."""
        gm = _map_from_string("********\n" * 8)
        cfg = _cfg()
        state = GameState(
            position=Position(0, 0), health=50,
            remaining_gold=frozenset(), status=Status.RUNNING,
        )
        obs = make_observation(gm, cfg, state)
        assert set(obs.legal_actions) == {Action.DOWN, Action.RIGHT}

    def test_edge_has_three_legal_actions(self) -> None:
        """(0,3) top edge: DOWN, LEFT, RIGHT."""
        gm = _map_from_string("********\n" * 8)
        cfg = _cfg()
        state = GameState(
            position=Position(0, 3), health=50,
            remaining_gold=frozenset(), status=Status.RUNNING,
        )
        obs = make_observation(gm, cfg, state)
        assert set(obs.legal_actions) == {Action.DOWN, Action.LEFT, Action.RIGHT}

    def test_center_has_four_legal_actions(self) -> None:
        gm = _map_from_string("********\n" * 8)
        cfg = _cfg()
        state = GameState(
            position=Position(3, 3), health=50,
            remaining_gold=frozenset(), status=Status.RUNNING,
        )
        obs = make_observation(gm, cfg, state)
        assert set(obs.legal_actions) == {Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT}

    def test_wall_blocks_action(self) -> None:
        """Wall to the right of agent → RIGHT not legal."""
        gm = _map_from_string(
            "*D******\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _cfg()
        state = GameState(
            position=Position(0, 0), health=50,
            remaining_gold=frozenset(), status=Status.RUNNING,
        )
        obs = make_observation(gm, cfg, state)
        assert Action.RIGHT not in obs.legal_actions
        assert Action.DOWN in obs.legal_actions


# ---------------------------------------------------------------------------
# Integration: observation after engine step
# ---------------------------------------------------------------------------

class TestObservationIntegration:

    def test_observation_after_step_near_hazards(self) -> None:
        """Moving next to both pit and wumpus gives breeze+stench."""
        gm = _map_from_string(
            "********\n"
            "********\n"
            "***P****\n"   # pit at (2,3)
            "***W****\n"   # wumpus at (3,3) - but we won't step on it
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _cfg()
        state = init_state(gm, cfg)

        # Move to (1,3): neighbor of pit (2,3) → breeze
        step(gm, cfg, state, Action.DOWN)     # (1,0)
        step(gm, cfg, state, Action.RIGHT)    # (1,1)
        step(gm, cfg, state, Action.RIGHT)    # (1,2)
        step(gm, cfg, state, Action.RIGHT)    # (1,3)

        obs = make_observation(gm, cfg, state)
        assert obs.breeze is True    # adjacent to pit at (2,3)
        assert obs.stench is False   # wumpus at (3,3) is 2 rows away
        assert obs.position == Position(1, 3)
        assert obs.health == 46      # 50 - 4 steps
        assert obs.steps == 4

    def test_observation_fields_match_state(self) -> None:
        """Observation reflects the current state accurately."""
        gm = _map_from_string("********\n" * 8)
        cfg = _cfg()
        state = init_state(gm, cfg)
        step(gm, cfg, state, Action.RIGHT)

        obs = make_observation(gm, cfg, state)
        assert obs.position == state.position
        assert obs.health == state.health
        assert obs.collected_gold == state.collected_gold
        assert obs.steps == state.steps
        assert obs.status == state.status

    def test_observation_does_not_expose_hidden_map(self) -> None:
        """Observation object has no reference to the GameMap."""
        gm = _map_from_string(
            "********\n"
            "***W****\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _cfg()
        state = init_state(gm, cfg)
        obs = make_observation(gm, cfg, state)

        # Check that the Observation object has no map-related attributes
        obs_attrs = set(dir(obs))
        assert "game_map" not in obs_attrs
        assert "map" not in obs_attrs
        assert "rows" not in obs_attrs
        assert "tiles" not in obs_attrs
