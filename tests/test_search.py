"""Tests for the A* search solver and SearchAgent.

Covers:
  T300 — SearchProblem definition (loss = C - score)
  T302 — A* correctness on golden examples
  T303 — Heuristic admissibility checks
  T304 — SearchAgent integration with Runner
"""

from pathlib import Path

import pytest

from wumpus.agents.search_agent import SearchAgent
from wumpus.domain import Action, GameConfig, GameMap, Position, Status, Tile
from wumpus.engine import compute_score, init_state, step
from wumpus.parser import parse_input
from wumpus.runner import run_episode
from wumpus.search import SearchResult, SearchState, _manhattan, solve_astar

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helper: build a simple map from strings
# ---------------------------------------------------------------------------

def _map_from_strings(rows: list[str]) -> GameMap:
    """Build a GameMap from 8 strings of 8 chars each."""
    tile_map = {t.value: t for t in Tile}
    parsed = tuple(tuple(tile_map[ch] for ch in row) for row in rows)
    return GameMap.from_rows(parsed)


def _default_config(exit_pos: Position = Position(7, 7),
                     health: int = 50,
                     gold_value: int = 10,
                     pit_delta: int = -15) -> GameConfig:
    return GameConfig(
        initial_health=health,
        gold_value=gold_value,
        pit_score_delta=pit_delta,
        exit_position=exit_pos,
    )


# ===================================================================
# T300: SearchProblem — verify loss/score relationship
# ===================================================================

class TestSearchProblemDefinition:
    """Verify that C - loss == final_score for solved plans."""

    def test_loss_score_identity_golden1(self):
        """Golden 1: empty map, 14 moves, score=36."""
        parsed = parse_input((FIXTURES / "golden1_straight.txt").read_text())
        result = solve_astar(parsed.game_map, parsed.config)

        assert result.solved
        assert result.predicted_score == 36

        # Verify via engine replay
        state = init_state(parsed.game_map, parsed.config)
        for action in result.plan:
            state = step(parsed.game_map, parsed.config, state, action)
        assert state.status == Status.WON
        assert compute_score(state, parsed.config) == 36

    def test_loss_score_identity_golden3(self):
        """Golden 3: gold + exit optimal, score=24."""
        parsed = parse_input((FIXTURES / "golden3_complex.txt").read_text())
        result = solve_astar(parsed.game_map, parsed.config)

        assert result.solved
        # A* finds optimal: 14 moves collecting gold, score = 16 + 10 = 26
        # (better than the manual example's 16-move backtracking path scoring 24)
        assert result.predicted_score == 26

        # Replay and verify
        state = init_state(parsed.game_map, parsed.config)
        for action in result.plan:
            state = step(parsed.game_map, parsed.config, state, action)
        assert state.status == Status.WON
        assert compute_score(state, parsed.config) == result.predicted_score


# ===================================================================
# T302: A* correctness
# ===================================================================

class TestAStarSolver:
    """Verify A* finds optimal plans on golden examples."""

    def test_golden1_straight_path(self):
        """Empty 8x8 map, exit (8,8). Optimal: 14 moves, score 36."""
        parsed = parse_input((FIXTURES / "golden1_straight.txt").read_text())
        result = solve_astar(parsed.game_map, parsed.config)

        assert result.solved
        assert len(result.plan) == 14
        assert result.predicted_score == 36
        assert result.expanded_nodes > 0

    def test_golden2_avoids_pit(self):
        """Map with pit at (4,5). Optimal path is 14 moves (7 DOWN + 7 RIGHT)
        which avoids pit entirely. Score = 50 - 14 = 36.
        The pit at (4,5) is NOT on the direct diagonal path."""
        parsed = parse_input((FIXTURES / "golden2_pit.txt").read_text())
        result = solve_astar(parsed.game_map, parsed.config)

        assert result.solved
        assert result.predicted_score == 36
        assert len(result.plan) == 14

    def test_golden3_collects_gold(self):
        """Map with gold, wall, wumpus, pit. Optimal collects gold: score 24."""
        parsed = parse_input((FIXTURES / "golden3_complex.txt").read_text())
        result = solve_astar(parsed.game_map, parsed.config)

        assert result.solved
        # A* finds 14-move path collecting gold: score 26
        assert result.predicted_score == 26
        assert len(result.plan) == 14

    def test_no_path_blocked(self):
        """Completely walled-off exit — should return solved=False."""
        gmap = _map_from_strings([
            "********",
            "********",
            "********",
            "********",
            "********",
            "********",
            "******D*",
            "******D*",
        ])
        # Exit at (7,7) is blocked by walls at (6,6) and (7,6)
        # Actually we need to fully block it:
        gmap2 = _map_from_strings([
            "********",
            "********",
            "********",
            "********",
            "********",
            "********",
            "*******D",
            "******D*",
        ])
        config = _default_config(exit_pos=Position(7, 7), health=50)
        result = solve_astar(gmap2, config)
        assert result.solved is False

    def test_wumpus_avoided(self):
        """Wumpus blocking direct path — A* must find safe detour."""
        gmap = _map_from_strings([
            "********",
            "W*******",
            "********",
            "********",
            "********",
            "********",
            "********",
            "********",
        ])
        config = _default_config(exit_pos=Position(7, 7), health=50)
        result = solve_astar(gmap, config)

        assert result.solved
        # Replay and verify agent doesn't die
        state = init_state(gmap, config)
        for action in result.plan:
            state = step(gmap, config, state, action)
        assert state.status == Status.WON

    def test_low_health_prefers_short_path(self):
        """With barely enough health, A* must find the shortest path."""
        gmap = _map_from_strings([
            "********",
            "********",
            "********",
            "********",
            "********",
            "********",
            "********",
            "********",
        ])
        # Exit at (0,7): 7 moves right. Health=8 means just enough.
        config = _default_config(exit_pos=Position(0, 7), health=8)
        result = solve_astar(gmap, config)

        assert result.solved
        assert len(result.plan) == 7
        assert result.predicted_score == 1  # 8 - 7 = 1

    def test_insufficient_health(self):
        """Health too low to reach exit — no solution."""
        gmap = _map_from_strings([
            "********",
            "********",
            "********",
            "********",
            "********",
            "********",
            "********",
            "********",
        ])
        # Exit (7,7) requires 14 moves but only 10 health
        config = _default_config(exit_pos=Position(7, 7), health=10)
        result = solve_astar(gmap, config)

        assert result.solved is False

    def test_plan_actions_are_all_legal(self):
        """Every action in the plan must be legal when replayed."""
        parsed = parse_input((FIXTURES / "golden3_complex.txt").read_text())
        result = solve_astar(parsed.game_map, parsed.config)

        assert result.solved
        state = init_state(parsed.game_map, parsed.config)
        for action in result.plan:
            # step() will raise ValueError if action is illegal
            state = step(parsed.game_map, parsed.config, state, action)
        assert state.status == Status.WON

    def test_predicted_score_matches_engine(self):
        """A*'s predicted score must exactly match engine's computed score."""
        for fixture in ["golden1_straight.txt", "golden2_pit.txt", "golden3_complex.txt"]:
            parsed = parse_input((FIXTURES / fixture).read_text())
            result = solve_astar(parsed.game_map, parsed.config)
            assert result.solved, f"Failed to solve {fixture}"

            state = init_state(parsed.game_map, parsed.config)
            for action in result.plan:
                state = step(parsed.game_map, parsed.config, state, action)

            engine_score = compute_score(state, parsed.config)
            assert result.predicted_score == engine_score, (
                f"{fixture}: predicted {result.predicted_score} != engine {engine_score}"
            )


# ===================================================================
# T303: Heuristic admissibility
# ===================================================================

class TestHeuristic:
    """Verify Manhattan heuristic never overestimates."""

    def test_manhattan_at_exit_is_zero(self):
        exit_pos = Position(7, 7)
        assert _manhattan(Position(7, 7), exit_pos) == 0

    def test_manhattan_basic(self):
        assert _manhattan(Position(0, 0), Position(7, 7)) == 14
        assert _manhattan(Position(3, 2), Position(5, 6)) == 6

    def test_heuristic_never_overestimates(self):
        """On all golden examples, h(start) <= actual cost of optimal plan."""
        for fixture in ["golden1_straight.txt", "golden2_pit.txt", "golden3_complex.txt"]:
            parsed = parse_input((FIXTURES / fixture).read_text())
            result = solve_astar(parsed.game_map, parsed.config)
            assert result.solved

            h_start = _manhattan(Position(0, 0), parsed.config.exit_position)
            # Actual movement cost = number of steps in plan
            assert h_start <= len(result.plan), (
                f"{fixture}: h={h_start} > plan_length={len(result.plan)}"
            )


# ===================================================================
# T304: SearchAgent integration with Runner
# ===================================================================

class TestSearchAgentIntegration:
    """Verify SearchAgent works end-to-end through the Runner."""

    def _run_search_agent(self, fixture_name: str):
        """Helper: run SearchAgent on a fixture via the runner."""
        parsed = parse_input((FIXTURES / fixture_name).read_text())
        agent = SearchAgent()

        # SearchAgent needs game_map in public_map_info
        public_info = {
            "grid_size": parsed.config.grid_size,
            "exit_position": parsed.config.exit_position,
            "game_map": parsed.game_map,
        }

        agent.reset(parsed.config, public_info, seed=42)
        state = init_state(parsed.game_map, parsed.config)

        while state.status == Status.RUNNING:
            from wumpus.observation import make_observation
            obs = make_observation(parsed.game_map, parsed.config, state)
            action = agent.choose_action(obs)
            state = step(parsed.game_map, parsed.config, state, action)

        return state, parsed.config, agent

    def test_golden1_wins(self):
        state, config, agent = self._run_search_agent("golden1_straight.txt")
        assert state.status == Status.WON
        assert compute_score(state, config) == 36

    def test_golden2_wins_optimally(self):
        state, config, agent = self._run_search_agent("golden2_pit.txt")
        assert state.status == Status.WON
        assert compute_score(state, config) == 36

    def test_golden3_wins_optimally(self):
        state, config, agent = self._run_search_agent("golden3_complex.txt")
        assert state.status == Status.WON
        assert compute_score(state, config) == 26

    def test_search_result_accessible(self):
        """Agent exposes search diagnostics."""
        _, _, agent = self._run_search_agent("golden1_straight.txt")
        sr = agent.search_result
        assert sr is not None
        assert sr.solved
        assert sr.expanded_nodes > 0
        assert sr.planning_time_ms >= 0

    def test_reproducible_with_same_seed(self):
        """Same fixture + seed → identical plan."""
        s1, c1, a1 = self._run_search_agent("golden3_complex.txt")
        s2, c2, a2 = self._run_search_agent("golden3_complex.txt")
        assert a1.search_result.plan == a2.search_result.plan
