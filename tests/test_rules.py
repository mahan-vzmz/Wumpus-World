"""Tests for KnowledgeBase and RuleAgent (Epic 4).

Covers:
  T400 — KnowledgeBase vocabulary and 3-valued logic (SAFE, DANGER, UNKNOWN)
  T401 — No-percept inference (no breeze/stench implies safe neighbors)
  T402 — Percept presence, hazard candidates, and single-candidate elimination
  T403 — Safe frontier exploration and BFS pathfinding
  T404 — Utility policy and safe retreat to exit
  T405 — Reasoning trace verification
"""

from pathlib import Path

import pytest

from wumpus.agents.rule_agent import RuleAgent
from wumpus.domain import Action, GameConfig, GameMap, Position, Status, Tile
from wumpus.engine import compute_score, init_state, step
from wumpus.knowledge import CellStatus, KnowledgeBase
from wumpus.observation import make_observation
from wumpus.parser import parse_input
from wumpus.runner import run_episode

FIXTURES = Path(__file__).parent / "fixtures"


def _default_config(exit_pos: Position = Position(7, 7), health: int = 50) -> GameConfig:
    return GameConfig(
        initial_health=health,
        gold_value=10,
        pit_score_delta=-15,
        exit_position=exit_pos,
    )


# ===================================================================
# T400 & T401: KnowledgeBase basic inference
# ===================================================================

class TestKnowledgeBaseInference:

    def test_start_position_is_visited_and_safe(self):
        kb = KnowledgeBase(grid_size=8)
        pos = Position(0, 0)
        kb.update(pos, breeze=False, stench=False, glitter=False,
                  legal_actions=(Action.RIGHT, Action.DOWN))

        assert kb.is_visited(pos)
        assert kb.is_safe(pos)

    def test_no_percept_marks_neighbors_safe(self):
        """No breeze and no stench at (0,0) -> neighbors (0,1) and (1,0) are SAFE."""
        kb = KnowledgeBase(grid_size=8)
        pos = Position(0, 0)
        kb.update(pos, breeze=False, stench=False, glitter=False,
                  legal_actions=(Action.RIGHT, Action.DOWN))

        assert kb.status(Position(0, 1)) == CellStatus.SAFE
        assert kb.status(Position(1, 0)) == CellStatus.SAFE

    def test_breeze_marks_neighbors_possible_pit(self):
        """Breeze at (0,0) -> unvisited neighbors become POSSIBLE_PIT, not confirmed."""
        kb = KnowledgeBase(grid_size=8)
        pos = Position(0, 0)
        kb.update(pos, breeze=True, stench=False, glitter=False,
                  legal_actions=(Action.RIGHT, Action.DOWN))

        assert kb.status(Position(0, 1)) == CellStatus.POSSIBLE_PIT
        assert kb.status(Position(1, 0)) == CellStatus.POSSIBLE_PIT
        # UNKNOWN is distinct from unsafe, but these are now POSSIBLE_PIT
        assert not kb.is_safe(Position(0, 1))
        assert not kb.is_dangerous(Position(0, 1))  # Not confirmed yet!

    def test_stench_marks_neighbors_possible_wumpus(self):
        """Stench at (0,0) -> unvisited neighbors become POSSIBLE_WUMPUS."""
        kb = KnowledgeBase(grid_size=8)
        pos = Position(0, 0)
        kb.update(pos, breeze=False, stench=True, glitter=False,
                  legal_actions=(Action.RIGHT, Action.DOWN))

        assert kb.status(Position(0, 1)) == CellStatus.POSSIBLE_WUMPUS
        assert kb.status(Position(1, 0)) == CellStatus.POSSIBLE_WUMPUS

    def test_clearing_suspicion_when_visited_elsewhere_no_breeze(self):
        """
        1. Breeze at (0,0) -> (0,1) and (1,0) are POSSIBLE_PIT.
        2. Move to (1,0): no breeze! -> (0,0) and (2,0) and (1,1) have no pit.
        3. Therefore (1,1) is safe, and (0,1) is no longer implicated by (1,0).
        """
        kb = KnowledgeBase(grid_size=8)
        kb.update(Position(0, 0), breeze=True, stench=False, glitter=False,
                  legal_actions=(Action.RIGHT, Action.DOWN))

        # Now visit (1,0) where there is NO breeze
        kb.update(Position(1, 0), breeze=False, stench=False, glitter=False,
                  legal_actions=(Action.UP, Action.DOWN, Action.RIGHT))

        # (1,1) neighbor of (1,0) must be SAFE (no breeze at 1,0)
        assert kb.status(Position(1, 1)) == CellStatus.SAFE

    # ===================================================================
    # T402: Constraint propagation / single candidate elimination
    # ===================================================================

    def test_single_candidate_pit_confirmation(self):
        """
        Corner (0,0) has breeze. Neighbors are (0,1) and (1,0).
        Agent visits (1,0) and finds NO breeze -> (1,0) clears (1,1) and (2,0).
        Wait, if (1,0) has NO breeze, then (1,0)'s neighbors cannot be pits.
        So (0,0) neighbor (1,0) is visited/safe. (0,0)'s ONLY remaining unvisited neighbor is (0,1).
        Thus (0,1) MUST be a pit!
        """
        kb = KnowledgeBase(grid_size=8)
        kb.update(Position(0, 0), breeze=True, stench=False, glitter=False,
                  legal_actions=(Action.RIGHT, Action.DOWN))

        # Visit (1,0), no breeze
        kb.update(Position(1, 0), breeze=False, stench=False, glitter=False,
                  legal_actions=(Action.UP, Action.DOWN, Action.RIGHT))

        # Single candidate elimination should confirm (0,1) as PIT
        assert kb.status(Position(0, 1)) == CellStatus.CONFIRMED_PIT

    def test_single_candidate_wumpus_confirmation(self):
        """Same single-candidate elimination logic for Wumpus."""
        kb = KnowledgeBase(grid_size=8)
        kb.update(Position(0, 0), stench=True, breeze=False, glitter=False,
                  legal_actions=(Action.RIGHT, Action.DOWN))

        kb.update(Position(1, 0), stench=False, breeze=False, glitter=False,
                  legal_actions=(Action.UP, Action.DOWN, Action.RIGHT))

        assert kb.status(Position(0, 1)) == CellStatus.CONFIRMED_WUMPUS


# ===================================================================
# T403 & T404 & T405: RuleAgent behavior & integration
# ===================================================================

class TestRuleAgent:

    def test_golden1_straight_clean(self):
        """On empty map with no hazards, RuleAgent explores safely to exit."""
        parsed = parse_input((FIXTURES / "golden1_straight.txt").read_text())
        agent = RuleAgent()

        result = run_episode(agent, parsed.game_map, parsed.config, seed=42)

        assert result.won
        assert result.state.status == Status.WON
        assert result.state.health > 0

    def test_golden2_pit_bypass(self):
        """On map with pit, RuleAgent senses breeze, infers caution, avoids confirmed pit."""
        parsed = parse_input((FIXTURES / "golden2_pit.txt").read_text())
        agent = RuleAgent()

        result = run_episode(agent, parsed.game_map, parsed.config, seed=42)

        assert result.won
        assert result.state.status == Status.WON
        # RuleAgent should not blindly step into pit
        assert result.state.pit_entries == 0

    def test_golden3_complex_safe_gold_collection(self):
        """RuleAgent on complex map gets gold if safe, reaches exit."""
        parsed = parse_input((FIXTURES / "golden3_complex.txt").read_text())
        agent = RuleAgent()

        result = run_episode(agent, parsed.game_map, parsed.config, seed=42)

        assert result.won
        assert result.state.status == Status.WON

    def test_reasoning_trace_populated(self):
        """T405: Agent maintains reasoning log with rules fired."""
        parsed = parse_input((FIXTURES / "golden1_straight.txt").read_text())
        agent = RuleAgent()

        run_episode(agent, parsed.game_map, parsed.config, seed=42)

        assert len(agent.reasoning_log) > 0
        # Check first step trace contains expected entries
        first_step_trace = agent.reasoning_log[0]
        assert any("NO_BREEZE" in line for line in first_step_trace)
