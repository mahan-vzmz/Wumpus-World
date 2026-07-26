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

from wumpus.agents.rule_agent import RuleAgent
from wumpus.ai.knowledge import CellPercept, CellStatus, KnowledgeBase
from wumpus.core.domain import Action, GameConfig, Position, Status
from wumpus.core.observation import Observation
from wumpus.core.parser import parse_input
from wumpus.core.runner import run_episode

FIXTURES = Path(__file__).parent.parent / "fixtures"


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

    def test_breeze_and_stench_track_independent_hazards(self):
        """A cell can be implicated by both percept types simultaneously."""
        kb = KnowledgeBase(grid_size=8)
        kb.update(
            Position(0, 0),
            breeze=True,
            stench=True,
            glitter=False,
            legal_actions=(Action.RIGHT, Action.DOWN),
        )

        for cell in (Position(0, 1), Position(1, 0)):
            assert kb.has_possible_pit(cell)
            assert kb.has_possible_wumpus(cell)
            assert not kb.is_safe(cell)
            assert not kb.is_dangerous(cell)

    def test_visited_cell_remains_safe_after_old_suspicion(self):
        """Visiting a previously suspected cell clears its actionable risk."""
        kb = KnowledgeBase(grid_size=8)
        kb.update(
            Position(0, 0),
            breeze=True,
            stench=False,
            glitter=False,
            legal_actions=(Action.RIGHT, Action.DOWN),
        )
        kb.update(
            Position(1, 0),
            breeze=True,
            stench=False,
            glitter=False,
            legal_actions=(Action.UP, Action.DOWN, Action.RIGHT),
        )

        assert kb.is_visited(Position(1, 0))
        assert kb.is_safe(Position(1, 0))

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

    def test_reusing_agent_does_not_leak_state_between_episodes(self):
        """Runner reset makes a reused RuleAgent behave like a fresh instance."""
        first_map = parse_input((FIXTURES / "golden2_pit.txt").read_text())
        second_map = parse_input((FIXTURES / "golden1_straight.txt").read_text())
        reused_agent = RuleAgent()

        first_result = run_episode(
            reused_agent,
            first_map.game_map,
            first_map.config,
            seed=42,
        )
        first_kb = reused_agent._kb
        first_reasoning_log = reused_agent.reasoning_log

        reused_result = run_episode(
            reused_agent,
            second_map.game_map,
            second_map.config,
            seed=42,
        )
        fresh_agent = RuleAgent()
        fresh_result = run_episode(
            fresh_agent,
            second_map.game_map,
            second_map.config,
            seed=42,
        )

        assert first_result.won
        assert first_reasoning_log
        assert reused_agent._kb is not first_kb
        assert reused_agent.reasoning_log is not first_reasoning_log
        assert reused_agent.reasoning_log == fresh_agent.reasoning_log
        assert reused_result.state == fresh_result.state


# ===================================================================
# Soundness fixes: persistent negative evidence, multi-hazard reasoning,
# pit memory, and real risk-ranked exploration.
# ===================================================================

class TestKnowledgeBaseSoundness:

    def test_no_breeze_clears_pit_suspicion_from_any_source(self):
        """A no-breeze percept proves a neighbour is not a pit even when a
        *different* cell's breeze first made it a candidate (persistent fact)."""
        kb = KnowledgeBase(grid_size=8)
        # Breeze at (0,1) implicates (1,1) as a possible pit.
        kb.update(Position(0, 1), breeze=True, stench=False, glitter=False,
                  legal_actions=(Action.LEFT, Action.RIGHT, Action.DOWN))
        assert kb.has_possible_pit(Position(1, 1))
        # No breeze at (1,2), which is also adjacent to (1,1), proves it safe.
        kb.update(Position(1, 2), breeze=False, stench=False, glitter=False,
                  legal_actions=(Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT))
        assert not kb.has_possible_pit(Position(1, 1))
        assert not kb.has_pit_suspicion(Position(1, 1))
        assert kb.is_safe(Position(1, 1))

    def test_no_stench_clears_wumpus_suspicion_from_any_source(self):
        """Symmetric persistent negative fact for Wumpus suspicion."""
        kb = KnowledgeBase(grid_size=8)
        kb.update(Position(0, 1), breeze=False, stench=True, glitter=False,
                  legal_actions=(Action.LEFT, Action.RIGHT, Action.DOWN))
        assert kb.has_possible_wumpus(Position(1, 1))
        kb.update(Position(1, 2), breeze=False, stench=False, glitter=False,
                  legal_actions=(Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT))
        assert not kb.has_possible_wumpus(Position(1, 1))
        assert kb.is_safe(Position(1, 1))

    def test_negative_fact_blocks_future_resuspicion(self):
        """Once proven not-a-pit, a later breeze must not re-suspect the cell."""
        kb = KnowledgeBase(grid_size=8)
        # (1,1) proven safe via no-breeze at (1,2).
        kb.update(Position(1, 2), breeze=False, stench=False, glitter=False,
                  legal_actions=(Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT))
        assert not kb.has_pit_suspicion(Position(1, 1))
        # A later breeze at (2,1) (also adjacent to (1,1)) must not re-suspect it.
        kb.update(Position(2, 1), breeze=True, stench=False, glitter=False,
                  legal_actions=(Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT))
        assert not kb.has_possible_pit(Position(1, 1))

    def test_confirmed_wumpus_does_not_clear_a_second_candidate(self):
        """Multi-hazard soundness: confirming one Wumpus must not mark another
        real Wumpus safe just because it shares a stench source."""
        w1, w2 = Position(0, 1), Position(2, 1)
        shared, solo, ruled_out = Position(1, 1), Position(0, 0), Position(1, 0)
        kb = KnowledgeBase(grid_size=8)
        for src in (shared, solo):
            kb._visited.add(src)
            kb._percepts[src] = CellPercept(breeze=False, stench=True, glitter=False)
        kb._visited.add(ruled_out)
        kb._percepts[ruled_out] = CellPercept(breeze=False, stench=False, glitter=False)
        # `shared` sees both Wumpuses; `solo` sees only w1 (ruled_out cleared).
        kb._wumpus_sources[w1].update({shared, solo})
        kb._wumpus_sources[w2].update({shared})

        kb._propagate()

        assert kb.has_confirmed_wumpus(w1)
        assert not kb.is_safe(w2), "a real Wumpus was incorrectly cleared to SAFE"
        assert kb.has_possible_wumpus(w2)

    def test_confirmed_pit_does_not_clear_a_second_candidate(self):
        """Same multi-hazard soundness for pits (which are always plural)."""
        p1, p2 = Position(0, 1), Position(2, 1)
        shared, solo, ruled_out = Position(1, 1), Position(0, 0), Position(1, 0)
        kb = KnowledgeBase(grid_size=8)
        for src in (shared, solo):
            kb._visited.add(src)
            kb._percepts[src] = CellPercept(breeze=True, stench=False, glitter=False)
        kb._visited.add(ruled_out)
        kb._percepts[ruled_out] = CellPercept(breeze=False, stench=False, glitter=False)
        kb._pit_sources[p1].update({shared, solo})
        kb._pit_sources[p2].update({shared})

        kb._propagate()

        assert kb.has_confirmed_pit(p1)
        assert not kb.is_safe(p2), "a real pit was incorrectly cleared to SAFE"
        assert kb.has_possible_pit(p2)

    def test_single_candidate_confirmation_still_works(self):
        """The sound single-candidate elimination survives the clearing removal."""
        kb = KnowledgeBase(grid_size=8)
        kb.update(Position(0, 0), breeze=True, stench=False, glitter=False,
                  legal_actions=(Action.RIGHT, Action.DOWN))
        kb.update(Position(1, 0), breeze=False, stench=False, glitter=False,
                  legal_actions=(Action.UP, Action.DOWN, Action.RIGHT))
        assert kb.status(Position(0, 1)) == CellStatus.CONFIRMED_PIT

    def test_visited_pit_does_not_cause_false_pit_confirmation(self):
        """A breeze explained by a real pit the agent has already entered must
        NOT confirm an innocent sibling cell as a pit (regression: a visited
        pit is neither a candidate nor previously counted as an explanation)."""
        kb = KnowledgeBase(grid_size=8)
        # Visit safe (0,0): rules its neighbours out as pit candidates.
        kb.observe_entry(Position(0, 0), was_pit=False)
        kb.update(Position(0, 0), breeze=False, stench=False, glitter=False,
                  legal_actions=(Action.RIGHT, Action.DOWN))
        # Step onto the real pit (2,0) and take damage.
        kb.observe_entry(Position(2, 0), was_pit=True)
        kb.update(Position(2, 0), breeze=False, stench=False, glitter=False,
                  legal_actions=(Action.UP, Action.DOWN, Action.RIGHT))
        # Perceive a breeze at (1,0): its only unvisited neighbour is (1,1),
        # but the breeze is fully explained by the known pit (2,0).
        kb.observe_entry(Position(1, 0), was_pit=False)
        kb.update(Position(1, 0), breeze=True, stench=False, glitter=False,
                  legal_actions=(Action.UP, Action.DOWN, Action.RIGHT))
        assert kb.is_known_pit(Position(2, 0))
        assert not kb.has_confirmed_pit(Position(1, 1))
        assert not kb.is_dangerous(Position(1, 1))

    def test_confirmed_pit_is_crossable_not_blocked(self):
        """A confirmed pit is dangerous for normal routing but not a wall, so a
        desperate retreat can still cross it (only Wumpuses and walls cannot)."""
        kb = KnowledgeBase(grid_size=8)
        kb._confirmed_pits.add(Position(2, 2))
        kb._status[Position(2, 2)] = CellStatus.CONFIRMED_PIT
        assert kb.is_dangerous(Position(2, 2))
        assert not kb.is_blocked(Position(2, 2))


class TestKnowledgeBasePitMemory:

    def test_known_pit_excluded_from_safe_routing_but_traversable(self):
        kb = KnowledgeBase(grid_size=8)
        p = Position(3, 3)
        kb.update(p, breeze=False, stench=False, glitter=False,
                  legal_actions=(Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT))
        assert p in kb.safe_and_visited_cells()  # visited -> normally routable
        kb.mark_known_pit(p)
        assert kb.is_known_pit(p)
        assert p not in kb.safe_and_visited_cells()  # excluded from safe routing
        assert not kb.is_dangerous(p)  # still traversable in emergencies


class TestRiskRankedFrontier:

    def test_frontier_candidates_include_suspected_cells(self):
        kb = KnowledgeBase(grid_size=8)
        kb.update(Position(0, 0), breeze=True, stench=False, glitter=False,
                  legal_actions=(Action.RIGHT, Action.DOWN))
        candidates = kb.frontier_candidates()
        assert Position(0, 1) in candidates
        assert Position(1, 0) in candidates
        assert kb.risk_score(Position(0, 1)) > 0.0

    def test_risk_score_ranks_wumpus_above_pit(self):
        kb = KnowledgeBase(grid_size=8)
        kb.update(Position(0, 0), breeze=True, stench=False, glitter=False,
                  legal_actions=(Action.RIGHT, Action.DOWN))
        kb.update(Position(7, 7), breeze=False, stench=True, glitter=False,
                  legal_actions=(Action.UP, Action.LEFT))
        assert kb.risk_score(Position(6, 7)) > kb.risk_score(Position(0, 1))

    def test_confirmed_hazards_excluded_from_frontier_candidates(self):
        kb = KnowledgeBase(grid_size=8)
        confirmed = Position(2, 2)
        kb._confirmed_wumpuses.add(confirmed)
        kb._status[confirmed] = CellStatus.CONFIRMED_WUMPUS
        kb._visited.add(Position(2, 1))
        assert confirmed not in kb.frontier_candidates()


class TestRuleAgentSafety:

    def _cornered_kb(self, agent: RuleAgent) -> None:
        kb = agent._kb
        here = Position(3, 3)
        kb._visited.add(here)
        kb._status[here] = CellStatus.SAFE
        kb._confirmed_wumpuses.add(Position(2, 3))
        kb._status[Position(2, 3)] = CellStatus.CONFIRMED_WUMPUS
        kb._confirmed_pits.add(Position(4, 3))
        kb._status[Position(4, 3)] = CellStatus.CONFIRMED_PIT
        kb._confirmed_pits.add(Position(3, 4))
        kb._status[Position(3, 4)] = CellStatus.CONFIRMED_PIT
        kb._visited.add(Position(3, 2))
        kb._status[Position(3, 2)] = CellStatus.SAFE

    def test_agent_never_steps_into_confirmed_hazard_when_safe_move_exists(self):
        agent = RuleAgent()
        agent.reset(_default_config(), {"grid_size": 8, "exit_position": Position(7, 7)}, seed=3)
        self._cornered_kb(agent)
        here = Position(3, 3)
        obs = Observation(
            position=here, health=40, collected_gold=0, steps=5, status=Status.RUNNING,
            breeze=True, stench=True, glitter=False, at_exit=False,
            legal_actions=(Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT),
        )
        action = agent.choose_action(obs)
        dest = here.moved(action)
        assert action in obs.legal_actions
        assert not agent._kb.is_dangerous(dest)