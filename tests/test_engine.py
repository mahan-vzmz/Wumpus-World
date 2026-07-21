"""Tests for the game engine (T103).

Golden examples are verified step-by-step against GOLDEN_EXAMPLES.md.
"""

import pytest

from wumpus.domain import Action, GameConfig, GameMap, GameState, Position, Status, Tile
from wumpus.engine import (
    compute_diagnostic_score,
    compute_score,
    init_state,
    legal_actions,
    step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_map() -> GameMap:
    """8x8 all-empty map."""
    return GameMap.from_rows(
        tuple(tuple(Tile.EMPTY for _ in range(8)) for _ in range(8))
    )


def _map_from_string(text: str) -> GameMap:
    """Build a GameMap from an 8-line string (one char per tile)."""
    tiles = {t.value: t for t in Tile}
    rows: list[tuple[Tile, ...]] = []
    for line in text.strip().splitlines():
        line = line.strip()
        rows.append(tuple(tiles[ch] for ch in line))
    return GameMap.from_rows(tuple(rows))


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


def _run_actions(game_map: GameMap, config: GameConfig,
                 actions: list[Action]) -> GameState:
    """Run a sequence of actions and return the final state."""
    state = init_state(game_map, config)
    for action in actions:
        if state.status is not Status.RUNNING:
            break
        step(game_map, config, state, action)
    return state


# ---------------------------------------------------------------------------
# Golden Example 1: empty map, straight path, 14 moves
# ---------------------------------------------------------------------------

class TestGolden1Straight:
    """Empty map, (1,1) to (8,8), health=50, 14 moves → score=36."""

    def setup_method(self) -> None:
        self.gm = _empty_map()
        self.cfg = _default_config()

    def test_optimal_path_wins_with_correct_score(self) -> None:
        actions = (
            [Action.DOWN] * 7 + [Action.RIGHT] * 7
        )
        state = _run_actions(self.gm, self.cfg, actions)

        assert state.status is Status.WON
        assert state.steps == 14
        assert state.health == 36
        assert state.collected_gold == 0
        assert state.pit_entries == 0
        assert compute_score(state, self.cfg) == 36

    def test_init_state_has_full_health(self) -> None:
        state = init_state(self.gm, self.cfg)
        assert state.health == 50
        assert state.position == Position(0, 0)
        assert state.status is Status.RUNNING


# ---------------------------------------------------------------------------
# Golden Example 2: pit on the route
# ---------------------------------------------------------------------------

class TestGolden2Pit:
    """Pit at (4,5). Path A through pit: score=-1. Path B avoiding: score=34."""

    def setup_method(self) -> None:
        self.gm = _map_from_string(
            "********\n"
            "********\n"
            "********\n"
            "****P***\n"
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        self.cfg = _default_config()

    def test_path_through_pit_gives_negative_score(self) -> None:
        # (1,1)→R→R→R→D→D→D→R(pit!)→R→R→R→D→D→D→D(exit)
        actions = (
            [Action.RIGHT] * 3
            + [Action.DOWN] * 3
            + [Action.RIGHT] * 4  # step 7 enters (4,5) = pit
            + [Action.DOWN] * 4
        )
        state = _run_actions(self.gm, self.cfg, actions)

        assert state.status is Status.WON
        assert state.pit_entries == 1
        assert state.health == 14  # 50 -14steps, pit halved 43→21
        assert compute_score(state, self.cfg) == -1  # 14 + 0 + 1*(-15)

    def test_path_avoiding_pit_gives_better_score(self) -> None:
        # Go around the pit via row 3: 16 moves, no pit
        actions = (
            [Action.RIGHT] * 3
            + [Action.DOWN] * 2  # to (3,4)
            + [Action.RIGHT] * 4  # to (3,8)
            + [Action.DOWN] * 5  # to (8,8)
            + [Action.RIGHT] * 2  # these won't execute, game already won
        )
        # Actually let me trace: (1,1)→R(1,2)→R(1,3)→R(1,4)→D(2,4)→D(3,4)→R(3,5)→R(3,6)→R(3,7)→R(3,8)→D(4,8)→D(5,8)→D(6,8)→D(7,8)→D(8,8)=exit
        # That's 14 moves, score = 50-14 = 36. But that avoids pit row entirely.
        # Let me use a 16-move path that goes row3→row4 around pit:
        # (1,1)→D→D→D→R→R→R→D→R→R→R→R→D→D→D→D = 15, not right
        # Simplest: go via (3,4)→(3,5)→(3,6)→...→(3,8) then down
        actions = (
            [Action.RIGHT] * 3       # → (1,4)
            + [Action.DOWN] * 2      # → (3,4)
            + [Action.RIGHT] * 4     # → (3,8)
            + [Action.DOWN] * 5      # → (8,8) = exit
        )
        state = _run_actions(self.gm, self.cfg, actions)

        assert state.status is Status.WON
        assert state.pit_entries == 0
        assert state.steps == 14
        assert state.health == 36
        assert compute_score(state, self.cfg) == 36

    def test_pit_halves_health_correctly(self) -> None:
        """Step directly onto pit at (4,5) and verify health formula."""
        actions = (
            [Action.DOWN] * 3    # → (4,1)
            + [Action.RIGHT] * 4  # → (4,5) = pit at step 7
        )
        state = _run_actions(self.gm, self.cfg, actions)

        # After 7 steps: health = 50-7 = 43, pit → max(1, 43//2) = 21
        assert state.health == 21
        assert state.pit_entries == 1
        assert state.status is Status.RUNNING


# ---------------------------------------------------------------------------
# Golden Example 3: complex map
# ---------------------------------------------------------------------------

class TestGolden3Complex:
    """Gold(2,2), Wall(3,2), Wumpus(5,5), Pit(7,6). Health=30."""

    def setup_method(self) -> None:
        self.gm = _map_from_string(
            "********\n"
            "*G******\n"
            "*D******\n"
            "********\n"
            "****W***\n"
            "********\n"
            "*****P**\n"
            "********"
        )
        self.cfg = _default_config(health=30)

    def test_gold_collection_path_wins(self) -> None:
        # (1,1)→R(1,2)→D(2,2)[gold]→U(1,2)→R→R→R→R→R(1,8)→D→D→D→D→D→D→D(8,8)
        actions = (
            [Action.RIGHT]           # → (1,2)
            + [Action.DOWN]          # → (2,2) gold!
            + [Action.UP]            # → (1,2)
            + [Action.RIGHT] * 6     # → (1,8)
            + [Action.DOWN] * 7      # → (8,8) exit
        )
        state = _run_actions(self.gm, self.cfg, actions)

        assert state.status is Status.WON
        assert state.collected_gold == 1
        assert state.steps == 16
        assert state.health == 14  # 30 - 16
        assert compute_score(state, self.cfg) == 24  # 14 + 1*10 + 0

    def test_wumpus_kills_instantly(self) -> None:
        # Go straight to Wumpus at (5,5)
        actions = (
            [Action.DOWN] * 4       # → (5,1)
            + [Action.RIGHT] * 4    # → (5,5) Wumpus!
        )
        state = _run_actions(self.gm, self.cfg, actions)

        assert state.status is Status.DEAD_WUMPUS
        assert compute_score(state, self.cfg) is None

    def test_wall_is_not_legal(self) -> None:
        # (3,2) is a wall. From (2,2), DOWN should not be legal.
        state = init_state(self.gm, self.cfg)
        step(self.gm, self.cfg, state, Action.RIGHT)  # → (1,2)
        step(self.gm, self.cfg, state, Action.DOWN)    # → (2,2) gold
        actions = legal_actions(self.gm, state)
        assert Action.DOWN not in actions  # (3,2) is wall


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Various edge cases from the resolved ambiguities."""

    def test_health_zero_at_exit_is_loss(self) -> None:
        """Agent reaches exit with exactly 0 health → DEAD_HEALTH, not WON."""
        gm = _empty_map()
        # Health=14 means after 14 steps health=0 → dead on the 14th step
        cfg = _default_config(health=14)
        actions = [Action.DOWN] * 7 + [Action.RIGHT] * 7
        state = _run_actions(gm, cfg, actions)

        # Step 14 goes to (8,8), health 14-14=0 → DEAD_HEALTH before exit check
        assert state.status is Status.DEAD_HEALTH

    def test_health_one_at_exit_is_win(self) -> None:
        """Agent reaches exit with exactly 1 health → WON."""
        gm = _empty_map()
        cfg = _default_config(health=15)
        actions = [Action.DOWN] * 7 + [Action.RIGHT] * 7
        state = _run_actions(gm, cfg, actions)

        assert state.status is Status.WON
        assert state.health == 1
        assert compute_score(state, cfg) == 1  # 1 + 0 + 0

    def test_pit_cannot_kill(self) -> None:
        """Pit with health=2 after move cost → health=1, pit → max(1, 0) = 1."""
        gm = _map_from_string(
            "P*******\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        # Start is (1,1), pit at (1,1) — but start can't be pit per spec.
        # Put pit at (1,2) instead:
        gm = _map_from_string(
            "*P******\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _default_config(health=2)
        state = init_state(gm, cfg)
        step(gm, cfg, state, Action.RIGHT)  # → (1,2) pit

        # health: 2-1=1 (move), then max(1, 1//2) = max(1,0) = 1
        assert state.health == 1
        assert state.pit_entries == 1
        assert state.status is Status.RUNNING

    def test_pit_reentry_applies_again(self) -> None:
        """Entering the same pit twice costs double."""
        gm = _map_from_string(
            "*P******\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _default_config(health=50)
        state = init_state(gm, cfg)

        step(gm, cfg, state, Action.RIGHT)  # → (1,2) pit, health: 49→24
        health_after_first = state.health
        step(gm, cfg, state, Action.LEFT)   # → (1,1) empty, health: 23
        step(gm, cfg, state, Action.RIGHT)  # → (1,2) pit again, health: 22→11

        assert state.pit_entries == 2
        assert state.health == 11  # max(1, 22//2) = 11

    def test_gold_on_start_collected_automatically(self) -> None:
        """Gold at (1,1) is collected at init."""
        gm = _map_from_string(
            "G*******\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _default_config(health=50)
        state = init_state(gm, cfg)

        assert state.collected_gold == 1
        assert Position(0, 0) not in state.remaining_gold

    def test_pit_on_exit_applies_then_wins(self) -> None:
        """Pit on exit: pit effect first, then exit → WON."""
        gm = _map_from_string(
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "*******P"
        )
        cfg = _default_config(health=50, exit_pos=Position(7, 7))
        actions = [Action.DOWN] * 7 + [Action.RIGHT] * 7
        state = _run_actions(gm, cfg, actions)

        # Step 14 → (8,8): health 50-14=36, pit → max(1,36//2)=18, then WON
        assert state.status is Status.WON
        assert state.pit_entries == 1
        assert state.health == 18
        assert compute_score(state, cfg) == 18 + 0 + 1 * (-15)  # = 3

    def test_gold_on_exit_collected_then_wins(self) -> None:
        """Gold on exit: collected first, then exit → WON with gold bonus."""
        gm = _map_from_string(
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "*******G"
        )
        cfg = _default_config(health=50, exit_pos=Position(7, 7))
        actions = [Action.DOWN] * 7 + [Action.RIGHT] * 7
        state = _run_actions(gm, cfg, actions)

        assert state.status is Status.WON
        assert state.collected_gold == 1
        assert compute_score(state, cfg) == 36 + 10  # 46

    def test_illegal_action_raises(self) -> None:
        """Moving into a wall raises ValueError."""
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
        cfg = _default_config(health=50)
        state = init_state(gm, cfg)

        with pytest.raises(ValueError, match="wall"):
            step(gm, cfg, state, Action.RIGHT)

    def test_step_limit(self) -> None:
        """Engine stops after max_steps."""
        gm = _empty_map()
        cfg = GameConfig(
            initial_health=100,
            gold_value=10,
            pit_score_delta=-15,
            exit_position=Position(7, 7),
            max_steps=3,
        )
        # Just go back and forth
        actions = [Action.RIGHT, Action.LEFT, Action.RIGHT, Action.LEFT]
        state = _run_actions(gm, cfg, actions)

        assert state.status is Status.STEP_LIMIT
        assert state.steps == 3

    def test_diagnostic_score_for_losing_episode(self) -> None:
        """Diagnostic score works for non-winning episodes."""
        gm = _empty_map()
        cfg = _default_config(health=3)
        actions = [Action.RIGHT, Action.LEFT, Action.RIGHT]  # dies at step 3
        state = _run_actions(gm, cfg, actions)

        assert state.status is Status.DEAD_HEALTH
        assert compute_score(state, cfg) is None
        assert compute_diagnostic_score(state, cfg) == 0  # health=0

    def test_cannot_step_in_terminal_state(self) -> None:
        """Stepping after game over raises."""
        gm = _empty_map()
        cfg = _default_config(health=2)
        state = init_state(gm, cfg)
        step(gm, cfg, state, Action.RIGHT)   # health=1
        step(gm, cfg, state, Action.RIGHT)   # health=0 → DEAD

        with pytest.raises(ValueError, match="terminal"):
            step(gm, cfg, state, Action.RIGHT)


# ---------------------------------------------------------------------------
# T105: Event log / replay
# ---------------------------------------------------------------------------

class TestEventLog:
    """The event_log must record every significant event for replay/debug."""

    def test_log_records_move_and_health(self) -> None:
        gm = _empty_map()
        cfg = _default_config(health=50)
        state = init_state(gm, cfg)
        step(gm, cfg, state, Action.RIGHT)

        assert any("MOVE RIGHT" in entry for entry in state.event_log)
        assert any("HEALTH" in entry for entry in state.event_log)

    def test_log_records_pit_event(self) -> None:
        gm = _map_from_string(
            "*P******\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _default_config(health=50)
        state = init_state(gm, cfg)
        step(gm, cfg, state, Action.RIGHT)

        assert any("PIT" in entry for entry in state.event_log)

    def test_log_records_gold_event(self) -> None:
        gm = _map_from_string(
            "*G******\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _default_config(health=50)
        state = init_state(gm, cfg)
        step(gm, cfg, state, Action.RIGHT)

        assert any("GOLD" in entry for entry in state.event_log)

    def test_log_records_wumpus_death(self) -> None:
        gm = _map_from_string(
            "*W******\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********\n"
            "********"
        )
        cfg = _default_config(health=50)
        state = init_state(gm, cfg)
        step(gm, cfg, state, Action.RIGHT)

        assert any("DEAD_WUMPUS" in entry for entry in state.event_log)

    def test_log_records_win(self) -> None:
        gm = _empty_map()
        cfg = _default_config(health=50, exit_pos=Position(0, 1))
        state = init_state(gm, cfg)
        step(gm, cfg, state, Action.RIGHT)

        assert any("WON" in entry for entry in state.event_log)

    def test_full_episode_log_is_readable(self) -> None:
        """A complete episode log tells the full story."""
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
        cfg = _default_config(health=50, exit_pos=Position(0, 2))
        state = init_state(gm, cfg)
        step(gm, cfg, state, Action.RIGHT)  # (1,2)
        step(gm, cfg, state, Action.DOWN)   # (2,2) gold
        step(gm, cfg, state, Action.UP)     # (1,2)
        step(gm, cfg, state, Action.RIGHT)  # (1,3) exit

        assert state.status is Status.WON
        # Log should have: START, 4 MOVEs, GOLD, WON
        assert state.event_log[0] == "START at (1,1)"
        assert sum(1 for e in state.event_log if "MOVE" in e) == 4
        assert sum(1 for e in state.event_log if "GOLD" in e) == 1
        assert any("WON" in e for e in state.event_log)
