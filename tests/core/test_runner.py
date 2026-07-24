"""Integration tests for the shared episode runner."""

from typing import Any

from wumpus.agents.search_agent import SearchAgent
from wumpus.core.domain import Action, GameConfig, Status
from wumpus.core.parser import parse_input
from wumpus.core.runner import run_episode

EMPTY_MAP = """\
********
********
********
********
********
********
********
********
50
10
-15
8 8
"""

UNSOLVABLE_MAP = """\
*D******
D*******
********
********
********
********
********
********
50
10
-15
8 8
"""


class FailingAgent:
    def reset(
        self, config: GameConfig, public_map_info: dict[str, Any], seed: int
    ) -> None:
        pass

    def choose_action(self, observation: Any) -> Action:
        raise RuntimeError("deliberate agent failure")

    def observe_transition(
        self, observation: Any, action: Action, outcome: Any
    ) -> None:
        pass


def test_agent_exception_becomes_terminal_structured_result() -> None:
    parsed = parse_input(EMPTY_MAP)

    result = run_episode(FailingAgent(), parsed.game_map, parsed.config)

    assert result.state.status is Status.AGENT_ERROR
    assert result.error == "deliberate agent failure"
    assert result.state.steps == 0
    assert result.state.event_log[-1].startswith("AGENT_ERROR:")


def test_search_no_solution_becomes_explicit_terminal_result() -> None:
    parsed = parse_input(UNSOLVABLE_MAP)

    result = run_episode(SearchAgent(), parsed.game_map, parsed.config)

    assert result.state.status is Status.NO_SOLUTION
    assert result.error is None
    assert result.state.steps == 0
    assert result.state.event_log[-1].startswith("NO_SOLUTION:")
