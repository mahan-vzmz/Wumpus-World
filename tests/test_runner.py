"""Integration tests for the shared episode runner."""

from typing import Any

from wumpus.domain import Action, GameConfig, Status
from wumpus.parser import parse_input
from wumpus.runner import run_episode

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
