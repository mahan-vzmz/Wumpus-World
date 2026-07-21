"""SearchAgent: offline A* agent with full map visibility.

This agent receives the complete GameMap at reset time, plans an optimal
path using A*, and then replays the plan action-by-action via choose_action.

Per SPEC §6, SearchAgent uses FullMapAgent semantics — it explicitly receives
the hidden map reference. This difference is flagged in benchmark results.
"""

from __future__ import annotations

from typing import Any

from wumpus.agents.base import Agent
from wumpus.domain import Action, GameConfig, GameMap, Status
from wumpus.observation import Observation
from wumpus.search import SearchResult, solve_astar


class SearchAgent(Agent):
    """A* offline agent that plans the entire path before the first move."""

    def __init__(self) -> None:
        self._plan: tuple[Action, ...] = ()
        self._step_index: int = 0
        self._search_result: SearchResult | None = None

    def reset(
        self, config: GameConfig, public_map_info: dict[str, Any], seed: int
    ) -> None:
        """Plan the full path at reset time.

        Unlike online agents, SearchAgent expects public_map_info to contain
        a 'game_map' key with the full GameMap. The runner must provide this
        for the search agent specifically.
        """
        game_map: GameMap = public_map_info["game_map"]
        self._search_result = solve_astar(game_map, config)
        self._plan = self._search_result.plan
        self._step_index = 0

    def choose_action(self, observation: Observation) -> Action:
        """Return the next pre-planned action."""
        if self._step_index >= len(self._plan):
            # Plan exhausted but game still running — should not happen
            # with a correct solver, but pick first legal action as fallback.
            return observation.legal_actions[0]

        action = self._plan[self._step_index]
        self._step_index += 1
        return action

    def observe_transition(
        self, observation: Observation, action: Action, outcome: Any
    ) -> None:
        # Offline agent has no use for transition feedback
        pass

    @property
    def search_result(self) -> SearchResult | None:
        """Access diagnostics from the last planning run."""
        return self._search_result
