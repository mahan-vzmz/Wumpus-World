import random
from typing import Any

from wumpus.agents.base import Agent
from wumpus.domain import Action, GameConfig, Position
from wumpus.observation import Observation


def manhattan_distance(p1: Position, p2: Position) -> int:
    return abs(p1.row - p2.row) + abs(p1.col - p2.col)


class GreedyExitAgent(Agent):
    """
    عاملی که به‌صورت حریصانه تلاش می‌کند فاصلهٔ منهتن خود را تا در خروج کم کند.
    اگر چند حرکت فاصله را به یک اندازه کم کنند، یکی را تصادفی انتخاب می‌کند.
    """

    def __init__(self) -> None:
        self._rng = random.Random()
        self._exit_position: Position | None = None

    def reset(
        self, config: GameConfig, public_map_info: dict[str, Any], seed: int
    ) -> None:
        self._rng.seed(seed)
        self._exit_position = config.exit_position

    def choose_action(self, observation: Observation) -> Action:
        assert self._exit_position is not None, "Agent was not reset properly."
        
        current_pos = observation.position
        best_actions = []
        min_dist = float("inf")

        for action in observation.legal_actions:
            next_pos = current_pos.moved(action)
            dist = manhattan_distance(next_pos, self._exit_position)
            
            if dist < min_dist:
                min_dist = dist
                best_actions = [action]
            elif dist == min_dist:
                best_actions.append(action)

        return self._rng.choice(best_actions)

    def observe_transition(
        self, observation: Observation, action: Action, outcome: Any
    ) -> None:
        pass
