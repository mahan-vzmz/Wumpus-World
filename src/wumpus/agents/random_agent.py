import random
from typing import Any

from wumpus.agents.base import Agent
from wumpus.core.domain import Action, GameConfig
from wumpus.core.observation import Observation


class RandomAgent(Agent):
    """Baseline agent that picks uniformly at random among the legal actions."""

    def __init__(self) -> None:
        self._rng = random.Random()

    def reset(
        self, config: GameConfig, public_map_info: dict[str, Any], seed: int
    ) -> None:
        self._rng.seed(seed)

    def choose_action(self, observation: Observation) -> Action:
        # Pick a valid action completely at random.
        return self._rng.choice(tuple(observation.legal_actions))

    def observe_transition(
        self, observation: Observation, action: Action, outcome: Any
    ) -> None:
        # The random agent has no memory or learning.
        pass
