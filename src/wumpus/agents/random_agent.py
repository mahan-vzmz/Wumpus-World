import random
from typing import Any

from wumpus.agents.base import Agent
from wumpus.domain import Action, GameConfig
from wumpus.observation import Observation


class RandomAgent(Agent):
    """
    عاملی که به‌صورت تصادفی از بین کنش‌های قانونی (Legal Actions) یکی را انتخاب می‌کند.
    """

    def __init__(self) -> None:
        self._rng = random.Random()

    def reset(
        self, config: GameConfig, public_map_info: dict[str, Any], seed: int
    ) -> None:
        self._rng.seed(seed)

    def choose_action(self, observation: Observation) -> Action:
        # انتخاب یک کنش معتبر به‌صورت کاملاً تصادفی
        return self._rng.choice(tuple(observation.legal_actions))

    def observe_transition(
        self, observation: Observation, action: Action, outcome: Any
    ) -> None:
        # عامل تصادفی نیازی به یادگیری یا حافظه ندارد
        pass
