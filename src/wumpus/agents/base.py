import abc
from typing import Any, Protocol

from wumpus.core.domain import Action, GameConfig
from wumpus.core.observation import Observation


class Agent(Protocol):
    """Shared interface implemented by every Wumpus World agent.

    It guarantees that all agents are driven uniformly by the runner and that
    none of them receives a direct reference to the hidden map.
    """

    @abc.abstractmethod
    def reset(self, config: GameConfig, public_map_info: dict[str, Any], seed: int) -> None:
        """Prepare the agent for a new episode.

        Args:
            config: Static game settings (health, gold value, pit penalty, ...).
            public_map_info: Public map data the agent is allowed to see
                (grid size, exit position, and — only for the search agent —
                the full map).
            seed: Random seed for reproducible agent decisions.
        """
        ...

    @abc.abstractmethod
    def choose_action(self, observation: Observation) -> Action:
        """Choose an action given the current observation.

        Args:
            observation: What the agent is allowed to see this step.

        Returns:
            The chosen legal action.
        """
        ...

    def observe_transition(self, observation: Observation, action: Action, outcome: Any) -> None:
        """Optional hook called after a move with its outcome.

        Useful for agents that learn from or record the transition.
        """
        pass
