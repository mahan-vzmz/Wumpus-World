"""MLAgent: online supervised learning agent (T506).

Uses a trained ML model (Random Forest / Decision Tree) to choose actions
from observation + belief features. Masks illegal actions so it never
commits an illegal move.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wumpus.agents.base import Agent
from wumpus.domain import Action, GameConfig
from wumpus.encoder import encode_observation
from wumpus.knowledge import KnowledgeBase
from wumpus.ml import load_model, predict_masked_action
from wumpus.observation import Observation


class MLAgent(Agent):
    """Online agent driven by a trained supervised classification model."""

    def __init__(self, model: Any = None, model_path: Path | None = None) -> None:
        if model is not None:
            self._model = model
        elif model_path is not None:
            self._model = load_model(model_path)
        else:
            self._model = None

        self._kb: KnowledgeBase = KnowledgeBase()
        self._config: GameConfig | None = None

    def load(self, model_path: Path) -> None:
        """Load a saved model from disk."""
        self._model = load_model(model_path)

    def reset(
        self, config: GameConfig, public_map_info: dict[str, Any], seed: int
    ) -> None:
        self._config = config
        self._kb = KnowledgeBase(grid_size=config.grid_size)

    def choose_action(self, observation: Observation) -> Action:
        assert self._config is not None

        # Fallback to legal action if no model loaded
        if self._model is None:
            return observation.legal_actions[0]

        # Update KB
        self._kb.update(
            pos=observation.position,
            breeze=observation.breeze,
            stench=observation.stench,
            glitter=observation.glitter,
            legal_actions=observation.legal_actions,
        )

        # Encode features
        x_vec = encode_observation(observation, self._kb, self._config)

        # Predict with legal action masking
        action = predict_masked_action(self._model, x_vec, observation.legal_actions)
        return action

    def observe_transition(
        self, observation: Observation, action: Action, outcome: Any
    ) -> None:
        pass
