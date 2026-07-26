"""MLAgent: online supervised learning agent (T506).

Uses a trained ML model (Random Forest / Decision Tree) to choose actions
from observation + belief features. Masks illegal actions so it never
commits an illegal move.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wumpus.agents.base import Agent
from wumpus.ai.encoder import encode_observation
from wumpus.ai.knowledge import KnowledgeBase
from wumpus.ai.ml import load_model, predict_masked_action
from wumpus.core.domain import Action, GameConfig
from wumpus.core.observation import Observation


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
        self._prev_health: int | None = None
        self._prev_position: Any = None

    def load(self, model_path: Path) -> None:
        """Load a saved model from disk."""
        self._model = load_model(model_path)

    def reset(
        self, config: GameConfig, public_map_info: dict[str, Any], seed: int
    ) -> None:
        self._config = config
        self._kb = KnowledgeBase(grid_size=config.grid_size)
        self._prev_health = None
        self._prev_position = None

    def choose_action(self, observation: Observation) -> Action:
        assert self._config is not None

        if self._model is None:
            raise RuntimeError(
                "MLAgent has no trained model. Run 'python -m wumpus train' "
                "or pass --model PATH."
            )

        # Record the ground truth of the just-entered cell BEFORE inference
        # (mirrors the rule agent) so a just-entered pit can explain a
        # neighbouring breeze during single-candidate confirmation.
        if self._prev_health is None:
            was_pit = False
        else:
            was_pit = (
                observation.position != self._prev_position
                and observation.health < self._prev_health - 1
            )
        self._kb.observe_entry(observation.position, was_pit=was_pit)
        self._prev_health = observation.health
        self._prev_position = observation.position

        # Update KB with the new percepts
        self._kb.update(
            pos=observation.position,
            breeze=observation.breeze,
            stench=observation.stench,
            glitter=observation.glitter,
            legal_actions=observation.legal_actions,
        )

        # Encode features
        x_vec = encode_observation(observation, self._kb, self._config)

        # Prefer moves that are neither confirmed-deadly nor a known pit, then
        # relax to merely-non-deadly, then to any legal move. Unknown cells stay
        # available because partial observability sometimes needs a calculated
        # risk.
        def _dest(a: Action):
            return observation.position.moved(a)

        preferred = tuple(
            a for a in observation.legal_actions
            if not self._kb.is_dangerous(_dest(a))
            and not self._kb.is_known_pit(_dest(a))
            and not self._kb.has_wumpus_suspicion(_dest(a))
        )
        non_dangerous = tuple(
            a for a in observation.legal_actions
            if not self._kb.is_dangerous(_dest(a))
        )
        candidate_actions = preferred or non_dangerous or observation.legal_actions

        # Predict with legal and knowledge-based action masking.
        action = predict_masked_action(self._model, x_vec, candidate_actions)
        return action

    def observe_transition(
        self, observation: Observation, action: Action, outcome: Any
    ) -> None:
        pass
