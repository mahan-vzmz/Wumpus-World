"""Feature encoder for Machine Learning agent (SPEC §9.2).

Extracts a flat, numerical feature vector from an Observation and KnowledgeBase state.
Uses strictly observable information and accumulated agent beliefs.
No hidden map leakage!
"""

from __future__ import annotations

import numpy as np

from wumpus.domain import Action, GameConfig, Position
from wumpus.knowledge import CellStatus, KnowledgeBase
from wumpus.observation import Observation

FEATURE_VERSION = "v1"

FEATURE_NAMES: list[str] = [
    "norm_health",
    "step_ratio",
    "collected_gold",
    "rel_exit_row",
    "rel_exit_col",
    "breeze",
    "stench",
    "glitter",
    "at_exit",
    "legal_UP",
    "legal_DOWN",
    "legal_LEFT",
    "legal_RIGHT",
]

# Add grid channels for each of the 64 cells
_CHANNEL_NAMES = ["visited", "safe", "possible_pit", "possible_wumpus", "blocked", "is_current"]
for r in range(8):
    for c in range(8):
        for ch in _CHANNEL_NAMES:
            FEATURE_NAMES.append(f"cell_{r}_{c}_{ch}")


def encode_observation(
    obs: Observation,
    kb: KnowledgeBase,
    config: GameConfig,
) -> np.ndarray:
    """Encode an Observation and KnowledgeBase into a 1D float32 feature vector (length 397)."""
    features: list[float] = []

    # 1. Scalar features
    max_steps = config.max_steps if config.max_steps else 100
    features.append(obs.health / config.initial_health)
    features.append(obs.steps / max_steps)
    features.append(float(obs.collected_gold))

    # Relative exit position
    rel_row = (config.exit_position.row - obs.position.row) / 7.0
    rel_col = (config.exit_position.col - obs.position.col) / 7.0
    features.append(rel_row)
    features.append(rel_col)

    # Percepts
    features.append(1.0 if obs.breeze else 0.0)
    features.append(1.0 if obs.stench else 0.0)
    features.append(1.0 if obs.glitter else 0.0)
    features.append(1.0 if obs.at_exit else 0.0)

    # Legal actions mask
    for action in Action:
        features.append(1.0 if action in obs.legal_actions else 0.0)

    # 2. Grid channels (64 cells x 6 channels = 384 features)
    for r in range(8):
        for c in range(8):
            p = Position(r, c)
            status = kb.status(p)

            is_visited = 1.0 if kb.is_visited(p) else 0.0
            is_safe = 1.0 if kb.is_safe(p) else 0.0
            is_pos_pit = 1.0 if status == CellStatus.POSSIBLE_PIT else 0.0
            is_pos_wumpus = 1.0 if status == CellStatus.POSSIBLE_WUMPUS else 0.0
            is_blocked = 1.0 if status == CellStatus.BLOCKED else 0.0
            is_current = 1.0 if p == obs.position else 0.0

            features.extend([is_visited, is_safe, is_pos_pit, is_pos_wumpus, is_blocked, is_current])

    return np.array(features, dtype=np.float32)


def action_to_label(action: Action) -> int:
    """Convert Action enum to class label index 0..3."""
    action_order = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    return action_order.index(action)


def label_to_action(label: int) -> Action:
    """Convert class label index 0..3 back to Action enum."""
    action_order = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    return action_order[label]
