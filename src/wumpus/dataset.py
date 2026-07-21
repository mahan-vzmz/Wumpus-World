"""Dataset generator and map-level dataset splitter (T502, T503).

Features:
- Generates datasets of expert (A*) demonstrations on random maps
- Maps are split strictly by map_id to guarantee zero data leakage between train/val/test
- Saves dataset to disk as npz / json metadata
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from wumpus.domain import Action, GameConfig, GameMap, Status
from wumpus.encoder import FEATURE_NAMES, FEATURE_VERSION, action_to_label, encode_observation
from wumpus.engine import init_state, step
from wumpus.generator import MapGenerationConfig, generate_map
from wumpus.knowledge import KnowledgeBase
from wumpus.observation import make_observation
from wumpus.search import solve_astar


@dataclass
class DatasetConfig:
    num_maps: int = 50
    seed: int = 100
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    map_gen_config: MapGenerationConfig = MapGenerationConfig()


def generate_dataset(
    config: DatasetConfig = DatasetConfig(),
) -> dict[str, np.ndarray]:
    """Generate a complete dataset of feature vectors and action labels from A* expert.

    Returns dict containing:
      X: float32 array of shape (N, num_features)
      y: int64 array of shape (N,)
      legal_masks: float32 array of shape (N, 4)
      map_ids: int64 array of shape (N,) — map identifier for splitting
    """
    all_X: list[np.ndarray] = []
    all_y: list[int] = []
    all_masks: list[list[float]] = []
    all_map_ids: list[int] = []

    for map_idx in range(config.num_maps):
        map_seed = config.seed + map_idx
        try:
            game_map, game_config = generate_map(config.map_gen_config, seed=map_seed)
        except RuntimeError:
            continue

        search_res = solve_astar(game_map, game_config)
        if not search_res.solved or not search_res.plan:
            continue

        state = init_state(game_map, game_config)
        kb = KnowledgeBase(grid_size=game_config.grid_size)

        for expert_action in search_res.plan:
            obs = make_observation(game_map, game_config, state)
            kb.update(
                pos=obs.position,
                breeze=obs.breeze,
                stench=obs.stench,
                glitter=obs.glitter,
                legal_actions=obs.legal_actions,
            )

            # Feature vector
            x_vec = encode_observation(obs, kb, game_config)
            all_X.append(x_vec)

            # Label
            all_y.append(action_to_label(expert_action))

            # Legal action mask (1.0 for legal, 0.0 for illegal)
            mask = [
                1.0 if act in obs.legal_actions else 0.0
                for act in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
            ]
            all_masks.append(mask)
            all_map_ids.append(map_idx)

            # Execute step
            state = step(game_map, game_config, state, expert_action)
            if state.status != Status.RUNNING:
                break

    X_arr = np.array(all_X, dtype=np.float32)
    y_arr = np.array(all_y, dtype=np.int64)
    masks_arr = np.array(all_masks, dtype=np.float32)
    map_ids_arr = np.array(all_map_ids, dtype=np.int64)

    return {
        "X": X_arr,
        "y": y_arr,
        "legal_masks": masks_arr,
        "map_ids": map_ids_arr,
    }


def split_dataset(
    data: dict[str, np.ndarray],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split dataset by map_id so no map appears in more than one split (T503)."""
    map_ids = np.unique(data["map_ids"])
    rng = np.random.default_rng(seed)
    rng.shuffle(map_ids)

    n_maps = len(map_ids)
    if n_maps < 3:
        n_train = n_maps
        n_val = 0
    else:
        n_val = max(1, int(n_maps * val_ratio))
        n_test = max(1, int(n_maps * (1.0 - train_ratio - val_ratio)))
        n_train = n_maps - n_val - n_test

    train_maps = set(map_ids[:n_train])
    val_maps = set(map_ids[n_train:n_train + n_val])
    test_maps = set(map_ids[n_train + n_val:])

    # Verify zero leakage
    assert len(train_maps & val_maps) == 0
    assert len(train_maps & test_maps) == 0
    assert len(val_maps & test_maps) == 0

    def filter_split(target_maps: set[int]) -> dict[str, np.ndarray]:
        idx_mask = np.isin(data["map_ids"], list(target_maps))
        return {
            "X": data["X"][idx_mask],
            "y": data["y"][idx_mask],
            "legal_masks": data["legal_masks"][idx_mask],
            "map_ids": data["map_ids"][idx_mask],
        }

    return filter_split(train_maps), filter_split(val_maps), filter_split(test_maps)


def save_dataset(
    output_dir: Path,
    data: dict[str, np.ndarray],
) -> None:
    """Save dataset dict to an .npz file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "dataset.npz", **data)
    meta = {
        "feature_version": FEATURE_VERSION,
        "num_samples": len(data["y"]),
        "num_features": data["X"].shape[1],
        "feature_names": FEATURE_NAMES,
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def load_dataset(input_dir: Path) -> dict[str, np.ndarray]:
    """Load dataset from an .npz file."""
    loaded = np.load(input_dir / "dataset.npz")
    return {k: loaded[k] for k in loaded.files}
