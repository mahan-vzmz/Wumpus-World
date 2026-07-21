"""Tests for map generation, feature encoding, dataset building, ML models, and MLAgent (Epic 5).

Covers:
  T500 — Map generator validity and A* solvability
  T501 — Feature encoder shape and no-leakage properties
  T502 — Dataset generation from A* teacher
  T503 — Zero-leakage map-level dataset splitting
  T504 — Baseline & Decision Tree training
  T505 — Random Forest classifier training
  T506 — Legal action masking, model save/load, and MLAgent execution
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from wumpus.agents.ml_agent import MLAgent
from wumpus.dataset import DatasetConfig, generate_dataset, split_dataset
from wumpus.domain import Action, GameConfig, GameMap, Position
from wumpus.encoder import FEATURE_NAMES, encode_observation
from wumpus.generator import MapGenerationConfig, generate_map
from wumpus.knowledge import KnowledgeBase
from wumpus.ml import (
    MajorityBaseline,
    load_model,
    predict_masked_action,
    save_model,
    train_models,
)
from wumpus.observation import make_observation
from wumpus.runner import run_episode
from wumpus.search import solve_astar


class TestMapGenerator:

    def test_generate_map_solvable(self):
        """Generated maps must be valid and solvable by A*."""
        gmap, config = generate_map(seed=42)
        assert isinstance(gmap, GameMap)
        assert isinstance(config, GameConfig)
        res = solve_astar(gmap, config)
        assert res.solved


class TestEncoder:

    def test_feature_vector_shape(self):
        """Encoder output must have shape (397,) matching FEATURE_NAMES."""
        gmap, config = generate_map(seed=42)
        from wumpus.engine import init_state
        state = init_state(gmap, config)
        obs = make_observation(gmap, config, state)
        kb = KnowledgeBase(grid_size=config.grid_size)
        kb.update(obs.position, obs.breeze, obs.stench, obs.glitter, obs.legal_actions)

        vec = encode_observation(obs, kb, config)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (len(FEATURE_NAMES),)
        assert vec.dtype == np.float32


class TestDatasetAndSplit:

    def test_dataset_generation_and_zero_leakage_split(self):
        """T502 & T503: Dataset generation and zero map-id leakage check."""
        ds_config = DatasetConfig(num_maps=5, seed=123)
        data = generate_dataset(ds_config)

        assert "X" in data and "y" in data and "map_ids" in data
        assert len(data["X"]) == len(data["y"])
        assert len(data["X"]) > 0

        train, val, test = split_dataset(data, train_ratio=0.6, val_ratio=0.2, seed=42)

        # Map IDs must have ZERO intersection across splits
        train_maps = set(np.unique(train["map_ids"]))
        val_maps = set(np.unique(val["map_ids"]))
        test_maps = set(np.unique(test["map_ids"]))

        assert len(train_maps & val_maps) == 0
        assert len(train_maps & test_maps) == 0
        assert len(val_maps & test_maps) == 0


class TestMLModels:

    def test_train_all_models(self):
        """T504 & T505: Train Majority, DT, RF models on dataset."""
        ds_config = DatasetConfig(num_maps=10, seed=200)
        data = generate_dataset(ds_config)
        train, val, test = split_dataset(data, train_ratio=0.6, val_ratio=0.2, seed=42)

        res = train_models(train, val, seed=42)
        metrics = res["metrics"]

        assert "majority" in metrics
        assert "decision_tree" in metrics
        assert "random_forest" in metrics

        assert 0.0 <= metrics["random_forest"]["accuracy"] <= 1.0

    def test_legal_action_masking(self):
        """Action masking must never pick an illegal action."""
        gmap, config = generate_map(seed=42)
        ds_config = DatasetConfig(num_maps=10, seed=300)
        data = generate_dataset(ds_config)
        train, val, test = split_dataset(data)
        res = train_models(train, val)
        rf_model = res["models"]["random_forest"]

        from wumpus.engine import init_state
        state = init_state(gmap, config)
        obs = make_observation(gmap, config, state)
        kb = KnowledgeBase(grid_size=config.grid_size)

        vec = encode_observation(obs, kb, config)
        chosen_action = predict_masked_action(rf_model, vec, obs.legal_actions)

        assert chosen_action in obs.legal_actions

    def test_model_save_and_load(self):
        """T506: Test joblib serialization and MLAgent execution."""
        ds_config = DatasetConfig(num_maps=10, seed=400)
        data = generate_dataset(ds_config)
        train, val, test = split_dataset(data)
        res = train_models(train, val)
        rf_model = res["models"]["random_forest"]

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.joblib"
            save_model(rf_model, model_path)

            loaded = load_model(model_path)
            agent = MLAgent(model=loaded)

            gmap, config = generate_map(seed=42)
            run_res = run_episode(agent, gmap, config, seed=42)

            assert run_res.state.steps > 0
