"""Tests for the standalone demo HTML builder (multi-agent payload)."""

import json
from pathlib import Path

import numpy as np
import pytest

from wumpus.ai.ml import MajorityBaseline, save_model
from wumpus.viz.html import (
    AGENT_ORDER,
    CURATED_EPISODES,
    build_demo_html,
    build_demo_payload,
    write_demo,
)

REPO_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    X = np.zeros((4, 397), dtype=np.float32)
    y = np.array([0, 1, 2, 3], dtype=np.int64)
    save_model(MajorityBaseline().fit(X, y), path)
    return path


@pytest.fixture(scope="module")
def payload(tiny_model_path) -> dict:
    return build_demo_payload(REPO_ROOT, seed=42, model_path=tiny_model_path)


class TestDemoPayload:

    def test_payload_covers_all_curated_episodes(self, payload):
        assert [e["id"] for e in payload["episodes"]] == [
            spec.episode_id for spec in CURATED_EPISODES
        ]
        stars = [e["stars"] for e in payload["episodes"]]
        assert stars == sorted(stars), "episodes must run easy -> hard"
        assert payload["episodes"][-1]["fatal"] is True

    def test_every_episode_embeds_every_mind(self, payload):
        assert payload["agents"] == list(AGENT_ORDER)
        for episode in payload["episodes"]:
            assert set(episode["runs"]) == set(AGENT_ORDER)
            for run in episode["runs"].values():
                assert run["frames"], "each run must have frames"
                assert run["frames"][-1]["status"] == run["result"]["status"]

    def test_search_run_is_full_visibility_with_planner(self, payload):
        for episode in payload["episodes"]:
            search = episode["runs"]["search"]
            assert search["visibility"] == "full"
            assert search["planner"]["solved"] is True
            assert search["planner"]["plan_length"] == search["result"]["steps"]
            assert search["result"]["won"] is True

    def test_belief_layers_by_agent_kind(self, payload):
        episode = payload["episodes"][0]
        assert episode["runs"]["rules"]["frames"][0]["belief"] is not None
        assert episode["runs"]["ml"]["frames"][0]["belief"] is not None
        assert episode["runs"]["greedy"]["frames"][0]["belief"] is None
        assert episode["runs"]["random"]["frames"][0]["belief"] is None

    def test_skip_ml_omits_the_ml_mind(self):
        payload = build_demo_payload(REPO_ROOT, seed=42, include_ml=False)
        assert "ml" not in payload["agents"]
        assert all("ml" not in e["runs"] for e in payload["episodes"])

    def test_missing_model_raises_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ML model not found"):
            build_demo_payload(
                REPO_ROOT, seed=42, model_path=tmp_path / "missing.joblib"
            )


class TestDemoHtml:

    def test_html_is_self_contained_and_embeds_data(self, payload):
        html = build_demo_html(payload)
        assert html.startswith("<!doctype html>")
        assert "__DATA_JSON__" not in html
        assert "const DATA" in html
        assert 'id="agentSeg"' in html  # the mind switcher exists
        for spec in CURATED_EPISODES:
            assert spec.title in html or spec.episode_id in html
        # Self-contained: the only external reference is the GitHub source link.
        stripped = html.replace('href="https://github.com/mahan-vzmz/Wumpus-World"', "")
        assert 'src="http' not in stripped and 'href="http' not in stripped

    def test_embedded_json_round_trips(self, payload):
        html = build_demo_html(payload)
        blob = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
        parsed = json.loads(blob.replace("<\\/", "</"))
        assert len(parsed["episodes"]) == len(CURATED_EPISODES)
        assert set(parsed["episodes"][0]["runs"]) == set(AGENT_ORDER)

    def test_write_demo_creates_file(self, tmp_path, tiny_model_path):
        out = tmp_path / "demo" / "index.html"
        size = write_demo(REPO_ROOT, out, seed=42, model_path=tiny_model_path)
        assert out.is_file()
        assert size == len(out.read_text(encoding="utf-8").encode("utf-8"))
        assert size > 100_000  # five minds x six maps actually embedded
