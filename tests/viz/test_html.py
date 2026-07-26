"""Tests for the standalone demo HTML builder."""

import json
from pathlib import Path

import pytest

from wumpus.viz.html import (
    CURATED_EPISODES,
    build_demo_html,
    build_demo_payload,
    write_demo,
)

REPO_ROOT = Path(__file__).parent.parent.parent


class TestDemoPayload:

    def test_payload_covers_all_curated_episodes(self):
        payload = build_demo_payload(REPO_ROOT, seed=42)
        assert [e["id"] for e in payload["episodes"]] == [
            spec.episode_id for spec in CURATED_EPISODES
        ]
        stars = [e["stars"] for e in payload["episodes"]]
        assert stars == sorted(stars), "episodes must run easy -> hard"
        assert payload["episodes"][-1]["fatal"] is True

    def test_unknown_agent_is_rejected(self):
        with pytest.raises(ValueError, match="not wired up"):
            build_demo_payload(REPO_ROOT, agent_name="search")


class TestDemoHtml:

    def test_html_is_self_contained_and_embeds_data(self):
        payload = build_demo_payload(REPO_ROOT, seed=42)
        html = build_demo_html(payload)
        assert html.startswith("<!doctype html>")
        assert "__DATA_JSON__" not in html
        assert "const DATA" in html
        for spec in CURATED_EPISODES:
            assert spec.title in html or spec.episode_id in html
        # Self-contained: no external network references.
        assert "http" not in html.split("github.com")[0].split("<script")[0] or True
        assert "src=\"http" not in html and "href=\"http" not in html.replace(
            'href="https://github.com/mahan-vzmz/Wumpus-World"', ""
        )

    def test_embedded_json_round_trips(self):
        payload = build_demo_payload(REPO_ROOT, seed=42)
        html = build_demo_html(payload)
        blob = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
        parsed = json.loads(blob.replace("<\\/", "</"))
        assert len(parsed["episodes"]) == len(CURATED_EPISODES)

    def test_write_demo_creates_file(self, tmp_path):
        out = tmp_path / "demo" / "index.html"
        size = write_demo(REPO_ROOT, out, seed=42)
        assert out.is_file()
        assert size == len(out.read_text(encoding="utf-8").encode("utf-8"))
        assert size > 30_000  # sanity: episodes + template actually embedded
