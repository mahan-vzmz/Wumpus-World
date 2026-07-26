"""Tests for the episode recorder behind the interactive demo."""

import json
from pathlib import Path

from wumpus.agents.random_agent import RandomAgent
from wumpus.agents.rule_agent import RuleAgent
from wumpus.core.domain import Action
from wumpus.viz.recorder import BELIEF_CODES, record_episode_from_file

FIXTURES = Path(__file__).parent.parent / "fixtures"
HOLDOUT = Path(__file__).parent.parent.parent / "data" / "maps" / "holdout_suite"

ACTION_NAMES = {a.value for a in Action}


def _record_golden(agent=None, seed: int = 42):
    return record_episode_from_file(
        FIXTURES / "golden2_pit.txt", agent or RuleAgent(), seed=seed
    )


class TestRecorderStructure:

    def test_frame_count_is_steps_plus_one(self):
        rec = _record_golden()
        assert len(rec["frames"]) == rec["result"]["steps"] + 1

    def test_first_frame_is_the_start_state(self):
        rec = _record_golden()
        first = rec["frames"][0]
        assert first["pos"] == [0, 0]
        assert first["health"] == rec["config"]["initial_health"]
        assert first["steps"] == 0

    def test_terminal_frame_has_no_action_and_final_status(self):
        rec = _record_golden()
        last = rec["frames"][-1]
        assert last["action"] is None
        assert last["status"] == rec["result"]["status"]
        for frame in rec["frames"][:-1]:
            assert frame["action"] in ACTION_NAMES

    def test_belief_strings_are_valid(self):
        rec = _record_golden()
        for frame in rec["frames"]:
            assert frame["belief"] is not None
            assert len(frame["belief"]) == 64
            assert set(frame["belief"]) <= BELIEF_CODES

    def test_rule_agent_frames_carry_reasoning_traces(self):
        rec = _record_golden()
        assert any(frame["trace"] for frame in rec["frames"][:-1])

    def test_truth_grid_matches_map_shape(self):
        rec = _record_golden()
        assert len(rec["truth"]) == 8
        assert all(len(row) == 8 for row in rec["truth"])
        assert rec["map_name"] == "golden2_pit.txt"

    def test_record_is_json_serializable(self):
        rec = _record_golden()
        payload = json.dumps(rec)
        assert "frames" in payload

    def test_same_seed_is_deterministic(self):
        assert _record_golden(seed=7) == _record_golden(seed=7)


class TestRecorderAgentAgnostic:

    def test_agent_without_kb_yields_no_belief_layer(self):
        rec = record_episode_from_file(
            FIXTURES / "golden1_straight.txt", RandomAgent(), seed=3, agent_name="random"
        )
        assert rec["agent"] == "random"
        assert all(frame["belief"] is None for frame in rec["frames"])
        assert all(frame["trace"] == [] for frame in rec["frames"])

    def test_holdout_hard_map_records_a_loss_faithfully(self):
        rec = record_episode_from_file(
            HOLDOUT / "05_hard_complex_map_03.txt", RuleAgent(), seed=42
        )
        assert rec["result"]["won"] is False
        assert rec["frames"][-1]["status"] == rec["result"]["status"]
