"""Tests for benchmark runner and suite generator (Epic 6).

Covers:
  T600 — Test map suite generation across 5 categories
  T601 — Batch runner execution & raw CSV results
  T602 — Test reproducibility (identical seed gives identical results)
  T603 — Statistical summary table computation
"""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wumpus.ai.ml import MajorityBaseline, load_model, save_model
from wumpus.evaluation.benchmark import (
    generate_summary_table,
    run_benchmark_suite,
)
from wumpus.evaluation.suite_generator import generate_map_suite


def _write_test_model(path: Path) -> Path:
    X = np.zeros((4, 397), dtype=np.float32)
    y = np.array([0, 1, 2, 3], dtype=np.int64)
    save_model(MajorityBaseline().fit(X, y), path)
    return path


class TestBenchmarkSuite:

    def test_generate_map_suite(self):
        """T600: Suite generator creates 20 maps across 5 categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            files = generate_map_suite(out_path, base_seed=700)

            assert len(files) == 20
            assert (out_path / "suite_manifest.json").is_file()
            categories = {f.name.split("_map_")[0] for f in files}
            assert len(categories) == 5

    def test_run_benchmark_suite_and_reproducibility(self):
        """T601 & T602: Batch runner creates CSV and seed reproducibility holds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            maps_p = tmp_p / "maps"
            res_p = tmp_p / "results"
            model_p = _write_test_model(tmp_p / "model.joblib")

            generate_map_suite(maps_p, base_seed=800)
            rows1 = run_benchmark_suite(maps_p, res_p, seed=42, model_path=model_p)

            assert len(rows1) == 20 * 5  # 20 maps x 5 agents
            assert (res_p / "benchmark_results.csv").is_file()
            assert (res_p / "benchmark_results.json").is_file()
            assert (res_p / "benchmark_summary.json").is_file()

            # T602 Reproducibility check
            rows2 = run_benchmark_suite(maps_p, res_p, seed=42, model_path=model_p)
            for r1, r2 in zip(rows1, rows2):
                assert r1.status == r2.status
                assert r1.final_score == r2.final_score
                assert r1.steps_taken == r2.steps_taken

    def test_generate_summary_table(self):
        """T603: Summary table computes accurate aggregate statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            maps_p = tmp_p / "maps"
            res_p = tmp_p / "results"
            model_p = _write_test_model(tmp_p / "model.joblib")

            generate_map_suite(maps_p, base_seed=900)
            rows = run_benchmark_suite(maps_p, res_p, seed=42, model_path=model_p)

            summary = generate_summary_table(rows)
            assert "search" in summary
            assert "rules" in summary
            assert "ml" in summary
            assert "greedy" in summary
            assert "random" in summary

            # SearchAgent with full visibility should have 100% win rate on solvable maps
            assert summary["search"]["win_rate_pct"] == 100.0

    def test_benchmark_requires_explicit_ml_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            maps_p = tmp_p / "maps"
            generate_map_suite(maps_p, base_seed=950)

            with pytest.raises(FileNotFoundError, match="ML model not found"):
                run_benchmark_suite(
                    maps_p,
                    tmp_p / "results",
                    model_path=tmp_p / "missing.joblib",
                )

    def test_malformed_map_is_skipped_not_fatal(self):
        """A bad input file is recorded and skipped; the run still completes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            maps_p = tmp_p / "maps"
            res_p = tmp_p / "results"
            model_p = _write_test_model(tmp_p / "model.joblib")
            generate_map_suite(maps_p, base_seed=1234)
            # Sorts after the valid maps, so real rows are written first.
            (maps_p / "zz_broken_map_99.txt").write_text("not a valid map\n", encoding="utf-8")

            rows = run_benchmark_suite(maps_p, res_p, seed=42, model_path=model_p)

            assert len(rows) == 20 * 5  # the 20 valid maps still ran
            assert (res_p / "benchmark_results.csv").is_file()
            experiment = json.loads(
                (res_p / "benchmark_summary.json").read_text(encoding="utf-8")
            )["experiment"]
            assert any(s["map"] == "zz_broken_map_99.txt" for s in experiment["skipped_maps"])
            assert experiment["maps_evaluated"] == 20

    def test_multi_seed_row_count_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            maps_p = tmp_p / "maps"
            res_p = tmp_p / "results"
            model_p = _write_test_model(tmp_p / "model.joblib")
            generate_map_suite(maps_p, base_seed=1500)

            rows1 = run_benchmark_suite(maps_p, res_p, model_path=model_p, seeds=[1, 2])
            assert len(rows1) == 20 * 5 * 2
            assert {r.seed for r in rows1} == {1, 2}

            rows2 = run_benchmark_suite(maps_p, res_p, model_path=model_p, seeds=[1, 2])
            for r1, r2 in zip(rows1, rows2):
                assert (r1.agent, r1.seed, r1.status, r1.steps_taken) == (
                    r2.agent, r2.seed, r2.status, r2.steps_taken
                )

    def test_summary_includes_confidence_intervals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            maps_p = tmp_p / "maps"
            res_p = tmp_p / "results"
            model_p = _write_test_model(tmp_p / "model.joblib")
            generate_map_suite(maps_p, base_seed=1600)

            rows = run_benchmark_suite(maps_p, res_p, model_path=model_p, seeds=[1, 2])
            summary = generate_summary_table(rows)
            for stats in summary.values():
                lo, hi = stats["win_rate_ci"]
                assert 0.0 <= lo <= hi <= 100.0
                assert "mean_score_ci" in stats
                assert "median_score" in stats
            # Search is deterministic -> a degenerate CI at 100%.
            assert summary["search"]["win_rate_pct"] == 100.0
            assert summary["search"]["win_rate_ci"] == [100.0, 100.0]

    def test_load_model_verifies_sha256(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "m.joblib"
            model = MajorityBaseline().fit(
                np.zeros((4, 397), dtype=np.float32), np.array([0, 1, 2, 3])
            )
            save_model(model, path)
            good = hashlib.sha256(path.read_bytes()).hexdigest()
            load_model(path, expected_sha256=good)  # correct hash loads fine
            with pytest.raises(ValueError, match="SHA-256 mismatch"):
                load_model(path, expected_sha256="0" * 64)
