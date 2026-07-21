"""Tests for benchmark runner and suite generator (Epic 6).

Covers:
  T600 — Test map suite generation across 5 categories
  T601 — Batch runner execution & raw CSV results
  T602 — Test reproducibility (identical seed gives identical results)
  T603 — Statistical summary table computation
"""

from pathlib import Path
import tempfile
import pytest

from wumpus.evaluation.benchmark import (
    generate_summary_table,
    run_benchmark_suite,
    run_single_benchmark,
)
from wumpus.evaluation.suite_generator import generate_map_suite


class TestBenchmarkSuite:

    def test_generate_map_suite(self):
        """T600: Suite generator creates 20 maps across 5 categories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            files = generate_map_suite(out_path, base_seed=700)

            assert len(files) == 20
            categories = {f.name.split("_map_")[0] for f in files}
            assert len(categories) == 5

    def test_run_benchmark_suite_and_reproducibility(self):
        """T601 & T602: Batch runner creates CSV and seed reproducibility holds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            maps_p = tmp_p / "maps"
            res_p = tmp_p / "results"

            generate_map_suite(maps_p, base_seed=800)
            rows1 = run_benchmark_suite(maps_p, res_p, seed=42)

            assert len(rows1) == 20 * 5  # 20 maps x 5 agents
            assert (res_p / "benchmark_results.csv").is_file()

            # T602 Reproducibility check
            rows2 = run_benchmark_suite(maps_p, res_p, seed=42)
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

            generate_map_suite(maps_p, base_seed=900)
            rows = run_benchmark_suite(maps_p, res_p, seed=42)

            summary = generate_summary_table(rows)
            assert "search" in summary
            assert "rules" in summary
            assert "ml" in summary
            assert "greedy" in summary
            assert "random" in summary

            # SearchAgent with full visibility should have 100% win rate on solvable maps
            assert summary["search"]["win_rate_pct"] == 100.0
