"""Benchmark suite runner and statistical analyzer.

- Runs every agent across a map suite, over one or more seeds.
- Streams raw results to CSV as they complete (so a crash never loses them).
- Skips (and records) unparseable maps instead of aborting the whole run.
- Produces an aggregate comparison with bootstrap confidence intervals.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from wumpus.agents.greedy_agent import GreedyExitAgent
from wumpus.agents.ml_agent import MLAgent
from wumpus.agents.random_agent import RandomAgent
from wumpus.agents.rule_agent import RuleAgent
from wumpus.agents.search_agent import SearchAgent
from wumpus.core.domain import GameConfig, GameMap, Status
from wumpus.core.engine import compute_diagnostic_score, compute_score
from wumpus.core.parser import parse_input
from wumpus.core.runner import run_episode

# Number of bootstrap resamples for confidence intervals; fixed RNG seed keeps
# the intervals reproducible.
_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_SEED = 0


@dataclass
class BenchmarkRow:
    map_name: str
    category: str
    agent: str
    visibility: str
    status: str
    won: bool
    final_score: int | None
    diagnostic_score: int
    health_remaining: int
    steps_taken: int
    gold_collected: int
    pit_entries: int
    wumpus_death: bool
    runtime_ms: float
    seed: int
    expanded_nodes: int | None = None
    peak_frontier: int | None = None
    error: str | None = None


def run_single_benchmark(
    agent_name: str,
    agent_obj: Any,
    map_name: str,
    game_map: GameMap,
    config: GameConfig,
    seed: int = 42,
) -> BenchmarkRow:
    """Run one agent on one (already parsed) map and collect metrics."""
    category = map_name.split("_map_")[0] if "_map_" in map_name else "general"
    visibility = "Full" if agent_name == "search" else "Partial"

    result = run_episode(agent_obj, game_map, config, seed=seed)
    state = result.state
    won = state.status == Status.WON
    search_result = getattr(agent_obj, "search_result", None)

    return BenchmarkRow(
        map_name=map_name,
        category=category,
        agent=agent_name,
        visibility=visibility,
        status=state.status.value,
        won=won,
        final_score=compute_score(state, config),
        diagnostic_score=compute_diagnostic_score(state, config),
        health_remaining=state.health,
        steps_taken=state.steps,
        gold_collected=state.collected_gold,
        pit_entries=state.pit_entries,
        wumpus_death=state.status == Status.DEAD_WUMPUS,
        runtime_ms=result.runtime_ms,
        seed=seed,
        expanded_nodes=search_result.expanded_nodes if search_result else None,
        peak_frontier=search_result.peak_frontier if search_result else None,
        error=result.error,
    )


def _build_agents(include_ml: bool, model_path: Path | None) -> dict[str, Any]:
    agents: dict[str, Any] = {
        "search": SearchAgent(),
        "rules": RuleAgent(),
        "greedy": GreedyExitAgent(),
        "random": RandomAgent(),
    }
    if include_ml:
        resolved_model = model_path or Path("artifacts/models/random_forest.joblib")
        if not resolved_model.is_file():
            raise FileNotFoundError(
                f"ML model not found at '{resolved_model}'. Run "
                "'python -m wumpus train' first, pass a model path, or "
                "disable ML for this benchmark."
            )
        agents["ml"] = MLAgent(model_path=resolved_model)
    return agents


def run_benchmark_suite(
    maps_dir: Path,
    results_dir: Path,
    seed: int = 42,
    model_path: Path | None = None,
    include_ml: bool = True,
    seeds: Sequence[int] | None = None,
) -> list[BenchmarkRow]:
    """Run every agent on every map (over one or more seeds) and save results.

    ``seeds`` defaults to ``[seed]`` for backward compatibility. Rows are
    streamed to CSV as they complete; a map that fails to parse is skipped and
    recorded in the summary rather than aborting the whole suite.
    """
    seed_list = list(seeds) if seeds else [seed]
    results_dir.mkdir(parents=True, exist_ok=True)
    map_files = sorted(maps_dir.glob("*.txt"))
    if not map_files:
        raise ValueError(f"No .txt map files found in '{maps_dir}'")

    agents = _build_agents(include_ml, model_path)

    rows: list[BenchmarkRow] = []
    skipped: list[dict[str, str]] = []
    fieldnames = [f.name for f in fields(BenchmarkRow)]
    csv_path = results_dir / "benchmark_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()

        for map_p in map_files:
            try:
                parsed = parse_input(map_p.read_text(encoding="utf-8"))
            except Exception as exc:  # bad input file: record and skip, do not abort
                skipped.append({"map": map_p.name, "error": str(exc)})
                print(f"WARNING: skipping unparseable map '{map_p.name}': {exc}", file=sys.stderr)
                continue

            for episode_seed in seed_list:
                for agent_name, agent_obj in agents.items():
                    try:
                        row = run_single_benchmark(
                            agent_name, agent_obj, map_p.name,
                            parsed.game_map, parsed.config, seed=episode_seed,
                        )
                    except Exception as exc:  # defensive: never abort the batch
                        row = _error_row(agent_name, map_p.name, episode_seed, exc)
                    rows.append(row)
                    writer.writerow(asdict(row))
            csv_file.flush()

    json_path = results_dir / "benchmark_results.json"
    json_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8"
    )

    resolved_model = model_path or Path("artifacts/models/random_forest.joblib")
    model_sha256 = (
        hashlib.sha256(resolved_model.read_bytes()).hexdigest()
        if include_ml and resolved_model.is_file()
        else None
    )

    summary_path = results_dir / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "experiment": {
                    "seeds": seed_list,
                    "map_count": len(map_files),
                    "maps_evaluated": len(map_files) - len(skipped),
                    "skipped_maps": skipped,
                    "maps_dir": str(maps_dir),
                    "include_ml": include_ml,
                    "model_sha256": model_sha256,
                    "python_version": platform.python_version(),
                    "platform": sys.platform,
                },
                "summary": generate_summary_table(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return rows


def _error_row(agent_name: str, map_name: str, seed: int, exc: Exception) -> BenchmarkRow:
    """Fallback row for an episode that raised outside the runner's guard."""
    category = map_name.split("_map_")[0] if "_map_" in map_name else "general"
    return BenchmarkRow(
        map_name=map_name,
        category=category,
        agent=agent_name,
        visibility="Full" if agent_name == "search" else "Partial",
        status=Status.ENGINE_ERROR.value,
        won=False,
        final_score=None,
        diagnostic_score=0,
        health_remaining=0,
        steps_taken=0,
        gold_collected=0,
        pit_entries=0,
        wumpus_death=False,
        runtime_ms=0.0,
        seed=seed,
        error=str(exc),
    )


def _bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    resamples: int = _BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
) -> list[float]:
    """Percentile bootstrap CI for ``statistic``; deterministic (fixed RNG)."""
    n = len(values)
    if n == 0:
        return [0.0, 0.0]
    if n == 1:
        v = statistic(values)
        return [round(v, 2), round(v, 2)]
    rng = random.Random(_BOOTSTRAP_SEED)
    stats = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic(sample))
    stats.sort()
    lo = stats[int((alpha / 2) * resamples)]
    hi = stats[int((1 - alpha / 2) * resamples)]
    return [round(lo, 2), round(hi, 2)]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def generate_summary_table(rows: list[BenchmarkRow]) -> dict[str, dict[str, Any]]:
    """Aggregate comparison metrics per agent, with bootstrap 95% CIs.

    The independent experimental unit is the MAP, not the (map, seed) episode:
    confidence intervals bootstrap over per-map values (each map's mean over
    seeds). This avoids understating uncertainty for deterministic agents,
    whose repeated seeds would otherwise look like extra independent samples.
    """
    agents = sorted({r.agent for r in rows})
    summary: dict[str, dict[str, Any]] = {}

    for agent in agents:
        agent_rows = [r for r in rows if r.agent == agent]
        map_names = sorted({r.map_name for r in agent_rows})

        # Per-map aggregates over the seeds run for that map.
        per_map_win: list[float] = []
        per_map_score: list[float] = []
        for name in map_names:
            map_rows = [r for r in agent_rows if r.map_name == name]
            per_map_win.append(_mean([1.0 if r.won else 0.0 for r in map_rows]))
            per_map_score.append(_mean([float(r.diagnostic_score) for r in map_rows]))

        scores = [float(r.diagnostic_score) for r in agent_rows]
        final_scores = [r.final_score for r in agent_rows if r.won and r.final_score is not None]

        summary[agent] = {
            "visibility": agent_rows[0].visibility if agent_rows else "Unknown",
            "total_runs": len(agent_rows),
            "n_maps": len(map_names),
            "seeds": sorted({r.seed for r in agent_rows}),
            "win_rate_pct": _mean(per_map_win) * 100.0,
            "win_rate_ci": _bootstrap_ci(per_map_win, lambda s: _mean(s) * 100.0),
            "mean_score": _mean(per_map_score),
            "mean_score_ci": _bootstrap_ci(per_map_score, _mean),
            "median_score": statistics.median(scores) if scores else 0.0,
            "mean_final_score_wins": _mean(final_scores) if final_scores else None,
            "mean_steps": _mean([float(r.steps_taken) for r in agent_rows]),
            "total_pits": sum(r.pit_entries for r in agent_rows),
            "wumpus_deaths": sum(1 for r in agent_rows if r.wumpus_death),
            "mean_runtime_ms": _mean([r.runtime_ms for r in agent_rows]),
        }

    return summary
