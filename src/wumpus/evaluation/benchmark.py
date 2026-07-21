"""Benchmark suite runner and statistical analyzer (T601, T603).

Per SPEC §11:
- Runs all agents across the test map suite
- Logs raw results to CSV and JSON
- Generates aggregate comparison tables (Win Rate, Score, Health, Steps, Pits, Wumpus Deaths, Runtime)
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from wumpus.agents.greedy_agent import GreedyExitAgent
from wumpus.agents.ml_agent import MLAgent
from wumpus.agents.random_agent import RandomAgent
from wumpus.agents.rule_agent import RuleAgent
from wumpus.agents.search_agent import SearchAgent
from wumpus.domain import Status
from wumpus.engine import compute_diagnostic_score, compute_score, init_state, step
from wumpus.observation import make_observation
from wumpus.parser import parse_input


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
    error: str | None = None


def run_single_benchmark(
    agent_name: str,
    agent_obj: Any,
    map_path: Path,
    seed: int = 42,
) -> BenchmarkRow:
    """Run one agent on one map and collect structured benchmark metrics."""
    category = map_path.name.split("_map_")[0] if "_map_" in map_path.name else "general"
    visibility = "Full" if agent_name == "search" else "Partial"

    text = map_path.read_text(encoding="utf-8")
    parsed = parse_input(text)

    # Build public_map_info
    public_info: dict[str, Any] = {
        "grid_size": parsed.config.grid_size,
        "exit_position": parsed.config.exit_position,
    }
    if agent_name == "search":
        public_info["game_map"] = parsed.game_map

    t0 = time.perf_counter()
    error_msg: str | None = None

    try:
        agent_obj.reset(parsed.config, public_info, seed)
        state = init_state(parsed.game_map, parsed.config)

        while state.status == Status.RUNNING:
            obs = make_observation(parsed.game_map, parsed.config, state)
            action = agent_obj.choose_action(obs)
            state = step(parsed.game_map, parsed.config, state, action)

    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000.0
        error_msg = str(e)
        if 'state' not in locals():
            state = init_state(parsed.game_map, parsed.config)
            state.status = Status.DEAD_HEALTH

    elapsed = (time.perf_counter() - t0) * 1000.0

    won = state.status == Status.WON
    score = compute_score(state, parsed.config)
    diag_score = compute_diagnostic_score(state, parsed.config)
    wumpus_death = state.status == Status.DEAD_WUMPUS

    return BenchmarkRow(
        map_name=map_path.name,
        category=category,
        agent=agent_name,
        visibility=visibility,
        status=state.status.value,
        won=won,
        final_score=score,
        diagnostic_score=diag_score,
        health_remaining=state.health,
        steps_taken=state.steps,
        gold_collected=state.collected_gold,
        pit_entries=state.pit_entries,
        wumpus_death=wumpus_death,
        runtime_ms=elapsed,
        error=error_msg,
    )


def run_benchmark_suite(
    maps_dir: Path,
    results_dir: Path,
    seed: int = 42,
) -> list[BenchmarkRow]:
    """Run all 5 agents on all maps in maps_dir and save raw CSV/JSON results."""
    results_dir.mkdir(parents=True, exist_ok=True)
    map_files = sorted(list(maps_dir.glob("*.txt")))

    if not map_files:
        raise ValueError(f"No .txt map files found in '{maps_dir}'")

    rows: list[BenchmarkRow] = []

    # Agent factory
    def get_agents():
        ml_agent = MLAgent()
        model_p = Path("artifacts/models/random_forest.joblib")
        if model_p.is_file():
            ml_agent.load(model_p)

        return {
            "search": SearchAgent(),
            "rules": RuleAgent(),
            "ml": ml_agent,
            "greedy": GreedyExitAgent(),
            "random": RandomAgent(),
        }

    agents = get_agents()

    for map_p in map_files:
        for agent_name, agent_obj in agents.items():
            row = run_single_benchmark(agent_name, agent_obj, map_p, seed=seed)
            rows.append(row)

    # Save to CSV
    csv_path = results_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    return rows


def generate_summary_table(rows: list[BenchmarkRow]) -> dict[str, dict[str, Any]]:
    """Compute aggregate comparison metrics grouped by Agent."""
    agents = sorted(list({r.agent for r in rows}))
    summary: dict[str, dict[str, Any]] = {}

    for agent in agents:
        agent_rows = [r for r in rows if r.agent == agent]
        total_runs = len(agent_rows)
        wins = sum(1 for r in agent_rows if r.won)
        win_rate = (wins / total_runs) * 100.0 if total_runs > 0 else 0.0

        scores = [r.diagnostic_score for r in agent_rows]
        mean_score = sum(scores) / total_runs if total_runs > 0 else 0.0

        steps = [r.steps_taken for r in agent_rows]
        mean_steps = sum(steps) / total_runs if total_runs > 0 else 0.0

        pits = sum(r.pit_entries for r in agent_rows)
        wumpus_deaths = sum(1 for r in agent_rows if r.wumpus_death)

        runtimes = [r.runtime_ms for r in agent_rows]
        mean_runtime = sum(runtimes) / total_runs if total_runs > 0 else 0.0

        visibility = agent_rows[0].visibility if agent_rows else "Unknown"

        summary[agent] = {
            "visibility": visibility,
            "total_runs": total_runs,
            "win_rate_pct": win_rate,
            "mean_score": mean_score,
            "mean_steps": mean_steps,
            "total_pits": pits,
            "wumpus_deaths": wumpus_deaths,
            "mean_runtime_ms": mean_runtime,
        }

    return summary
