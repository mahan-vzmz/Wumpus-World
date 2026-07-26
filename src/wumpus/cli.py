import argparse
import json
import sys
from pathlib import Path

from wumpus.agents.greedy_agent import GreedyExitAgent
from wumpus.agents.ml_agent import MLAgent
from wumpus.agents.random_agent import RandomAgent
from wumpus.agents.rule_agent import RuleAgent
from wumpus.agents.search_agent import SearchAgent
from wumpus.ai.dataset import DatasetConfig, generate_dataset, save_dataset, split_dataset
from wumpus.ai.ml import evaluate_classifier, save_model, train_models
from wumpus.core.domain import Status
from wumpus.core.engine import compute_score
from wumpus.core.parser import InputFormatError, parse_input
from wumpus.core.runner import run_episode
from wumpus.evaluation.benchmark import generate_summary_table, run_benchmark_suite
from wumpus.evaluation.suite_generator import generate_map_suite


def _create_agent(name: str, parsed, model_path: Path | None = None):
    if name == "random":
        return RandomAgent()
    elif name == "greedy":
        return GreedyExitAgent()
    elif name == "search":
        return SearchAgent()
    elif name == "rules":
        return RuleAgent()
    elif name == "ml":
        resolved_model = model_path or Path("artifacts/models/random_forest.joblib")
        if not resolved_model.is_file():
            raise FileNotFoundError(
                f"ML model not found at '{resolved_model}'. Run "
                "'python -m wumpus train' first or pass --model PATH."
            )
        return MLAgent(model_path=resolved_model)
    raise ValueError(f"Unknown agent: {name}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Wumpus World Simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared flag: machine-readable JSON output (SPEC §12).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON output"
    )

    # Command: validate
    val_parser = subparsers.add_parser("validate", help="Validate a map file", parents=[common])
    val_parser.add_argument("--input", required=True, type=str, help="Path to the map file")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run an agent on a map", parents=[common])
    run_parser.add_argument("--input", required=True, type=str, help="Path to the map file")
    run_parser.add_argument("--agent", choices=["random", "greedy", "search", "rules", "ml"], default="random", help="Which agent to run")
    run_parser.add_argument("--model", type=str, default=None, help="Path to trained model file (for ML agent)")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--trace", action="store_true", help="Print reasoning trace (for rule agent)")

    # Command: dataset
    ds_parser = subparsers.add_parser("dataset", help="Generate dataset from A* demonstrations", parents=[common])
    ds_parser.add_argument("--num-maps", type=int, default=20, help="Number of maps to generate")
    ds_parser.add_argument("--seed", type=int, default=100, help="Seed for dataset generation")
    ds_parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for dataset")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train ML models on dataset", parents=[common])
    train_parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to dataset directory")
    train_parser.add_argument("--output-dir", type=str, default="artifacts/models", help="Output directory for saved models")

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run comprehensive benchmark comparing all agents", parents=[common])
    bench_parser.add_argument("--maps-dir", type=str, default="data/maps/test_suite", help="Path to test maps suite")
    bench_parser.add_argument("--results-dir", type=str, default="results", help="Path to save benchmark CSV results")
    bench_parser.add_argument("--generate-suite", action="store_true", help="Generate 20 test maps across 5 categories first")
    bench_parser.add_argument("--model", type=str, default="artifacts/models/random_forest.joblib", help="Path to trained model file")
    bench_parser.add_argument("--skip-ml", action="store_true", help="Run the benchmark without MLAgent")
    bench_parser.add_argument("--seeds", type=int, nargs="+", default=None, help="One or more seeds for multi-seed benchmarking (default: 42)")

    # Command: visualize
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Build the interactive single-file HTML demo (belief map + reasoning log)",
        parents=[common],
    )
    viz_parser.add_argument("--output", type=str, default="docs/demo/index.html", help="Output HTML file")
    viz_parser.add_argument("--seed", type=int, default=42, help="Episode seed")
    viz_parser.add_argument("--model", type=str, default=None, help="Path to trained model for the ML mind")
    viz_parser.add_argument("--skip-ml", action="store_true", help="Build the demo without the ML mind")

    args = parser.parse_args()

    if args.command == "visualize":
        from wumpus.viz.html import AGENT_ORDER, CURATED_EPISODES, write_demo

        out_path = Path(args.output)
        model_p = Path(args.model) if args.model else None
        try:
            size = write_demo(
                Path.cwd(), out_path, seed=args.seed,
                model_path=model_p, include_ml=not args.skip_ml,
            )
        except FileNotFoundError as exc:
            if args.json:
                print(json.dumps({"error": str(exc)}, indent=2))
            else:
                print(f"Error: {exc}")
            return 2
        minds = [a for a in AGENT_ORDER if not args.skip_ml or a != "ml"]
        if args.json:
            print(json.dumps({
                "command": "visualize",
                "output": str(out_path),
                "bytes": size,
                "minds": minds,
                "episodes": [spec.episode_id for spec in CURATED_EPISODES],
            }, indent=2))
        else:
            print(f"Interactive demo written to '{out_path}' ({size / 1024:.0f} KB).")
            print(f"Minds embedded: {', '.join(minds)}.")
            print("Open it directly in a browser — no server needed.")
        return 0

    elif args.command == "dataset":
        if not args.json:
            print(f"Generating dataset from {args.num_maps} maps (seed={args.seed})...")
        config = DatasetConfig(num_maps=args.num_maps, seed=args.seed)
        data = generate_dataset(config)
        out_path = Path(args.output_dir)
        save_dataset(out_path, data, config=config)
        num_samples = len(data["y"])
        num_maps = len(set(data["map_ids"]))
        if args.json:
            print(json.dumps({
                "command": "dataset",
                "num_samples": num_samples,
                "num_maps": num_maps,
                "output_dir": str(out_path),
            }, indent=2))
        else:
            print(f"Dataset generated with {num_samples} samples across {num_maps} maps.")
            print(f"Saved to '{out_path}'.")
        return 0

    elif args.command == "train":
        from wumpus.ai.dataset import load_dataset
        data_path = Path(args.data_dir)
        if not (data_path / "dataset.npz").is_file():
            msg = f"Dataset not found at '{data_path}'. Run 'dataset' command first."
            if args.json:
                print(json.dumps({"error": msg}, indent=2))
            else:
                print(f"Error: {msg}")
            return 1

        data = load_dataset(data_path)
        split_config = {}
        metadata_path = data_path / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            split_config = metadata.get("dataset_config", {})
        train, val, test = split_dataset(
            data,
            train_ratio=float(split_config.get("train_ratio", 0.7)),
            val_ratio=float(split_config.get("val_ratio", 0.15)),
            test_ratio=float(split_config.get("test_ratio", 0.15)),
        )

        results = train_models(train, val)
        metrics = results["metrics"]

        out_dir = Path(args.output_dir)
        rf_path = out_dir / "random_forest.joblib"
        save_model(results["models"]["random_forest"], rf_path)
        test_metrics = evaluate_classifier(results["models"]["random_forest"], test)
        split_sizes = {
            "train": len(train["y"]),
            "validation": len(val["y"]),
            "test": len(test["y"]),
        }
        metrics_path = out_dir / "training_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {"validation": metrics, "test": test_metrics, "split_sizes": split_sizes},
                indent=2,
            ),
            encoding="utf-8",
        )

        if args.json:
            print(json.dumps({
                "command": "train",
                "split_sizes": split_sizes,
                "validation": metrics,
                "test": test_metrics,
                "model_path": str(rf_path),
                "metrics_path": str(metrics_path),
            }, indent=2))
        else:
            print(f"Dataset loaded: {len(data['y'])} samples.")
            print(f"Splits -> Train: {split_sizes['train']}, "
                  f"Val: {split_sizes['validation']}, Test: {split_sizes['test']}")
            print("Training models...")
            print("\n--- Validation Metrics ---")
            for m_name, m_val in metrics.items():
                print(f"  {m_name:15s} -> Accuracy: {m_val['accuracy']:.4f}, "
                      f"Macro-F1: {m_val['macro_f1']:.4f}")
            print(f"\nSaved main Random Forest model to '{rf_path}'.")
            print(f"Saved validation/test metrics to '{metrics_path}'.")
        return 0

    elif args.command == "benchmark":
        maps_path = Path(args.maps_dir)
        if args.generate_suite or not maps_path.exists() or not list(maps_path.glob("*.txt")):
            if not args.json:
                print("Generating test suite maps across 5 categories...")
            generate_map_suite(maps_path, base_seed=500)
            if not args.json:
                print(f"20 test maps generated at '{maps_path}'.")

        res_path = Path(args.results_dir)
        seeds = args.seeds or [42]
        if not args.json:
            print(f"\nRunning benchmark suite on all maps in '{maps_path}' "
                  f"over seed(s) {seeds}...")
        try:
            rows = run_benchmark_suite(
                maps_path,
                res_path,
                model_path=Path(args.model),
                include_ml=not args.skip_ml,
                seeds=seeds,
            )
        except FileNotFoundError as exc:
            if args.json:
                print(json.dumps({"error": str(exc)}, indent=2))
            else:
                print(f"Error: {exc}")
            return 2

        summary = generate_summary_table(rows)

        if args.json:
            print(json.dumps({
                "command": "benchmark",
                "summary": summary,
                "results_csv": str(res_path / "benchmark_results.csv"),
                "results_json": str(res_path / "benchmark_results.json"),
                "summary_json": str(res_path / "benchmark_summary.json"),
            }, indent=2))
            return 0

        n_runs = next(iter(summary.values()))["total_runs"] if summary else 0
        print("\n" + "=" * 92)
        print("🏆 WUMPUS WORLD BENCHMARK COMPARISON SUMMARY")
        print(f"seed(s): {seeds}  ·  {n_runs} episodes per agent  ·  95% bootstrap CI")
        print("=" * 92)
        header = (
            f"{'Agent':10s} | {'Vis':7s} | {'Win% (95% CI)':18s} | "
            f"{'Diag Score':10s} | {'Final(win)':10s} | {'Steps':6s} | "
            f"{'Pits':5s} | {'Wumpus':6s} | {'ms':8s}"
        )
        print(header)
        print("-" * 92)

        for agent_name, stats in summary.items():
            lo, hi = stats["win_rate_ci"]
            win_str = f"{stats['win_rate_pct']:.0f}% [{lo:.0f}-{hi:.0f}]"
            final_win = stats["mean_final_score_wins"]
            final_str = f"{final_win:.1f}" if final_win is not None else "-"
            line = (
                f"{agent_name:10s} | "
                f"{stats['visibility']:7s} | "
                f"{win_str:18s} | "
                f"{stats['mean_score']:10.1f} | "
                f"{final_str:>10s} | "
                f"{stats['mean_steps']:6.1f} | "
                f"{stats['total_pits']:5d} | "
                f"{stats['wumpus_deaths']:6d} | "
                f"{stats['mean_runtime_ms']:8.2f}"
            )
            print(line)

        print("=" * 92)
        print(
            "Diag Score = mean diagnostic score over ALL runs; Final(win) = mean "
            "official score over WINS only."
        )
        print(f"Raw results saved to '{res_path / 'benchmark_results.csv'}'.")
        print(f"JSON summary saved to '{res_path / 'benchmark_summary.json'}'.")
        return 0

    input_path = Path(args.input)
    if not input_path.is_file():
        msg = f"File not found -> {input_path}"
        if args.json:
            print(json.dumps({"error": msg}, indent=2))
        else:
            print(f"Error: {msg}")
        return 1

    try:
        text = input_path.read_text(encoding="utf-8")
        parsed = parse_input(text)
    except InputFormatError as e:
        if args.json:
            print(json.dumps({"valid": False, "error": str(e)}, indent=2))
        else:
            print(f"Validation Error: {e}")
        return 1
    except Exception as e:
        if args.json:
            print(json.dumps({"error": f"error reading file: {e}"}, indent=2))
        else:
            print(f"Error reading file: {e}")
        return 1

    if args.command == "validate":
        if args.json:
            print(json.dumps({"valid": True, "warnings": list(parsed.warnings)}, indent=2))
        else:
            print(f"Map '{input_path.name}' is valid.")
            for w in parsed.warnings:
                print(f"Warning: {w}")
        return 0

    elif args.command == "run":
        model_p = Path(args.model) if args.model else None
        try:
            agent = _create_agent(args.agent, parsed, model_path=model_p)
        except FileNotFoundError as exc:
            if args.json:
                print(json.dumps({"error": str(exc)}, indent=2))
            else:
                print(f"Error: {exc}")
            return 2

        result = run_episode(agent, parsed.game_map, parsed.config, seed=args.seed)
        state = result.state
        final_score = compute_score(state, parsed.config) if state.status is Status.WON else None

        if args.json:
            payload: dict = {
                "command": "run",
                "agent": args.agent,
                "seed": args.seed,
                "status": state.status.value,
                "won": result.won,
                "steps": state.steps,
                "health_remaining": state.health,
                "collected_gold": state.collected_gold,
                "pit_entries": state.pit_entries,
                "final_score": final_score,
                "error": result.error,
                "event_log": list(state.event_log),
            }
            if args.agent == "search" and getattr(agent, "search_result", None):
                sr = agent.search_result
                payload["search"] = {
                    "expanded_nodes": sr.expanded_nodes,
                    "peak_frontier": sr.peak_frontier,
                    "planning_time_ms": sr.planning_time_ms,
                }
            if args.agent == "rules" and args.trace and hasattr(agent, "reasoning_log"):
                payload["reasoning_trace"] = agent.reasoning_log
            print(json.dumps(payload, indent=2))
            return 0

        print(f"Running '{args.agent}' agent on '{input_path.name}' with seed {args.seed}...")

        print("\n--- Event Log ---")
        for event in state.event_log:
            print(f"  {event}")

        print("\n--- Results ---")
        print(f"Status: {state.status.value}")
        print(f"Steps taken: {state.steps}")
        print(f"Health remaining: {state.health}")
        print(f"Gold collected: {state.collected_gold}")
        print(f"Pit entries: {state.pit_entries}")

        if final_score is not None:
            print(f"Final score: {final_score}")

        if args.agent == "search" and hasattr(agent, "search_result"):
            sr = agent.search_result
            if sr:
                print("\n--- A* Diagnostics ---")
                print(f"Expanded nodes: {sr.expanded_nodes}")
                print(f"Peak frontier: {sr.peak_frontier}")
                print(f"Planning time: {sr.planning_time_ms:.2f} ms")

        if args.agent == "rules" and args.trace and hasattr(agent, "reasoning_log"):
            print(f"\n--- Reasoning Trace ({len(agent.reasoning_log)} steps) ---")
            for step_idx, log_lines in enumerate(agent.reasoning_log, start=1):
                print(f"  [Step {step_idx}]")
                for line in log_lines:
                    print(f"    {line}")

        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
