import argparse
import json
import sys
from pathlib import Path

from wumpus.agents.greedy_agent import GreedyExitAgent
from wumpus.agents.ml_agent import MLAgent
from wumpus.agents.random_agent import RandomAgent
from wumpus.agents.rule_agent import RuleAgent
from wumpus.agents.search_agent import SearchAgent
from wumpus.dataset import DatasetConfig, generate_dataset, save_dataset, split_dataset
from wumpus.domain import Status
from wumpus.engine import compute_score
from wumpus.evaluation.benchmark import generate_summary_table, run_benchmark_suite
from wumpus.evaluation.suite_generator import generate_map_suite
from wumpus.ml import evaluate_classifier, save_model, train_models
from wumpus.parser import InputFormatError, parse_input
from wumpus.runner import run_episode


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

    # Command: validate
    val_parser = subparsers.add_parser("validate", help="Validate a map file")
    val_parser.add_argument("--input", required=True, type=str, help="Path to the map file")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run an agent on a map")
    run_parser.add_argument("--input", required=True, type=str, help="Path to the map file")
    run_parser.add_argument("--agent", choices=["random", "greedy", "search", "rules", "ml"], default="random", help="Which agent to run")
    run_parser.add_argument("--model", type=str, default=None, help="Path to trained model file (for ML agent)")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--trace", action="store_true", help="Print reasoning trace (for rule agent)")

    # Command: dataset
    ds_parser = subparsers.add_parser("dataset", help="Generate dataset from A* demonstrations")
    ds_parser.add_argument("--num-maps", type=int, default=20, help="Number of maps to generate")
    ds_parser.add_argument("--seed", type=int, default=100, help="Seed for dataset generation")
    ds_parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for dataset")

    # Command: train
    train_parser = subparsers.add_parser("train", help="Train ML models on dataset")
    train_parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to dataset directory")
    train_parser.add_argument("--output-dir", type=str, default="artifacts/models", help="Output directory for saved models")

    # Command: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run comprehensive benchmark comparing all agents")
    bench_parser.add_argument("--maps-dir", type=str, default="data/maps/test_suite", help="Path to test maps suite")
    bench_parser.add_argument("--results-dir", type=str, default="results", help="Path to save benchmark CSV results")
    bench_parser.add_argument("--generate-suite", action="store_true", help="Generate 20 test maps across 5 categories first")
    bench_parser.add_argument("--model", type=str, default="artifacts/models/random_forest.joblib", help="Path to trained model file")
    bench_parser.add_argument("--skip-ml", action="store_true", help="Run the benchmark without MLAgent")

    args = parser.parse_args()

    if args.command == "dataset":
        print(f"Generating dataset from {args.num_maps} maps (seed={args.seed})...")
        config = DatasetConfig(num_maps=args.num_maps, seed=args.seed)
        data = generate_dataset(config)
        out_path = Path(args.output_dir)
        save_dataset(out_path, data, config=config)
        print(f"Dataset generated with {len(data['y'])} samples across {len(set(data['map_ids']))} maps.")
        print(f"Saved to '{out_path}'.")
        return 0

    elif args.command == "train":
        from wumpus.dataset import load_dataset
        data_path = Path(args.data_dir)
        if not (data_path / "dataset.npz").is_file():
            print(f"Error: Dataset not found at '{data_path}'. Run 'dataset' command first.")
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

        print(f"Dataset loaded: {len(data['y'])} samples.")
        print(f"Splits -> Train: {len(train['y'])}, Val: {len(val['y'])}, Test: {len(test['y'])}")

        print("Training models...")
        results = train_models(train, val)
        metrics = results["metrics"]

        print("\n--- Validation Metrics ---")
        for m_name, m_val in metrics.items():
            print(f"  {m_name:15s} -> Accuracy: {m_val['accuracy']:.4f}, Macro-F1: {m_val['macro_f1']:.4f}")

        out_dir = Path(args.output_dir)
        rf_path = out_dir / "random_forest.joblib"
        save_model(results["models"]["random_forest"], rf_path)
        test_metrics = evaluate_classifier(results["models"]["random_forest"], test)
        metrics_path = out_dir / "training_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "validation": metrics,
                    "test": test_metrics,
                    "split_sizes": {
                        "train": len(train["y"]),
                        "validation": len(val["y"]),
                        "test": len(test["y"]),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved main Random Forest model to '{rf_path}'.")
        print(f"Saved validation/test metrics to '{metrics_path}'.")
        return 0

    elif args.command == "benchmark":
        maps_path = Path(args.maps_dir)
        if args.generate_suite or not maps_path.exists() or not list(maps_path.glob("*.txt")):
            print("Generating test suite maps across 5 categories...")
            generate_map_suite(maps_path, base_seed=500)
            print(f"20 test maps generated at '{maps_path}'.")

        res_path = Path(args.results_dir)
        print(f"\nRunning benchmark suite on all maps in '{maps_path}'...")
        try:
            rows = run_benchmark_suite(
                maps_path,
                res_path,
                seed=42,
                model_path=Path(args.model),
                include_ml=not args.skip_ml,
            )
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return 2

        summary = generate_summary_table(rows)

        print("\n" + "=" * 80)
        print("🏆 WUMPUS WORLD BENCHMARK COMPARISON SUMMARY")
        print("=" * 80)
        header = f"{'Agent':12s} | {'Visibility':10s} | {'Win Rate':10s} | {'Mean Score':10s} | {'Mean Steps':10s} | {'Pit Entries':11s} | {'Wumpus Deaths':13s} | {'Runtime (ms)':12s}"
        print(header)
        print("-" * 80)

        for agent_name, stats in summary.items():
            line = (
                f"{agent_name:12s} | "
                f"{stats['visibility']:10s} | "
                f"{stats['win_rate_pct']:8.1f}% | "
                f"{stats['mean_score']:10.1f} | "
                f"{stats['mean_steps']:10.1f} | "
                f"{stats['total_pits']:11d} | "
                f"{stats['wumpus_deaths']:13d} | "
                f"{stats['mean_runtime_ms']:12.2f}"
            )
            print(line)

        print("=" * 80)
        print(f"Raw results saved to '{res_path / 'benchmark_results.csv'}'.")
        print(f"JSON summary saved to '{res_path / 'benchmark_summary.json'}'.")
        return 0

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: File not found -> {input_path}")
        return 1

    try:
        text = input_path.read_text(encoding="utf-8")
        parsed = parse_input(text)
    except InputFormatError as e:
        print(f"Validation Error: {e}")
        return 1
    except Exception as e:
        print(f"Error reading file: {e}")
        return 1

    if args.command == "validate":
        print(f"Map '{input_path.name}' is valid.")
        if parsed.warnings:
            for w in parsed.warnings:
                print(f"Warning: {w}")
        return 0

    elif args.command == "run":
        model_p = Path(args.model) if args.model else None
        try:
            agent = _create_agent(args.agent, parsed, model_path=model_p)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return 2

        print(f"Running '{args.agent}' agent on '{input_path.name}' with seed {args.seed}...")
        result = run_episode(agent, parsed.game_map, parsed.config, seed=args.seed)
        state = result.state

        print("\n--- Event Log ---")
        for event in state.event_log:
            print(f"  {event}")

        print("\n--- Results ---")
        print(f"Status: {state.status.value}")
        print(f"Steps taken: {state.steps}")
        print(f"Health remaining: {state.health}")
        print(f"Gold collected: {state.collected_gold}")
        print(f"Pit entries: {state.pit_entries}")

        if state.status is Status.WON:
            score = compute_score(state, parsed.config)
            print(f"Final score: {score}")

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
