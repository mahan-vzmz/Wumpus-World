import argparse
import sys
from pathlib import Path

from wumpus.agents.greedy_agent import GreedyExitAgent
from wumpus.agents.ml_agent import MLAgent
from wumpus.agents.random_agent import RandomAgent
from wumpus.agents.rule_agent import RuleAgent
from wumpus.agents.search_agent import SearchAgent
from wumpus.dataset import DatasetConfig, generate_dataset, save_dataset, split_dataset
from wumpus.engine import compute_score
from wumpus.ml import save_model, train_models
from wumpus.parser import InputFormatError, parse_input


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
        agent = MLAgent()
        if model_path and model_path.is_file():
            agent.load(model_path)
        else:
            default_model = Path("artifacts/models/random_forest.joblib")
            if default_model.is_file():
                agent.load(default_model)
        return agent
    raise ValueError(f"Unknown agent: {name}")


def _get_public_map_info(agent_name: str, parsed):
    info = {
        "grid_size": parsed.config.grid_size,
        "exit_position": parsed.config.exit_position,
    }
    if agent_name == "search":
        info["game_map"] = parsed.game_map
    return info


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

    args = parser.parse_args()

    if args.command == "dataset":
        print(f"Generating dataset from {args.num_maps} maps (seed={args.seed})...")
        config = DatasetConfig(num_maps=args.num_maps, seed=args.seed)
        data = generate_dataset(config)
        out_path = Path(args.output_dir)
        save_dataset(out_path, data)
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
        train, val, test = split_dataset(data)

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
        print(f"\nSaved main Random Forest model to '{rf_path}'.")
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
        agent = _create_agent(args.agent, parsed, model_path=model_p)

        public_info = _get_public_map_info(args.agent, parsed)

        print(f"Running '{args.agent}' agent on '{input_path.name}' with seed {args.seed}...")

        agent.reset(parsed.config, public_info, args.seed)

        from wumpus.domain import Status
        from wumpus.engine import init_state, step
        from wumpus.observation import make_observation

        state = init_state(parsed.game_map, parsed.config)
        while state.status == Status.RUNNING:
            obs = make_observation(parsed.game_map, parsed.config, state)
            action = agent.choose_action(obs)
            state = step(parsed.game_map, parsed.config, state, action)

        print("\n--- Event Log ---")
        for event in state.event_log:
            print(f"  {event}")

        print("\n--- Results ---")
        print(f"Status: {state.status.value}")
        print(f"Steps taken: {state.steps}")
        print(f"Health remaining: {state.health}")
        print(f"Gold collected: {state.collected_gold}")
        print(f"Pit entries: {state.pit_entries}")

        if state.status == Status.WON:
            score = compute_score(state, parsed.config)
            print(f"Final score: {score}")

        if args.agent == "search" and hasattr(agent, "search_result"):
            sr = agent.search_result
            if sr:
                print(f"\n--- A* Diagnostics ---")
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
