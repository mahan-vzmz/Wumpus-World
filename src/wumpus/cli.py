import argparse
import sys
from pathlib import Path

from wumpus.agents.greedy_agent import GreedyExitAgent
from wumpus.agents.random_agent import RandomAgent
from wumpus.parser import InputFormatError, parse_input
from wumpus.runner import run_episode


def _create_agent(name: str):
    if name == "random":
        return RandomAgent()
    elif name == "greedy":
        return GreedyExitAgent()
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
    run_parser.add_argument("--agent", choices=["random", "greedy"], default="random", help="Which agent to run")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
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
        agent = _create_agent(args.agent)
        print(f"Running '{args.agent}' agent on '{input_path.name}' with seed {args.seed}...")
        
        result = run_episode(agent, parsed.game_map, parsed.config, seed=args.seed)
        
        print("\n--- Event Log ---")
        for event in result.state.event_log:
            print(f"  {event}")
        
        print("\n--- Results ---")
        print(f"Status: {result.state.status.value}")
        print(f"Steps taken: {result.state.steps}")
        print(f"Health remaining: {result.state.health}")
        print(f"Gold collected: {result.state.collected_gold}")
        print(f"Pit entries: {result.state.pit_entries}")
        
        if result.error:
            print(f"\nAGENT ERROR: {result.error}")
            return 1
            
        return 0

    return 1

if __name__ == "__main__":
    sys.exit(main())
