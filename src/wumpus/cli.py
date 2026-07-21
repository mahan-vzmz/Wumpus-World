import argparse
import sys
from pathlib import Path

from wumpus.agents.greedy_agent import GreedyExitAgent
from wumpus.agents.random_agent import RandomAgent
from wumpus.agents.rule_agent import RuleAgent
from wumpus.agents.search_agent import SearchAgent
from wumpus.engine import compute_score
from wumpus.parser import InputFormatError, parse_input
from wumpus.runner import run_episode


def _create_agent(name: str, parsed):
    if name == "random":
        return RandomAgent()
    elif name == "greedy":
        return GreedyExitAgent()
    elif name == "search":
        return SearchAgent()
    elif name == "rules":
        return RuleAgent()
    raise ValueError(f"Unknown agent: {name}")


def _get_public_map_info(agent_name: str, parsed):
    """Build public_map_info dict. SearchAgent gets the full map."""
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
    run_parser.add_argument("--agent", choices=["random", "greedy", "search", "rules"], default="random", help="Which agent to run")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--trace", action="store_true", help="Print reasoning trace (for rule agent)")

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
        agent = _create_agent(args.agent, parsed)

        # Build public_map_info (search agent gets full map)
        public_info = _get_public_map_info(args.agent, parsed)

        print(f"Running '{args.agent}' agent on '{input_path.name}' with seed {args.seed}...")

        # Reset agent manually so we can pass the right public_map_info
        agent.reset(parsed.config, public_info, args.seed)

        # Run via engine directly
        from wumpus.engine import init_state, step
        from wumpus.observation import make_observation
        from wumpus.domain import Status

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

        # Show A* diagnostics if search agent
        if args.agent == "search" and hasattr(agent, "search_result"):
            sr = agent.search_result
            if sr:
                print(f"\n--- A* Diagnostics ---")
                print(f"Expanded nodes: {sr.expanded_nodes}")
                print(f"Peak frontier: {sr.peak_frontier}")
                print(f"Planning time: {sr.planning_time_ms:.2f} ms")

        # Show reasoning trace if rules agent and --trace requested
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

