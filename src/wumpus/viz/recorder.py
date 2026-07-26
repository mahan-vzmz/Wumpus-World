"""Episode recorder for the interactive HTML demo.

Replays one agent episode decision-by-decision and captures everything the
demo player needs per frame: the agent's position and vitals, its percepts,
a snapshot of its belief state (when the agent keeps a knowledge base), the
reasoning trace behind the chosen action, and the action itself.

The recorder is agent-agnostic by design: any object implementing the Agent
protocol can be recorded. Agents without a ``_kb`` attribute simply produce
frames with ``belief=None`` (the demo then hides the belief layer), and
agents without a ``reasoning_log`` produce empty traces. This is what lets
future steps add the search/ML/baseline agents without changing the format.

Belief snapshots use one character per cell (row-major, 64 chars for 8x8):

    'u' unknown          's' inferred safe      'v' visited
    'p' possible pit     'w' possible Wumpus    'b' both possible
    'P' confirmed pit    'W' confirmed Wumpus   'k' known pit (entered)
    'x' blocked (wall)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wumpus.core.domain import GameConfig, GameMap, Position, Status
from wumpus.core.engine import compute_diagnostic_score, init_state, step
from wumpus.core.observation import Observation, make_observation
from wumpus.core.parser import parse_input

RECORD_VERSION = 1

#: Every belief code the recorder may emit (kept in sync with the docstring).
BELIEF_CODES = frozenset("usvpwbPWkx")


def snapshot_belief(agent: Any, grid_size: int = 8) -> str | None:
    """Encode the agent's per-cell beliefs as a compact row-major string.

    Returns ``None`` for agents that do not maintain a knowledge base.
    """
    kb = getattr(agent, "_kb", None)
    if kb is None:
        return None
    from wumpus.ai.knowledge import CellStatus

    codes: list[str] = []
    for r in range(grid_size):
        for c in range(grid_size):
            p = Position(r, c)
            if kb.has_confirmed_pit(p):
                code = "P"
            elif kb.has_confirmed_wumpus(p):
                code = "W"
            elif kb.is_known_pit(p):
                code = "k"
            elif kb.status(p) is CellStatus.BLOCKED:
                code = "x"
            elif kb.is_visited(p):
                code = "v"
            else:
                possible_pit = kb.has_possible_pit(p)
                possible_wumpus = kb.has_possible_wumpus(p)
                if possible_pit and possible_wumpus:
                    code = "b"
                elif possible_pit:
                    code = "p"
                elif possible_wumpus:
                    code = "w"
                elif kb.is_safe(p):
                    code = "s"
                else:
                    code = "u"
            codes.append(code)
    return "".join(codes)


def _frame(
    state_pos: Position,
    obs: Observation,
    diag_score: int,
    status: Status,
    action: str | None,
    belief: str | None,
    trace: list[str],
) -> dict[str, Any]:
    return {
        "pos": [state_pos.row, state_pos.col],
        "health": obs.health,
        "steps": obs.steps,
        "gold": obs.collected_gold,
        "score": diag_score,
        "status": status.value,
        "breeze": obs.breeze,
        "stench": obs.stench,
        "glitter": obs.glitter,
        "action": action,
        "belief": belief,
        "trace": trace,
    }


def record_episode(
    agent: Any,
    game_map: GameMap,
    config: GameConfig,
    seed: int = 42,
    agent_name: str = "rules",
) -> dict[str, Any]:
    """Run one episode and return a JSON-serializable playback record.

    Frame ``i`` captures the state *before* action ``i`` together with the
    belief the agent held when choosing it and the reasoning trace behind it.
    One extra terminal frame captures the final state with ``action=None``.
    """
    state = init_state(game_map, config)

    public_map_info: dict[str, Any] = {
        "grid_size": config.grid_size,
        "exit_position": config.exit_position,
    }
    if getattr(agent, "requires_full_map", False):
        public_map_info["game_map"] = game_map
    agent.reset(config, public_map_info, seed)

    # Offline planners (SearchAgent) expose their plan diagnostics after reset.
    planner: dict[str, Any] | None = None
    search_result = getattr(agent, "search_result", None)
    if search_result is not None:
        planner = {
            "solved": search_result.solved,
            "plan_length": len(search_result.plan),
            "predicted_score": search_result.predicted_score,
            "expanded_nodes": search_result.expanded_nodes,
            "planning_time_ms": round(search_result.planning_time_ms, 2),
        }
        if not search_result.solved:
            # Mirror the runner: an unsolvable map ends before the first move.
            state.status = Status.NO_SOLUTION

    frames: list[dict[str, Any]] = []
    while state.status is Status.RUNNING:
        obs = make_observation(game_map, config, state)

        log = getattr(agent, "reasoning_log", None)
        log_len_before = len(log) if log is not None else 0

        action = agent.choose_action(obs)

        trace: list[str] = []
        if log is not None and len(log) > log_len_before:
            trace = list(log[-1])
        belief = snapshot_belief(agent, config.grid_size)

        frames.append(
            _frame(
                state.position,
                obs,
                compute_diagnostic_score(state, config),
                state.status,
                action.value,
                belief,
                trace,
            )
        )

        state = step(game_map, config, state, action)
        agent.observe_transition(obs, action, state.status)

    # Terminal frame: where the episode ended. The belief is the one the agent
    # held for its final decision (it never observes the terminal cell).
    final_obs = make_observation(game_map, config, state)
    frames.append(
        _frame(
            state.position,
            final_obs,
            compute_diagnostic_score(state, config),
            state.status,
            None,
            frames[-1]["belief"] if frames else snapshot_belief(agent, config.grid_size),
            [],
        )
    )

    record: dict[str, Any] = {
        "version": RECORD_VERSION,
        "agent": agent_name,
        "seed": seed,
        "grid_size": config.grid_size,
        "truth": ["".join(t.value for t in row) for row in game_map.rows],
        "exit": [config.exit_position.row, config.exit_position.col],
        "config": {
            "initial_health": config.initial_health,
            "gold_value": config.gold_value,
            "pit_score_delta": config.pit_score_delta,
            "max_steps": config.max_steps,
        },
        "result": {
            "status": state.status.value,
            "won": state.status is Status.WON,
            "steps": state.steps,
            "score": compute_diagnostic_score(state, config),
        },
        "frames": frames,
    }
    if planner is not None:
        record["planner"] = planner
    return record


def record_episode_from_file(
    map_path: Path,
    agent: Any,
    seed: int = 42,
    agent_name: str = "rules",
) -> dict[str, Any]:
    """Parse a map file and record one episode of ``agent`` on it."""
    parsed = parse_input(map_path.read_text(encoding="utf-8"))
    record = record_episode(agent, parsed.game_map, parsed.config, seed=seed, agent_name=agent_name)
    record["map_name"] = map_path.name
    return record
