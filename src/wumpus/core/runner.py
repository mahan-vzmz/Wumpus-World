import time
from dataclasses import dataclass

from wumpus.agents.base import Agent
from wumpus.core.domain import GameConfig, GameMap, GameState, Status
from wumpus.core.engine import init_state, step
from wumpus.core.observation import make_observation


@dataclass(frozen=True)
class RunResult:
    """Result of running one episode of an agent on a map."""
    state: GameState
    error: str | None = None
    runtime_ms: float = 0.0

    @property
    def won(self) -> bool:
        return self.state.status == Status.WON


def run_episode(
    agent: Agent,
    game_map: GameMap,
    config: GameConfig,
    seed: int = 42,
) -> RunResult:
    """Run one full episode for the given agent on the given map.

    Failures are caught and converted into a structured result (with an
    ``AGENT_ERROR`` or ``ENGINE_ERROR`` status) so that a single bad episode
    never aborts a whole batch experiment.
    """
    start_time = time.perf_counter()

    def _elapsed() -> float:
        return (time.perf_counter() - start_time) * 1000.0

    def _fail(status: Status, exc: Exception) -> RunResult:
        state.status = status
        state.event_log.append(f"{status.value}: {exc}")
        return RunResult(state=state, error=str(exc), runtime_ms=_elapsed())

    state = init_state(game_map, config)

    # Public info the agent is allowed to see.
    public_map_info = {
        "grid_size": config.grid_size,
        "exit_position": config.exit_position,
    }
    if getattr(agent, "requires_full_map", False):
        public_map_info["game_map"] = game_map

    # 1. Set up the agent — a failure here is the agent's fault.
    try:
        agent.reset(config, public_map_info, seed)
    except Exception as e:
        return _fail(Status.AGENT_ERROR, e)

    # SearchAgent can prove that no safe route exists before the first
    # transition.  Represent that outcome explicitly instead of asking an
    # empty plan to choose a fallback action.
    search_result = getattr(agent, "search_result", None)
    if search_result is not None and not search_result.solved:
        state.status = Status.NO_SOLUTION
        reason = search_result.reason or "no safe path to exit exists"
        state.event_log.append(f"NO_SOLUTION: {reason}")
        return RunResult(state=state, error=None, runtime_ms=_elapsed())

    # 2. Main game loop.
    #
    # Errors are separated by scope: an engine/observation fault is recorded as
    # ENGINE_ERROR and never blamed on the agent, while a fault in the agent's
    # own code or an illegal action is recorded as AGENT_ERROR.
    while state.status == Status.RUNNING:
        # Build the observation (engine side).
        try:
            obs = make_observation(game_map, config, state)
        except Exception as e:
            return _fail(Status.ENGINE_ERROR, e)

        # Ask the agent for an action (agent side).
        try:
            action = agent.choose_action(obs)
        except Exception as e:
            return _fail(Status.AGENT_ERROR, e)

        # Agent contract: the action must be legal.
        if action not in obs.legal_actions:
            label = getattr(action, "value", action)
            return _fail(
                Status.AGENT_ERROR,
                ValueError(f"illegal action {label!r} is not in legal_actions"),
            )

        # Apply the transition (action already validated; a fault here is the
        # engine's).
        try:
            state = step(game_map, config, state, action)
        except Exception as e:
            return _fail(Status.ENGINE_ERROR, e)

        # Notify the agent of the outcome (optional). A failure in this hook
        # must not erase an already-decided terminal result.
        try:
            agent.observe_transition(obs, action, state.status)
        except Exception as e:
            if state.status == Status.RUNNING:
                return _fail(Status.AGENT_ERROR, e)
            state.event_log.append(f"WARN observe_transition raised post-terminal: {e}")

    return RunResult(state=state, error=None, runtime_ms=_elapsed())
