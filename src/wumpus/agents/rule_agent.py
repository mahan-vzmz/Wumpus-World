"""Rule-based agent with knowledge base and safe pathfinding.

Per SPEC §8: this agent uses ONLY observations (breeze, stench, glitter,
legal_actions) to infer cell safety. It NEVER receives the hidden map.

Architecture:
  1. KnowledgeBase — tracks beliefs via forward-chaining rules
  2. Safe pathfinder — BFS on visited/safe cells
  3. Policy — prioritized action selection (SPEC §8.3)

Action policy priorities:
  P1. Emergency retreat to the exit if the health budget is tight
  P2. Explore the nearest safe frontier cell (prefer progress toward the exit)
  P3. Safe retreat to the exit if no safe exploration target remains
  P4. Last-resort risky move: the least-risky non-confirmed-deadly frontier
      cell, ranked by hazard type and source count with a seeded tie-break

Gold is collected automatically on entry, so there is no active gold-seeking
priority: an online agent can never observe uncollected gold (glitter) before
the engine has already picked it up.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

from wumpus.agents.base import Agent
from wumpus.ai.knowledge import KnowledgeBase
from wumpus.core.domain import Action, GameConfig, Position
from wumpus.core.observation import Observation


def _manhattan(a: Position, b: Position) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


def _bfs_path(
    start: Position,
    goal: Position,
    allowed: set[Position],
) -> list[Position] | None:
    """BFS shortest path from start to goal through allowed cells only.
    Returns list of positions (excluding start, including goal), or None.
    """
    if start == goal:
        return []
    if goal not in allowed:
        return None

    queue: deque[Position] = deque([start])
    parent: dict[Position, Position] = {start: start}

    while queue:
        current = queue.popleft()
        for neighbor in current.neighbors():
            if neighbor in allowed and neighbor not in parent:
                parent[neighbor] = current
                if neighbor == goal:
                    path: list[Position] = []
                    n = neighbor
                    while n != start:
                        path.append(n)
                        n = parent[n]
                    path.reverse()
                    return path
                queue.append(neighbor)
    return None


def _action_to_reach(src: Position, dest: Position) -> Action | None:
    """Return the Action that moves from src to dest (must be adjacent)."""
    for action in Action:
        if src.moved(action) == dest:
            return action
    return None


class RuleAgent(Agent):
    """Online rule-based agent with partial observability."""

    def __init__(self) -> None:
        self._kb = KnowledgeBase()
        self._config: GameConfig | None = None
        self._rng = random.Random()
        self._exit: Position | None = None
        self._path_cache: list[Position] = []
        self.reasoning_log: list[list[str]] = []
        # Previous-step observation, used to detect a pit entry from the
        # abnormal health drop it causes.
        self._prev_health: int | None = None
        self._prev_position: Position | None = None

    def reset(
        self, config: GameConfig, public_map_info: dict[str, Any], seed: int
    ) -> None:
        self._kb = KnowledgeBase(grid_size=config.grid_size)
        self._config = config
        self._exit = config.exit_position
        self._rng = random.Random(seed)
        self._path_cache = []
        self.reasoning_log = []
        self._prev_health = None
        self._prev_position = None

    def choose_action(self, observation: Observation) -> Action:
        assert self._config is not None and self._exit is not None

        pos = observation.position
        health = observation.health

        # Record the ground truth of having just entered this cell BEFORE the
        # KB runs its inference: an ordinary move costs exactly 1 health, so a
        # larger drop means the cell is a (survivable) pit. Doing this first
        # lets a just-entered pit explain a neighbouring breeze during
        # single-candidate confirmation.
        if self._prev_health is None:
            was_pit = False  # start cell is guaranteed safe by the parser
        else:
            was_pit = pos != self._prev_position and health < self._prev_health - 1
        self._kb.observe_entry(pos, was_pit=was_pit)
        self._prev_health = health
        self._prev_position = pos

        # Update knowledge base with the new percepts
        self._kb.update(
            pos=pos,
            breeze=observation.breeze,
            stench=observation.stench,
            glitter=observation.glitter,
            legal_actions=observation.legal_actions,
        )

        trace = list(self._kb.trace)  # copy KB trace
        if was_pit:
            trace.insert(0, f"KNOWN_PIT entered at ({pos.row+1},{pos.col+1})")

        passable = self._kb.safe_and_visited_cells()

        # Survivable cells for a desperate retreat: every cell except walls and
        # anything suspected of a Wumpus. Pits — possible, confirmed, or already
        # entered — are INCLUDED, because a pit only halves health (survivable)
        # while a Wumpus kills instantly. So the agent will cross any pit to
        # reach the exit, but never routes through a possible Wumpus.
        survivable = {
            Position(r, c)
            for r in range(self._config.grid_size)
            for c in range(self._config.grid_size)
            if not self._kb.is_blocked(Position(r, c))
            and not self._kb.has_wumpus_suspicion(Position(r, c))
        }

        min_dist_to_exit = _manhattan(pos, self._exit)
        safety_margin = 4
        emergency = health <= min_dist_to_exit + safety_margin

        # --- Follow a cached multi-step path only while it is still valid ---
        # It is dropped when the health budget turns tight (so P1 can re-plan a
        # retreat) or when the next hop is no longer provably safe (so newly
        # inferred hazards are never walked into on a stale plan).
        if self._path_cache and not emergency:
            next_cell = self._path_cache[0]
            if next_cell in passable or next_cell == self._exit:
                action = _action_to_reach(pos, next_cell)
                if action and action in observation.legal_actions:
                    self._path_cache.pop(0)
                    trace.append(f"FOLLOW cached path -> ({next_cell.row+1},{next_cell.col+1})")
                    self.reasoning_log.append(trace)
                    return action
            self._path_cache = []  # path invalidated
        elif emergency:
            self._path_cache = []  # drop stale exploration path before retreat

        # === POLICY ===

        # P1: Emergency retreat — if health budget is tight
        if emergency:
            trace.append(f"P1 EMERGENCY RETREAT: health={health}, min_exit_dist={min_dist_to_exit}")
            # Try 1: 100% safe path to exit
            path_to_exit = _bfs_path(pos, self._exit, passable | {self._exit})
            if path_to_exit is not None:
                action = self._follow_path(pos, path_to_exit, trace, observation)
                if action:
                    self.reasoning_log.append(trace)
                    return action

            # Try 2: Path through survivable cells — accept pit risk to reach
            # the exit, but never route through a possible Wumpus.
            risky_path = _bfs_path(pos, self._exit, survivable)
            if risky_path is not None:
                action = self._follow_path(pos, risky_path, trace, observation)
                if action:
                    self.reasoning_log.append(trace)
                    return action

        # P2: Explore safe frontier (utility = prioritize cells closer to exit)
        safe_frontier = self._kb.frontier()
        if safe_frontier:
            safe_frontier.sort(key=lambda f: (_manhattan(f, self._exit), _manhattan(pos, f)))
            for target in safe_frontier:
                path = _bfs_path(pos, target, passable | {target})
                if path:
                    trace.append(f"P2 EXPLORE safe frontier ({target.row+1},{target.col+1})")
                    action = self._follow_path(pos, path, trace, observation)
                    if action:
                        self.reasoning_log.append(trace)
                        return action

        # P3: Safe retreat to exit if reachable and safe
        path_to_exit = _bfs_path(pos, self._exit, passable | {self._exit})
        if path_to_exit is not None and len(path_to_exit) > 0:
            trace.append("P3 SAFE RETREAT: heading to exit")
            action = self._follow_path(pos, path_to_exit, trace, observation)
            if action:
                self.reasoning_log.append(trace)
                return action

        # P4: Last-resort risky move. Rank every non-confirmed-deadly frontier
        # cell by estimated risk (a possible Wumpus scores far higher than a
        # possible pit, which scores higher than an unknown cell), then by
        # proximity to the exit, with a deterministic seeded tie-break. A
        # possible Wumpus is therefore only ever chosen when nothing safer is
        # left — the agent never *prefers* an instant-death gamble.
        risky = self._kb.frontier_candidates()
        if risky:
            risky.sort(key=lambda r: (
                self._kb.risk_score(r),
                _manhattan(r, self._exit),
                self._rng.random(),
            ))
            for target in risky:
                trace.append(
                    f"P4 RISKY MOVE to ({target.row+1},{target.col+1}) "
                    f"[risk={self._kb.risk_score(target):.0f}]"
                )
                action = _action_to_reach(pos, target)
                if action and action in observation.legal_actions:
                    self.reasoning_log.append(trace)
                    return action
                path = _bfs_path(pos, target, passable | {target})
                if path:
                    step_action = self._follow_path(pos, path, trace, observation)
                    if step_action:
                        self.reasoning_log.append(trace)
                        return step_action

        # Absolute fallback: pick the least-bad legal action. Prefer moves that
        # risk neither a confirmed hazard nor a possible Wumpus; then merely
        # avoid confirmed hazards; only as the very last resort, any legal move.
        def _dest(a: Action) -> Position:
            return pos.moved(a)

        no_wumpus_risk = [
            a for a in observation.legal_actions
            if not self._kb.is_dangerous(_dest(a))
            and not self._kb.has_wumpus_suspicion(_dest(a))
        ]
        non_deadly_actions = [
            a for a in observation.legal_actions
            if not self._kb.is_dangerous(_dest(a))
        ]
        choices = no_wumpus_risk or non_deadly_actions or list(observation.legal_actions)
        trace.append("FALLBACK: least-bad legal action")
        action = self._rng.choice(choices)
        self.reasoning_log.append(trace)
        return action

    def _follow_path(
        self, pos: Position, path: list[Position],
        trace: list[str], observation: Observation
    ) -> Action | None:
        """Take the first step of a path, caching the rest."""
        if not path:
            return None
        next_cell = path[0]
        action = _action_to_reach(pos, next_cell)
        if action and action in observation.legal_actions:
            self._path_cache = path[1:]
            trace.append(f"  STEP -> ({next_cell.row+1},{next_cell.col+1}) via {action.value}")
            return action
        return None

    def observe_transition(
        self, observation: Observation, action: Action, outcome: Any
    ) -> None:
        pass
