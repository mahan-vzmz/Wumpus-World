"""Rule-based agent with knowledge base and safe pathfinding.

Per SPEC §8: this agent uses ONLY observations (breeze, stench, glitter,
legal_actions) to infer cell safety. It NEVER receives the hidden map.

Architecture:
  1. KnowledgeBase — tracks beliefs via forward-chaining rules
  2. Safe pathfinder — BFS on visited/safe cells
  3. Policy — prioritized action selection (SPEC §8.3)

Action policy priorities:
  P1. Emergency retreat to exit if health budget is tight
  P2. Move toward known gold via safe path
  P3. Explore safe frontier with highest utility (towards exit/gold)
  P4. Safe retreat to exit if no safe exploration targets remain
  P5. Risky move to least-suspicious unknown cell (tie-break by seed)
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

from wumpus.agents.base import Agent
from wumpus.domain import Action, GameConfig, Position
from wumpus.knowledge import CellStatus, KnowledgeBase
from wumpus.observation import Observation


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

    def reset(
        self, config: GameConfig, public_map_info: dict[str, Any], seed: int
    ) -> None:
        self._kb = KnowledgeBase(grid_size=config.grid_size)
        self._config = config
        self._exit = config.exit_position
        self._rng = random.Random(seed)
        self._path_cache = []
        self.reasoning_log = []

    def choose_action(self, observation: Observation) -> Action:
        assert self._config is not None and self._exit is not None

        pos = observation.position
        health = observation.health

        # Update knowledge base with new observation
        self._kb.update(
            pos=pos,
            breeze=observation.breeze,
            stench=observation.stench,
            glitter=observation.glitter,
            legal_actions=observation.legal_actions,
        )

        # Track gold collection
        if observation.glitter:
            self._kb.known_gold.add(pos)
        self._kb.gold_collected(pos)

        trace = list(self._kb.trace)  # copy KB trace

        passable = self._kb.safe_and_visited_cells()

        # Non-deadly cells (all except confirmed pits, confirmed wumpus, and walls)
        non_deadly = {
            Position(r, c)
            for r in range(self._config.grid_size)
            for c in range(self._config.grid_size)
            if not self._kb.is_dangerous(Position(r, c))
        }

        # --- If we have a cached multi-step path, try to follow it ---
        if self._path_cache:
            next_cell = self._path_cache[0]
            if next_cell in passable or next_cell == self._exit or next_cell in non_deadly:
                action = _action_to_reach(pos, next_cell)
                if action and action in observation.legal_actions:
                    self._path_cache.pop(0)
                    trace.append(f"FOLLOW cached path -> ({next_cell.row+1},{next_cell.col+1})")
                    self.reasoning_log.append(trace)
                    return action
            # Path invalidated
            self._path_cache = []

        min_dist_to_exit = _manhattan(pos, self._exit)

        # === POLICY ===

        # P1: Emergency retreat — if health budget is tight
        safety_margin = 4
        if health <= min_dist_to_exit + safety_margin:
            trace.append(f"P1 EMERGENCY RETREAT: health={health}, min_exit_dist={min_dist_to_exit}")
            # Try 1: 100% safe path to exit
            path_to_exit = _bfs_path(pos, self._exit, passable | {self._exit})
            if path_to_exit is not None:
                action = self._follow_path(pos, path_to_exit, trace, observation)
                if action:
                    self.reasoning_log.append(trace)
                    return action

            # Try 2: Path through non-deadly cells (willing to take risk to survive)
            risky_path = _bfs_path(pos, self._exit, non_deadly)
            if risky_path is not None:
                action = self._follow_path(pos, risky_path, trace, observation)
                if action:
                    self.reasoning_log.append(trace)
                    return action

        # P2: Move toward known gold via safe path
        for gold_pos in sorted(self._kb.known_gold, key=lambda g: _manhattan(pos, g)):
            if gold_pos == pos:
                continue
            path = _bfs_path(pos, gold_pos, passable)
            if path:
                trace.append(f"P2 GO TO GOLD at ({gold_pos.row+1},{gold_pos.col+1})")
                action = self._follow_path(pos, path, trace, observation)
                if action:
                    self.reasoning_log.append(trace)
                    return action

        # P3: Explore safe frontier (utility = prioritize cells closer to exit)
        safe_frontier = self._kb.frontier()
        if safe_frontier:
            safe_frontier.sort(key=lambda f: (_manhattan(f, self._exit), _manhattan(pos, f)))
            for target in safe_frontier:
                path = _bfs_path(pos, target, passable | {target})
                if path:
                    trace.append(f"P3 EXPLORE safe frontier ({target.row+1},{target.col+1})")
                    action = self._follow_path(pos, path, trace, observation)
                    if action:
                        self.reasoning_log.append(trace)
                        return action

        # P4: Safe retreat to exit if reachable and safe
        path_to_exit = _bfs_path(pos, self._exit, passable | {self._exit})
        if path_to_exit is not None and len(path_to_exit) > 0:
            trace.append("P4 SAFE RETREAT: heading to exit")
            action = self._follow_path(pos, path_to_exit, trace, observation)
            if action:
                self.reasoning_log.append(trace)
                return action

        # P5: Risky move — pick least suspicious unknown neighbor
        risky = self._kb.risky_frontier()
        if risky:
            risky.sort(key=lambda r: (_manhattan(r, self._exit), self._rng.random()))
            target = risky[0]
            trace.append(f"P5 RISKY MOVE to ({target.row+1},{target.col+1})")
            action = _action_to_reach(pos, target)
            if action and action in observation.legal_actions:
                self.reasoning_log.append(trace)
                return action
            path = _bfs_path(pos, target, passable | {target})
            if path:
                action = self._follow_path(pos, path, trace, observation)
                if action:
                    self.reasoning_log.append(trace)
                    return action

        # Absolute fallback: random legal action
        trace.append("FALLBACK: random legal action")
        action = self._rng.choice(list(observation.legal_actions))
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
