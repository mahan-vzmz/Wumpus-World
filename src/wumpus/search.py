"""A* search solver for Wumpus World with full map visibility.

This module defines the search problem as a cost-minimization task
equivalent to score-maximization (SPEC §7.2):

    C = initial_health + total_gold_count * gold_value
    loss = health_spent + pit_entries * |pit_score_delta| + remaining_gold * gold_value
    final_score = C - loss

So minimizing `loss` ≡ maximizing `final_score`.

Search state: (position, health, remaining_gold_frozenset)
- Wumpus cells are pruned from successors (instant death).
- Pit cells apply their health penalty in the transition cost.
- Gold cells remove the gold from remaining_gold.
- The exit is the goal; reaching it adds terminal cost for uncollected gold.

Heuristic (admissible): Manhattan distance from current position to exit.
This is a lower bound on the movement cost (each step costs 1 health)
and never overestimates because it ignores walls, pits, and missed gold.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import NamedTuple

from wumpus.domain import Action, GameConfig, GameMap, Position, Tile


# ---------------------------------------------------------------------------
# Search state (hashable, used as dict key)
# ---------------------------------------------------------------------------

class SearchState(NamedTuple):
    """Immutable, hashable state for A* search."""
    position: Position
    health: int
    remaining_gold: frozenset[Position]


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchResult:
    """Output of the A* planner."""
    solved: bool
    plan: tuple[Action, ...] = ()
    predicted_score: int | None = None
    expanded_nodes: int = 0
    peak_frontier: int = 0
    planning_time_ms: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manhattan(a: Position, b: Position) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


def _legal_actions_for_search(game_map: GameMap, pos: Position) -> list[Action]:
    """Return actions leading to in-grid, non-wall cells."""
    result: list[Action] = []
    for action in Action:
        dest = pos.moved(action)
        if dest.is_inside(8) and game_map.tile_at(dest) is not Tile.WALL:
            result.append(action)
    return result


# ---------------------------------------------------------------------------
# A* solver
# ---------------------------------------------------------------------------

# Priority queue entry: (f, tie_breaker, g, state, parent_index, action)
# tie_breaker ensures FIFO ordering for equal f values.

@dataclass
class _Node:
    f: int
    tie: int
    g: int
    state: SearchState
    parent_idx: int  # -1 for start
    action: Action | None

    def __lt__(self, other: "_Node") -> bool:
        return (self.f, self.tie) < (other.f, other.tie)


def solve_astar(
    game_map: GameMap,
    config: GameConfig,
) -> SearchResult:
    """Find the score-maximizing plan using A* with full map knowledge.

    Returns a SearchResult with the optimal action sequence or a failure reason.
    """
    t0 = time.perf_counter()

    exit_pos = config.exit_position

    # Collect gold positions from the map
    all_gold: frozenset[Position] = frozenset(
        Position(r, c)
        for r in range(8)
        for c in range(8)
        if game_map.rows[r][c] is Tile.GOLD
    )

    # Initial state — gold at start (0,0) is auto-collected per engine rules
    start_pos = Position(0, 0)
    start_gold = all_gold
    start_collected = 0
    if start_pos in start_gold:
        start_gold = start_gold - {start_pos}
        start_collected = 1

    start_state = SearchState(
        position=start_pos,
        health=config.initial_health,
        remaining_gold=start_gold,
    )

    # h: admissible heuristic
    # Component 1: Manhattan distance to exit (lower bound on movement cost).
    # Component 2: remaining_gold * gold_value — this is the EXACT terminal
    #   cost if no more gold is collected. Collecting gold can only reduce
    #   this term, so adding it never overestimates → still admissible.
    def h(s: SearchState) -> int:
        return (_manhattan(s.position, exit_pos)
                + len(s.remaining_gold) * config.gold_value)

    # g=0 at start (no loss incurred yet)
    start_node = _Node(f=h(start_state), tie=0, g=0, state=start_state,
                       parent_idx=-1, action=None)

    # Open list (min-heap) and closed set (best-g per state)
    open_list: list[_Node] = [start_node]
    best_g: dict[SearchState, int] = {start_state: 0}
    closed: list[_Node] = []  # indexed list for path reconstruction
    tie_counter = 1
    expanded = 0
    peak_frontier = 1

    pit_penalty = abs(config.pit_score_delta)

    while open_list:
        node = heapq.heappop(open_list)
        s = node.state

        # Skip if we already found a better path to this state
        if node.g > best_g.get(s, float("inf")):
            continue

        node_idx = len(closed)
        closed.append(node)
        expanded += 1

        # --- Goal test: reached exit alive ---
        if s.position == exit_pos:
            # Terminal cost: uncollected gold value
            terminal_cost = len(s.remaining_gold) * config.gold_value
            total_loss = node.g + terminal_cost

            # Reconstruct score
            C = config.initial_health + len(all_gold) * config.gold_value
            predicted_score = C - total_loss

            # Reconstruct plan
            plan = _reconstruct_plan(closed, node_idx)

            elapsed = (time.perf_counter() - t0) * 1000.0
            return SearchResult(
                solved=True,
                plan=plan,
                predicted_score=predicted_score,
                expanded_nodes=expanded,
                peak_frontier=peak_frontier,
                planning_time_ms=elapsed,
                reason="optimal path found",
            )

        # --- Expand successors ---
        for action in _legal_actions_for_search(game_map, s.position):
            dest = s.position.moved(action)
            tile = game_map.tile_at(dest)

            # Prune lethal moves
            if tile is Tile.WUMPUS:
                continue  # instant death — skip

            new_health = s.health - 1  # movement cost
            if new_health <= 0:
                continue  # would die of exhaustion

            # Transition cost for this edge = 1 (health spent on movement)
            edge_cost = 1

            # Pit effect
            new_pits = 0
            if tile is Tile.PIT:
                new_health = max(1, new_health // 2)
                edge_cost += pit_penalty  # score penalty from pit
                # Also account for the extra health lost due to pit
                # The health_spent component: we need to track actual health
                # loss vs just the movement cost of 1.
                # health_loss_from_pit = (s.health - 1) - max(1, (s.health - 1) // 2)
                # But this is already captured via the final health in the state.

            # Gold collection
            new_gold = s.remaining_gold
            if dest in new_gold:
                new_gold = new_gold - {dest}

            new_state = SearchState(
                position=dest,
                health=new_health,
                remaining_gold=new_gold,
            )

            # g tracks total loss:
            #   loss = (initial_health - remaining_health) + pit_entries * |pit_delta| + remaining_gold_count * gold_value
            # But remaining_gold cost is only paid at terminal, so g tracks:
            #   g = (initial_health - current_health) + pit_entries * |pit_delta|
            # This means g = (initial_health - new_health) + cumulative_pit_penalties
            # 
            # Simpler: g(successor) = initial_health - new_health + cumulative pit penalties
            # Let's rewrite: since we track health in state, we can compute:
            #   new_g = (config.initial_health - new_health) + pit_score_loss
            #
            # But pit_score_loss is separate from health loss. We need to track both.
            # 
            # Actually, let's think carefully:
            #   loss = health_spent + pit_score_penalties + uncollected_gold_value
            #   health_spent = initial_health - remaining_health
            #   remaining_health is tracked in state
            #   pit_score_penalties need to be accumulated
            #
            # So: g = health_spent + pit_score_penalties = (initial_health - state.health) + pit_penalties_so_far
            # 
            # For the edge: health goes from s.health to new_health
            #   health_spent_this_edge = s.health - new_health (includes movement AND pit)
            #   pit_penalty_this_edge = pit_penalty if PIT else 0
            #
            # new_g = node.g + (s.health - new_health) + (pit_penalty if PIT else 0)

            health_spent_this_edge = s.health - new_health
            pit_penalty_this_edge = pit_penalty if tile is Tile.PIT else 0
            new_g = node.g + health_spent_this_edge + pit_penalty_this_edge

            if new_g < best_g.get(new_state, float("inf")):
                best_g[new_state] = new_g
                f_val = new_g + h(new_state)
                new_node = _Node(f=f_val, tie=tie_counter, g=new_g,
                                 state=new_state, parent_idx=node_idx,
                                 action=action)
                tie_counter += 1
                heapq.heappush(open_list, new_node)

        peak_frontier = max(peak_frontier, len(open_list))

    # Exhausted search — no solution
    elapsed = (time.perf_counter() - t0) * 1000.0
    return SearchResult(
        solved=False,
        expanded_nodes=expanded,
        peak_frontier=peak_frontier,
        planning_time_ms=elapsed,
        reason="no safe path to exit exists",
    )


def _reconstruct_plan(closed: list[_Node], goal_idx: int) -> tuple[Action, ...]:
    """Walk parent pointers back to start and reverse."""
    actions: list[Action] = []
    idx = goal_idx
    while idx >= 0:
        node = closed[idx]
        if node.action is not None:
            actions.append(node.action)
        idx = node.parent_idx
    actions.reverse()
    return tuple(actions)
