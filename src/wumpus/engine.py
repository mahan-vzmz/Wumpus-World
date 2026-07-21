"""Game engine: transition logic for one Wumpus-World episode.

Event ordering per SPEC v1.0 (DEC-004):
  1. Position changes to destination.
  2. steps += 1, health -= 1.
  3. If health <= 0 → DEAD_HEALTH.
  4. If destination is Wumpus → DEAD_WUMPUS.
  5. If destination is Pit → pit_entries += 1, health = max(1, health // 2).
  6. If destination has remaining gold → collect.
  7. If destination is exit and agent alive → WON.
  8. Otherwise → continue (RUNNING).
"""

from __future__ import annotations

from .domain import Action, GameConfig, GameMap, GameState, Position, Status, Tile


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def _find_gold_positions(game_map: GameMap) -> frozenset[Position]:
    """Return positions of all gold tiles on the map."""
    golds: set[Position] = set()
    for r in range(8):
        for c in range(8):
            if game_map.rows[r][c] is Tile.GOLD:
                golds.add(Position(r, c))
    return frozenset(golds)


def init_state(game_map: GameMap, config: GameConfig) -> GameState:
    """Create the initial game state.

    If gold is on the start cell (1,1), it is collected automatically.
    """
    start = Position(0, 0)
    remaining = _find_gold_positions(game_map)
    collected = 0

    # Auto-collect gold at start position
    if start in remaining:
        remaining = remaining - {start}
        collected = 1

    return GameState(
        position=start,
        health=config.initial_health,
        collected_gold=collected,
        remaining_gold=remaining,
        pit_entries=0,
        steps=0,
        status=Status.RUNNING,
        event_log=["START at (1,1)"]
        + ([f"GOLD collected at (1,1), total={collected}"] if collected else []),
    )


# ---------------------------------------------------------------------------
# Legal actions
# ---------------------------------------------------------------------------

def legal_actions(game_map: GameMap, state: GameState) -> list[Action]:
    """Return all actions that lead to a valid, non-wall destination."""
    result: list[Action] = []
    for action in Action:
        dest = state.position.moved(action)
        if dest.is_inside(8) and game_map.tile_at(dest) is not Tile.WALL:
            result.append(action)
    return result


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def compute_score(state: GameState, config: GameConfig) -> int | None:
    """Compute final score for a WON episode, or None for non-winning."""
    if state.status is not Status.WON:
        return None
    return (
        state.health
        + state.collected_gold * config.gold_value
        + state.pit_entries * config.pit_score_delta
    )


def compute_diagnostic_score(state: GameState, config: GameConfig) -> int:
    """Compute score using the same formula, regardless of outcome.

    Useful for analysis of non-winning episodes.
    """
    return (
        state.health
        + state.collected_gold * config.gold_value
        + state.pit_entries * config.pit_score_delta
    )


# ---------------------------------------------------------------------------
# Transition (one step)
# ---------------------------------------------------------------------------

def step(
    game_map: GameMap,
    config: GameConfig,
    state: GameState,
    action: Action,
) -> GameState:
    """Apply *action* to *state* and return the updated state (mutated in place).

    Raises ValueError if action is illegal or the game has already ended.
    """
    if state.status is not Status.RUNNING:
        raise ValueError(f"cannot step in terminal state {state.status.value}")

    dest = state.position.moved(action)

    # Validate legality
    if not dest.is_inside(8):
        raise ValueError(f"action {action.value} leads outside the grid")
    if game_map.tile_at(dest) is Tile.WALL:
        raise ValueError(f"action {action.value} leads into a wall at {dest}")

    tile = game_map.tile_at(dest)

    # --- Step 1: Move ---
    state.position = dest
    row_ext, col_ext = dest.row + 1, dest.col + 1
    state.event_log.append(f"MOVE {action.value} → ({row_ext},{col_ext})")

    # --- Step 2: Movement cost ---
    state.steps += 1
    state.health -= 1
    state.event_log.append(f"HEALTH -= 1 → {state.health}")

    # --- Step 3: Death by exhaustion ---
    if state.health <= 0:
        state.status = Status.DEAD_HEALTH
        state.event_log.append("DEAD_HEALTH: health reached 0")
        return state

    # --- Step 4: Wumpus ---
    if tile is Tile.WUMPUS:
        state.status = Status.DEAD_WUMPUS
        state.event_log.append("DEAD_WUMPUS: entered Wumpus cell")
        return state

    # --- Step 5: Pit ---
    if tile is Tile.PIT:
        state.pit_entries += 1
        old_health = state.health
        state.health = max(1, state.health // 2)
        state.event_log.append(
            f"PIT: health {old_health} → max(1, {old_health}//2) = {state.health}, "
            f"pit_entries={state.pit_entries}"
        )

    # --- Step 6: Gold ---
    if dest in state.remaining_gold:
        state.remaining_gold = state.remaining_gold - {dest}
        state.collected_gold += 1
        state.event_log.append(
            f"GOLD collected at ({row_ext},{col_ext}), total={state.collected_gold}"
        )

    # --- Step 7: Exit ---
    if dest == config.exit_position:
        state.status = Status.WON
        score = compute_score(state, config)
        state.event_log.append(f"WON: reached exit, score={score}")
        return state

    # --- Step 8: Check step limit ---
    if config.max_steps is not None and state.steps >= config.max_steps:
        state.status = Status.STEP_LIMIT
        state.event_log.append(
            f"STEP_LIMIT: reached {state.steps} steps"
        )

    return state
