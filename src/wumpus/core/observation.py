"""Observation: the agent-visible snapshot after each transition.

Per SPEC v1.0 §5.3, an observation contains:
  - position, health, collected_gold, steps  (agent's own state)
  - breeze   : True if any orthogonal neighbor is a Pit
  - stench   : True if any orthogonal neighbor is a Wumpus
  - glitter  : True if uncollected Gold is on the current cell
  - at_exit  : True if current position is the exit
  - legal_actions : list of non-wall, in-grid moves

The observation NEVER reveals the hidden map, tile types of unvisited
cells, or exact locations of hazards.
"""

from __future__ import annotations

from dataclasses import dataclass

from .domain import Action, GameConfig, GameMap, GameState, Position, Status, Tile


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything an online agent is allowed to see at one time-step."""

    position: Position
    health: int
    collected_gold: int
    steps: int
    status: Status

    breeze: bool
    stench: bool
    glitter: bool
    at_exit: bool

    legal_actions: tuple[Action, ...]


def _has_neighbor_tile(
    game_map: GameMap, pos: Position, tile_type: Tile
) -> bool:
    """Return True if any orthogonal neighbor of *pos* contains *tile_type*."""
    for neighbor in pos.neighbors():
        if neighbor.is_inside(8):
            if game_map.tile_at(neighbor) is tile_type:
                return True
    return False


def make_observation(
    game_map: GameMap,
    config: GameConfig,
    state: GameState,
) -> Observation:
    """Build an Observation from the current game state.

    This function reads the hidden map only to compute percepts (breeze,
    stench, glitter); it does NOT expose tile identities to the caller.
    """
    pos = state.position

    breeze = _has_neighbor_tile(game_map, pos, Tile.PIT)
    stench = _has_neighbor_tile(game_map, pos, Tile.WUMPUS)
    glitter = pos in state.remaining_gold
    at_exit = pos == config.exit_position

    # Legal actions: in-grid and not a wall
    actions: list[Action] = []
    for action in Action:
        dest = pos.moved(action)
        if dest.is_inside(8) and game_map.tile_at(dest) is not Tile.WALL:
            actions.append(action)

    return Observation(
        position=pos,
        health=state.health,
        collected_gold=state.collected_gold,
        steps=state.steps,
        status=state.status,
        breeze=breeze,
        stench=stench,
        glitter=glitter,
        at_exit=at_exit,
        legal_actions=tuple(actions),
    )
