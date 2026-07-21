"""Knowledge Base for the rule-based agent.

Maintains beliefs about each cell using observations and logical inference.
Per SPEC §8, the KB tracks:
  - Visited cells and their percepts (breeze, stench, glitter)
  - Cell status: SAFE, POSSIBLE_PIT, CONFIRMED_PIT, POSSIBLE_WUMPUS,
                 CONFIRMED_WUMPUS, BLOCKED, UNKNOWN
  - Frontier: unvisited cells adjacent to the explored region
  - Reasoning trace for every deduction

Key inference rules (§8.2):
  - No breeze at c  →  all valid neighbors are NOT pits
  - No stench at c  →  all valid neighbors are NOT wumpus
  - If cell has no pit/wumpus suspicion  →  SAFE
  - Breeze at c  →  unresolved neighbors become PossiblePit candidates
  - Stench at c →  unresolved neighbors become PossibleWumpus candidates
  - If only one candidate remains for a percept source  →  CONFIRMED
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto

from wumpus.domain import Action, Position


class CellStatus(Enum):
    """Belief status about a single cell."""
    UNKNOWN = auto()
    SAFE = auto()
    POSSIBLE_PIT = auto()
    CONFIRMED_PIT = auto()
    POSSIBLE_WUMPUS = auto()
    CONFIRMED_WUMPUS = auto()
    BLOCKED = auto()   # wall or out-of-grid


@dataclass
class CellPercept:
    """Recorded percept when the agent visited a cell."""
    breeze: bool = False
    stench: bool = False
    glitter: bool = False


class KnowledgeBase:
    """Rule-based knowledge base with forward-chaining inference."""

    def __init__(self, grid_size: int = 8) -> None:
        self.grid_size = grid_size
        self._status: dict[Position, CellStatus] = {}
        self._visited: set[Position] = set()
        self._percepts: dict[Position, CellPercept] = {}

        # Track which visited cells sourced a breeze/stench that
        # implicates each candidate cell.
        self._pit_sources: dict[Position, set[Position]] = defaultdict(set)
        self._wumpus_sources: dict[Position, set[Position]] = defaultdict(set)

        # Known gold locations (seen via glitter but not yet collected)
        self.known_gold: set[Position] = set()

        # Reasoning trace for the current step
        self.trace: list[str] = []

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def status(self, pos: Position) -> CellStatus:
        return self._status.get(pos, CellStatus.UNKNOWN)

    def is_safe(self, pos: Position) -> bool:
        return self.status(pos) in (CellStatus.SAFE,)

    def is_visited(self, pos: Position) -> bool:
        return pos in self._visited

    def is_passable(self, pos: Position) -> bool:
        """Can the agent walk on this cell? (safe or visited, not blocked/confirmed danger)."""
        s = self.status(pos)
        return s in (CellStatus.SAFE, CellStatus.UNKNOWN) or pos in self._visited

    def is_dangerous(self, pos: Position) -> bool:
        s = self.status(pos)
        return s in (CellStatus.CONFIRMED_PIT, CellStatus.CONFIRMED_WUMPUS, CellStatus.BLOCKED)

    def frontier(self) -> list[Position]:
        """Unvisited, safe cells adjacent to visited cells — best exploration targets."""
        result: list[Position] = []
        for v in self._visited:
            for n in self._valid_neighbors(v):
                if n not in self._visited and self.is_safe(n):
                    result.append(n)
        # Deduplicate preserving order
        seen: set[Position] = set()
        deduped: list[Position] = []
        for p in result:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        return deduped

    def risky_frontier(self) -> list[Position]:
        """Unvisited cells adjacent to visited that are UNKNOWN (not safe, not confirmed danger)."""
        result: set[Position] = set()
        for v in self._visited:
            for n in self._valid_neighbors(v):
                if n not in self._visited and self.status(n) == CellStatus.UNKNOWN:
                    result.add(n)
        return sorted(result, key=lambda p: (p.row, p.col))

    def safe_and_visited_cells(self) -> set[Position]:
        """All cells the agent can safely walk through."""
        cells: set[Position] = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                p = Position(r, c)
                if p in self._visited or self.is_safe(p):
                    cells.add(p)
        return cells

    # ------------------------------------------------------------------
    # Update: process a new observation
    # ------------------------------------------------------------------

    def update(self, pos: Position, breeze: bool, stench: bool,
               glitter: bool, legal_actions: tuple[Action, ...]) -> None:
        """Incorporate a new observation at the given position."""
        self.trace = []  # reset trace for this step

        # Mark visited and safe
        self._visited.add(pos)
        self._set_status(pos, CellStatus.SAFE, f"visited ({pos.row+1},{pos.col+1})")

        # Record percepts
        self._percepts[pos] = CellPercept(breeze=breeze, stench=stench, glitter=glitter)

        # Mark out-of-grid / wall neighbors as blocked
        for action in Action:
            n = pos.moved(action)
            if not n.is_inside(self.grid_size):
                self._set_status(n, CellStatus.BLOCKED, "out of grid")
            elif action not in legal_actions and self.status(n) == CellStatus.UNKNOWN:
                # If action is not legal, neighbor must be a wall
                self._set_status(n, CellStatus.BLOCKED, f"wall detected from ({pos.row+1},{pos.col+1})")

        # Gold tracking
        if glitter:
            self.known_gold.add(pos)
            self.trace.append(f"GLITTER at ({pos.row+1},{pos.col+1})")

        # --- Apply inference rules ---
        neighbors = self._valid_neighbors(pos)

        # Rule 1: No breeze → neighbors are NOT pits
        if not breeze:
            self.trace.append(f"NO_BREEZE at ({pos.row+1},{pos.col+1})")
            for n in neighbors:
                self._clear_pit_suspicion(n, pos)
        else:
            self.trace.append(f"BREEZE at ({pos.row+1},{pos.col+1})")
            # Rule 4: Breeze → unresolved neighbors become pit candidates
            self._add_hazard_candidates(pos, neighbors, is_pit=True)

        # Rule 2: No stench → neighbors are NOT wumpus
        if not stench:
            self.trace.append(f"NO_STENCH at ({pos.row+1},{pos.col+1})")
            for n in neighbors:
                self._clear_wumpus_suspicion(n, pos)
        else:
            self.trace.append(f"STENCH at ({pos.row+1},{pos.col+1})")
            # Rule 5: Stench → unresolved neighbors become wumpus candidates
            self._add_hazard_candidates(pos, neighbors, is_pit=False)

        # Rule 3: If no suspicion at all → SAFE
        for n in neighbors:
            self._try_mark_safe(n)

        # Run constraint propagation (single-candidate elimination)
        self._propagate()

    # ------------------------------------------------------------------
    # Notify gold collected
    # ------------------------------------------------------------------

    def gold_collected(self, pos: Position) -> None:
        self.known_gold.discard(pos)

    # ------------------------------------------------------------------
    # Internal inference helpers
    # ------------------------------------------------------------------

    def _valid_neighbors(self, pos: Position) -> list[Position]:
        return [n for n in pos.neighbors() if n.is_inside(self.grid_size)]

    def _set_status(self, pos: Position, status: CellStatus, reason: str) -> None:
        old = self._status.get(pos, CellStatus.UNKNOWN)
        if old != status:
            self._status[pos] = status
            self.trace.append(f"SET ({pos.row+1},{pos.col+1}) = {status.name} [{reason}]")

    def _clear_pit_suspicion(self, cell: Position, source: Position) -> None:
        """Remove pit suspicion on `cell` caused by `source`."""
        s = self.status(cell)
        if s in (CellStatus.POSSIBLE_PIT,):
            self._pit_sources[cell].discard(source)
            if not self._pit_sources[cell]:
                # No more sources implicating this cell as a possible pit
                if self.status(cell) == CellStatus.POSSIBLE_PIT:
                    self._set_status(cell, CellStatus.UNKNOWN,
                                     f"pit cleared: no breeze from ({source.row+1},{source.col+1})")
                    self._try_mark_safe(cell)

    def _clear_wumpus_suspicion(self, cell: Position, source: Position) -> None:
        s = self.status(cell)
        if s in (CellStatus.POSSIBLE_WUMPUS,):
            self._wumpus_sources[cell].discard(source)
            if not self._wumpus_sources[cell]:
                if self.status(cell) == CellStatus.POSSIBLE_WUMPUS:
                    self._set_status(cell, CellStatus.UNKNOWN,
                                     f"wumpus cleared: no stench from ({source.row+1},{source.col+1})")
                    self._try_mark_safe(cell)

    def _add_hazard_candidates(self, source: Position,
                               neighbors: list[Position], is_pit: bool) -> None:
        """Mark unresolved neighbors as possible pit/wumpus candidates."""
        for n in neighbors:
            s = self.status(n)
            if n in self._visited or s in (CellStatus.SAFE, CellStatus.BLOCKED,
                                            CellStatus.CONFIRMED_PIT, CellStatus.CONFIRMED_WUMPUS):
                continue
            if is_pit:
                if s not in (CellStatus.POSSIBLE_WUMPUS, CellStatus.CONFIRMED_WUMPUS):
                    self._pit_sources[n].add(source)
                    if s != CellStatus.POSSIBLE_PIT:
                        self._set_status(n, CellStatus.POSSIBLE_PIT,
                                         f"breeze at ({source.row+1},{source.col+1})")
            else:
                if s not in (CellStatus.POSSIBLE_PIT, CellStatus.CONFIRMED_PIT):
                    self._wumpus_sources[n].add(source)
                    if s != CellStatus.POSSIBLE_WUMPUS:
                        self._set_status(n, CellStatus.POSSIBLE_WUMPUS,
                                         f"stench at ({source.row+1},{source.col+1})")

    def _try_mark_safe(self, cell: Position) -> None:
        """If cell has no hazard suspicion and isn't visited/blocked, mark SAFE."""
        s = self.status(cell)
        if s == CellStatus.UNKNOWN and cell not in self._visited:
            self._set_status(cell, CellStatus.SAFE, "no hazard suspicion")

    def _propagate(self) -> None:
        """Single-candidate elimination: if a breeze/stench source has only
        one unresolved candidate, that candidate is confirmed."""
        changed = True
        while changed:
            changed = False

            # Check each visited cell with breeze
            for src, percept in self._percepts.items():
                if percept.breeze:
                    candidates = [
                        n for n in self._valid_neighbors(src)
                        if self.status(n) == CellStatus.POSSIBLE_PIT
                    ]
                    # Also count confirmed pits that already explain this breeze
                    confirmed = [
                        n for n in self._valid_neighbors(src)
                        if self.status(n) == CellStatus.CONFIRMED_PIT
                    ]
                    if not confirmed and len(candidates) == 1:
                        c = candidates[0]
                        self._set_status(c, CellStatus.CONFIRMED_PIT,
                                         f"only candidate for breeze at ({src.row+1},{src.col+1})")
                        changed = True

                if percept.stench:
                    candidates = [
                        n for n in self._valid_neighbors(src)
                        if self.status(n) == CellStatus.POSSIBLE_WUMPUS
                    ]
                    confirmed = [
                        n for n in self._valid_neighbors(src)
                        if self.status(n) == CellStatus.CONFIRMED_WUMPUS
                    ]
                    if not confirmed and len(candidates) == 1:
                        c = candidates[0]
                        self._set_status(c, CellStatus.CONFIRMED_WUMPUS,
                                         f"only candidate for stench at ({src.row+1},{src.col+1})")
                        changed = True

            # After confirming a hazard, re-check if other cells can be cleared
            for src, percept in self._percepts.items():
                neighbors = self._valid_neighbors(src)
                if percept.breeze:
                    has_confirmed_pit = any(
                        self.status(n) == CellStatus.CONFIRMED_PIT for n in neighbors
                    )
                    if has_confirmed_pit:
                        for n in neighbors:
                            if self.status(n) == CellStatus.POSSIBLE_PIT:
                                self._pit_sources[n].discard(src)
                                if not self._pit_sources[n]:
                                    self._set_status(n, CellStatus.UNKNOWN,
                                                     f"pit explained by confirmed neighbor of ({src.row+1},{src.col+1})")
                                    self._try_mark_safe(n)
                                    changed = True

                if percept.stench:
                    has_confirmed_wumpus = any(
                        self.status(n) == CellStatus.CONFIRMED_WUMPUS for n in neighbors
                    )
                    if has_confirmed_wumpus:
                        for n in neighbors:
                            if self.status(n) == CellStatus.POSSIBLE_WUMPUS:
                                self._wumpus_sources[n].discard(src)
                                if not self._wumpus_sources[n]:
                                    self._set_status(n, CellStatus.UNKNOWN,
                                                     f"wumpus explained by confirmed neighbor of ({src.row+1},{src.col+1})")
                                    self._try_mark_safe(n)
                                    changed = True
