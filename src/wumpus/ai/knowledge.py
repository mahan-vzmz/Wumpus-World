"""Knowledge Base for the rule-based agent.

Maintains beliefs about each cell using observations and logical inference.
The KB tracks:
  - Visited cells and their percepts (breeze, stench, glitter)
  - Cell status: SAFE, POSSIBLE_PIT, CONFIRMED_PIT, POSSIBLE_WUMPUS,
                 CONFIRMED_WUMPUS, BLOCKED, UNKNOWN
  - Independent pit and Wumpus suspicion sources per cell
  - Persistent negative facts (NotPit / NotWumpus) so a proven-clear cell can
    never be re-suspected by a later observation
  - Known pits: cells the agent stepped into and survived (traversable but
    kept out of ordinary safe routing)
  - Frontier: unvisited cells adjacent to the explored region
  - Reasoning trace for every deduction

Key inference rules:
  - No breeze at c  →  every valid neighbour is permanently NotPit
  - No stench at c  →  every valid neighbour is permanently NotWumpus
  - If a cell has no pit/Wumpus suspicion  →  SAFE
  - Breeze at c  →  unresolved, non-NotPit neighbours become PossiblePit
  - Stench at c  →  unresolved, non-NotWumpus neighbours become PossibleWumpus
  - If exactly one unconfirmed candidate remains for a percept  →  CONFIRMED

This is a multi-hazard environment (several pits and up to two Wumpuses), so
confirming one hazard never clears a source's other candidates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto

from wumpus.core.domain import Action, Position


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
        # Pit and Wumpus beliefs are independent.  A cell can be implicated
        # by both percepts at the same time, so a single CellStatus value is
        # kept only as a backwards-compatible aggregate view.
        self._confirmed_pits: set[Position] = set()
        self._confirmed_wumpuses: set[Position] = set()

        # Persistent negative facts.  A no-breeze/no-stench percept proves a
        # neighbour cannot hold that hazard, regardless of which earlier
        # positive percept made it a candidate.  These sets make that proof
        # permanent so a later observation can never re-suspect the cell.
        self._not_pit: set[Position] = set()
        self._not_wumpus: set[Position] = set()

        # Pits are survivable in this environment.  A cell the agent has
        # actually stepped into and taken pit damage on is a *known* pit:
        # traversable in emergencies but excluded from ordinary safe routing.
        self._known_pits: set[Position] = set()

        # Reasoning trace for the current step
        self.trace: list[str] = []

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------

    def status(self, pos: Position) -> CellStatus:
        return self._status.get(pos, CellStatus.UNKNOWN)

    def is_safe(self, pos: Position) -> bool:
        return (
            self.status(pos) is CellStatus.SAFE
            and not self.has_pit_suspicion(pos)
            and not self.has_wumpus_suspicion(pos)
        )

    def is_visited(self, pos: Position) -> bool:
        return pos in self._visited

    def is_dangerous(self, pos: Position) -> bool:
        return (
            self.status(pos) is CellStatus.BLOCKED
            or pos in self._confirmed_pits
            or pos in self._confirmed_wumpuses
        )

    def is_blocked(self, pos: Position) -> bool:
        """Return whether *pos* is a wall or outside the grid."""
        return self.status(pos) is CellStatus.BLOCKED

    def has_pit_suspicion(self, pos: Position) -> bool:
        """Return whether *pos* is a possible or confirmed pit."""
        if pos in self._not_pit:
            return False
        return pos in self._confirmed_pits or (
            pos not in self._visited and bool(self._pit_sources.get(pos))
        )

    def has_wumpus_suspicion(self, pos: Position) -> bool:
        """Return whether *pos* is a possible or confirmed Wumpus."""
        if pos in self._not_wumpus:
            return False
        return pos in self._confirmed_wumpuses or (
            pos not in self._visited and bool(self._wumpus_sources.get(pos))
        )

    def has_possible_pit(self, pos: Position) -> bool:
        """Return whether *pos* is a possible (not confirmed) pit."""
        return (
            pos not in self._visited
            and pos not in self._confirmed_pits
            and pos not in self._not_pit
            and bool(self._pit_sources.get(pos))
        )

    def has_possible_wumpus(self, pos: Position) -> bool:
        """Return whether *pos* is a possible (not confirmed) Wumpus."""
        return (
            pos not in self._visited
            and pos not in self._confirmed_wumpuses
            and pos not in self._not_wumpus
            and bool(self._wumpus_sources.get(pos))
        )

    def has_confirmed_pit(self, pos: Position) -> bool:
        return pos in self._confirmed_pits

    def has_confirmed_wumpus(self, pos: Position) -> bool:
        return pos in self._confirmed_wumpuses

    def is_known_pit(self, pos: Position) -> bool:
        """Return whether the agent has stepped into *pos* and taken pit damage."""
        return pos in self._known_pits

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

    def frontier_candidates(self) -> list[Position]:
        """Unvisited, non-confirmed-deadly cells adjacent to the explored region.

        This includes both ``UNKNOWN`` cells and cells merely *suspected* of a
        pit or Wumpus.  It is the option set for a last-resort risky move when
        no risk-free frontier remains; the caller ranks it via ``risk_score``.
        """
        result: set[Position] = set()
        for v in self._visited:
            for n in self._valid_neighbors(v):
                if n in self._visited:
                    continue
                if self.is_dangerous(n):  # confirmed pit / Wumpus / wall
                    continue
                result.add(n)
        return sorted(result, key=lambda p: (p.row, p.col))

    def risk_score(self, pos: Position) -> float:
        """Deterministic risk estimate for an unconfirmed frontier cell.

        A possible Wumpus dominates the score because it kills instantly,
        whereas a pit is survivable; an unknown cell with no suspicion is the
        safest gamble.  More implicating sources raise the estimate slightly.
        """
        score = 0.0
        if self.has_possible_wumpus(pos):
            score += 100.0 + 10.0 * len(self._wumpus_sources.get(pos, ()))
        if self.has_possible_pit(pos):
            score += 10.0 + 1.0 * len(self._pit_sources.get(pos, ()))
        return score

    def safe_and_visited_cells(self) -> set[Position]:
        """All cells the agent can safely walk through for *ordinary* routing.

        Known pits are excluded here: they are survivable but costly, so a
        normal safe path should route around them.  Emergency policies may
        still cross them via the wider ``is_dangerous`` filter.
        """
        cells: set[Position] = set()
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                p = Position(r, c)
                if p in self._known_pits:
                    continue
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

        # Glitter is recorded as a percept above.  Because the engine collects
        # gold automatically on entry, glitter is effectively never observed in
        # a normal episode, so no active gold-seeking target list is kept.
        if glitter:
            self.trace.append(f"GLITTER at ({pos.row+1},{pos.col+1})")

        # --- Apply inference rules ---
        neighbors = self._valid_neighbors(pos)

        # Rule 1: No breeze → neighbors are provably NOT pits (permanent fact)
        if not breeze:
            self.trace.append(f"NO_BREEZE at ({pos.row+1},{pos.col+1})")
            for n in neighbors:
                self._mark_not_pit(n, pos)
        else:
            self.trace.append(f"BREEZE at ({pos.row+1},{pos.col+1})")
            # Rule 4: Breeze → unresolved neighbors become pit candidates
            self._add_hazard_candidates(pos, neighbors, is_pit=True)

        # Rule 2: No stench → neighbors are provably NOT Wumpuses (permanent fact)
        if not stench:
            self.trace.append(f"NO_STENCH at ({pos.row+1},{pos.col+1})")
            for n in neighbors:
                self._mark_not_wumpus(n, pos)
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
    # Notify: the agent stepped into a pit (learned from a health drop)
    # ------------------------------------------------------------------

    def mark_known_pit(self, pos: Position) -> None:
        """Record that *pos* is a pit the agent has entered and survived.

        The cell stays walkable in emergencies but is kept out of ordinary
        safe routing (see ``safe_and_visited_cells``).
        """
        if pos not in self._known_pits:
            self._known_pits.add(pos)
            # A cell the agent physically stood on is certainly not a Wumpus.
            self._not_wumpus.add(pos)
            self.trace.append(f"KNOWN_PIT at ({pos.row+1},{pos.col+1})")

    def observe_entry(self, pos: Position, was_pit: bool) -> None:
        """Record the ground truth of physically entering *pos*.

        Standing on a cell while still alive proves it is not a Wumpus.  A
        health drop larger than the single point every move costs proves the
        cell is a (survivable) pit; an ordinary drop proves it is *not* a pit.
        This must be called BEFORE :meth:`update` for the same step so that a
        just-entered pit can already explain a neighbouring breeze during
        single-candidate confirmation (otherwise an innocent sibling cell could
        be wrongly confirmed as a pit).

        Blind spot: entering a pit at health 2 loses exactly one point, so it
        is indistinguishable from a normal move and is mis-recorded as not a
        pit — an accepted, documented limitation shared with the agent-side
        detection.
        """
        self._not_wumpus.add(pos)
        if was_pit:
            self.mark_known_pit(pos)
        else:
            self._not_pit.add(pos)

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

    def _mark_not_pit(self, cell: Position, source: Position) -> None:
        """Record the permanent fact that `cell` cannot be a pit.

        A no-breeze percept at `source` (adjacent to `cell`) proves this
        regardless of which earlier breeze made `cell` a candidate, so every
        pit source for the cell is dropped and it can never be re-suspected.
        Negative direct evidence overrides any prior inferred confirmation.
        """
        self._not_pit.add(cell)
        self._confirmed_pits.discard(cell)
        self._pit_sources[cell] = set()
        self._refresh_aggregate_status(
            cell,
            f"not a pit: no breeze from ({source.row+1},{source.col+1})",
        )

    def _mark_not_wumpus(self, cell: Position, source: Position) -> None:
        """Record the permanent fact that `cell` cannot be a Wumpus.

        Symmetric to :meth:`_mark_not_pit` for the no-stench percept.
        """
        self._not_wumpus.add(cell)
        self._confirmed_wumpuses.discard(cell)
        self._wumpus_sources[cell] = set()
        self._refresh_aggregate_status(
            cell,
            f"not a Wumpus: no stench from ({source.row+1},{source.col+1})",
        )

    def _add_hazard_candidates(self, source: Position,
                               neighbors: list[Position], is_pit: bool) -> None:
        """Mark unresolved neighbors as possible pit/wumpus candidates."""
        for n in neighbors:
            s = self.status(n)
            if n in self._visited or s in (CellStatus.SAFE, CellStatus.BLOCKED):
                continue
            if is_pit:
                if (
                    n in self._confirmed_pits
                    or n in self._confirmed_wumpuses
                    or n in self._not_pit  # proven safe from pits earlier
                ):
                    continue
                self._pit_sources[n].add(source)
                self._refresh_aggregate_status(
                    n, f"breeze at ({source.row+1},{source.col+1})"
                )
            else:
                if (
                    n in self._confirmed_pits
                    or n in self._confirmed_wumpuses
                    or n in self._not_wumpus  # proven safe from Wumpuses earlier
                ):
                    continue
                self._wumpus_sources[n].add(source)
                self._refresh_aggregate_status(
                    n, f"stench at ({source.row+1},{source.col+1})"
                )

    def _try_mark_safe(self, cell: Position) -> None:
        """If cell has no hazard suspicion and isn't visited/blocked, mark SAFE."""
        s = self.status(cell)
        if (
            s == CellStatus.UNKNOWN
            and cell not in self._visited
            and not self.has_pit_suspicion(cell)
            and not self.has_wumpus_suspicion(cell)
        ):
            self._set_status(cell, CellStatus.SAFE, "no hazard suspicion")

    def _refresh_aggregate_status(self, cell: Position, reason: str) -> None:
        """Refresh the legacy single-status view from independent beliefs."""
        if self._status.get(cell) is CellStatus.BLOCKED:
            return
        if cell in self._visited:
            target = CellStatus.SAFE
        elif cell in self._confirmed_pits:
            target = CellStatus.CONFIRMED_PIT
        elif cell in self._confirmed_wumpuses:
            target = CellStatus.CONFIRMED_WUMPUS
        elif self.has_possible_pit(cell):
            target = CellStatus.POSSIBLE_PIT
        elif self.has_possible_wumpus(cell):
            target = CellStatus.POSSIBLE_WUMPUS
        else:
            target = CellStatus.UNKNOWN
        self._set_status(cell, target, reason)

    def _propagate(self) -> None:
        """Single-candidate elimination.

        If a breeze (resp. stench) source has exactly one remaining unconfirmed
        candidate and no already-confirmed hazard that explains the percept,
        that candidate must be the hazard and is confirmed.  This stays sound
        even with multiple pits or Wumpuses: the percept guarantees *at least
        one* adjacent hazard, so a lone surviving candidate is forced.

        Note: this environment allows several pits and several Wumpuses, so we
        deliberately do NOT clear other candidates once one hazard is confirmed
        — a confirmed hazard proves a percept has *an* explanation, never that
        the source's other neighbours are hazard-free.
        """
        changed = True
        while changed:
            changed = False

            for src, percept in self._percepts.items():
                if percept.breeze:
                    candidates = [
                        n for n in self._valid_neighbors(src)
                        if self.has_possible_pit(n)
                    ]
                    # The breeze is already accounted for if a neighbour is a
                    # confirmed pit OR a pit the agent has already stepped into
                    # (visited, hence not a candidate but still the real cause).
                    # Missing the known-pit case would wrongly confirm the lone
                    # remaining candidate as a pit.
                    explained = any(
                        self.has_confirmed_pit(n) or self.is_known_pit(n)
                        for n in self._valid_neighbors(src)
                    )
                    if not explained and len(candidates) == 1:
                        c = candidates[0]
                        self._confirmed_pits.add(c)
                        self._set_status(c, CellStatus.CONFIRMED_PIT,
                                         f"only candidate for breeze at ({src.row+1},{src.col+1})")
                        changed = True

                if percept.stench:
                    candidates = [
                        n for n in self._valid_neighbors(src)
                        if self.has_possible_wumpus(n)
                    ]
                    confirmed = [
                        n for n in self._valid_neighbors(src)
                        if self.has_confirmed_wumpus(n)
                    ]
                    if not confirmed and len(candidates) == 1:
                        c = candidates[0]
                        self._confirmed_wumpuses.add(c)
                        self._set_status(c, CellStatus.CONFIRMED_WUMPUS,
                                         f"only candidate for stench at ({src.row+1},{src.col+1})")
                        changed = True
