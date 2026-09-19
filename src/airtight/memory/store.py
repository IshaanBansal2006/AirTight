from __future__ import annotations

from collections import defaultdict
from typing import Any

from airtight.memory.items import (
    Claim,
    CoverageCell,
    Evidence,
    Key,
    MemoryItem,
    item_adapter,
    key_of,
)

Region = tuple[float, float, float, float]


class FleetMemoryStore:
    """A state-based CRDT: every entry kind has a merge rule that is commutative, associative and idempotent,
    so replicas converge whatever order deltas arrive in and however often they are repeated.

    `version` counts state changes on this replica; each entry carries the version at which it last changed,
    which is what `delta(since_version)` selects on. A merged entry that changed local state gets a fresh
    stamp so it keeps propagating through further hops.
    """

    def __init__(self, cell_m: float = 5.0) -> None:
        self.cell_m = cell_m
        self._items: dict[Key, CoverageCell | Claim | Evidence] = {}
        self._stamps: dict[Key, int] = {}
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def __len__(self) -> int:
        return len(self._items)

    def observe(self, item: CoverageCell | Claim | Evidence) -> None:
        self._apply(item)

    def merge(self, delta: bytes) -> None:
        for line in delta.decode().splitlines():
            if line.strip():
                self._apply(item_adapter.validate_json(line))

    def delta(self, since_version: int, byte_budget: int) -> bytes:
        """Entries changed after since_version, newest first, as JSONL cut to fit the budget."""
        changed = sorted(
            ((stamp, key) for key, stamp in self._stamps.items() if stamp > since_version),
            key=lambda p: (-p[0], p[1]),
        )
        lines: list[bytes] = []
        used = 0
        for _, key in changed:
            line = self._items[key].model_dump_json().encode() + b"\n"
            if used + len(line) > byte_budget:
                break
            lines.append(line)
            used += len(line)
        return b"".join(lines)

    def query(self, kind: str, region: Region | None = None) -> list[Any]:
        out = [it for (k, _), it in self._items.items() if k == kind]
        if region is None:
            return out
        xmin, ymin, xmax, ymax = region
        return [
            it
            for it in out
            if (p := self._position(it)) is None or (xmin <= p[0] <= xmax and ymin <= p[1] <= ymax)
        ]

    def evidence_score(self, object_id: str) -> float:
        return float(sum(it.score for it in self.query("evidence") if it.object_id == object_id))

    def evidence_scores(self) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for it in self.query("evidence"):
            totals[it.object_id] += it.score
        return dict(totals)

    def snapshot(self) -> list[dict[str, Any]]:
        """Canonical state for equality checks: items sorted by key, stamps excluded."""
        return [self._items[k].model_dump() for k in sorted(self._items)]

    def _position(self, it: CoverageCell | Claim | Evidence) -> tuple[float, float] | None:
        if isinstance(it, Evidence):
            return (it.x, it.y)
        if isinstance(it, CoverageCell):
            return ((it.cx + 0.5) * self.cell_m, (it.cy + 0.5) * self.cell_m)
        return None

    def _apply(self, item: CoverageCell | Claim | Evidence) -> bool:
        key = key_of(item)
        current = self._items.get(key)
        merged = item if current is None else _merge_pair(current, item)
        if current is not None and merged == current:
            return False
        self._version += 1
        self._items[key] = merged
        self._stamps[key] = self._version
        return True


def _merge_pair(a: MemoryItem, b: MemoryItem) -> MemoryItem:
    if isinstance(a, CoverageCell) and isinstance(b, CoverageCell):
        if (b.last_seen_t, b.by) > (a.last_seen_t, a.by):
            return b
        return a
    if isinstance(a, Claim) and isinstance(b, Claim):
        if (b.t, b.agent_id) > (a.t, a.agent_id):
            return b
        return a
    if isinstance(a, Evidence) and isinstance(b, Evidence):
        return b if b.model_dump_json() > a.model_dump_json() else a
    raise TypeError(f"cannot merge {type(a).__name__} with {type(b).__name__} under the same key")
