from __future__ import annotations

import threading
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
        self._lock = threading.RLock()

    @property
    def version(self) -> int:
        return self._version

    def __len__(self) -> int:
        return len(self._items)

    def observe(self, item: CoverageCell | Claim | Evidence | dict[str, Any]) -> None:
        """Accepts the typed items or the dict shapes lane A's module emits. Safe to call from any thread."""
        with self._lock:
            self._apply(self.coerce(item) if isinstance(item, dict) else item)

    def coerce(self, raw: dict[str, Any]) -> CoverageCell | Claim | Evidence:
        """Dict to item. Coverage dicts carry x/y in metres and are binned to this store's cell size."""
        kind = raw.get("kind")
        if kind == "coverage":
            if "cx" in raw and "cy" in raw:
                return CoverageCell.model_validate(raw)
            return CoverageCell(
                cx=int(float(raw["x"]) // self.cell_m),
                cy=int(float(raw["y"]) // self.cell_m),
                last_seen_t=float(raw.get("last_seen_t", raw.get("last_seen", 0.0))),
                by=str(raw.get("by", "unknown")),
            )
        if kind == "claim":
            return Claim(
                task_id=str(raw.get("task_id", raw.get("key"))),
                agent_id=str(raw.get("agent_id", raw.get("value"))),
                t=float(raw.get("t", 0.0)),
            )
        if kind == "evidence":
            return Evidence(
                evidence_id=str(raw.get("evidence_id", raw.get("key"))),
                object_id=str(raw.get("object_id", "unknown")),
                agent_id=str(raw.get("agent_id", "unknown")),
                t=float(raw.get("t", 0.0)),
                score=float(raw.get("score", 0.0)),
                x=float(raw.get("x", 0.0)),
                y=float(raw.get("y", 0.0)),
            )
        raise ValueError(f"memory item needs kind in (coverage, claim, evidence), got {kind!r}")

    def records(
        self, kind: str, region: Region | None = None, now: float | None = None
    ) -> list[dict[str, Any]]:
        """Query as plain dicts with an `age` field, for skills that print entries with their age."""
        out = []
        for it in self.query(kind, region):
            d = it.model_dump()
            t = d.get("last_seen_t", d.get("t", 0.0))
            d["age"] = (now - t) if now is not None else 0.0
            pos = self._position(it)
            if isinstance(it, CoverageCell) and pos is not None:
                d["x"], d["y"] = pos
            out.append(d)
        return out

    def merge(self, delta: bytes) -> None:
        with self._lock:
            for line in delta.decode().splitlines():
                if line.strip():
                    self._apply(item_adapter.validate_json(line))

    def delta(self, since_version: int, byte_budget: int) -> bytes:
        """Entries changed after since_version, newest first, as JSONL cut to fit the budget."""
        with self._lock:
            return self._delta_locked(since_version, byte_budget)

    def _delta_locked(self, since_version: int, byte_budget: int) -> bytes:
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
        with self._lock:
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
