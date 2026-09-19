"""Fleet-memory module. Wraps C's FleetMemoryStore; LocalFleetMemory is the import fallback."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from dimos.agents.annotation import skill
from dimos.core.module import Module

if TYPE_CHECKING:
    from airtight.memory.interface import FleetMemory


def _cell(x: float, y: float, res: float = 2.0) -> tuple[int, int]:
    return int(x // res), int(y // res)


class LocalFleetMemory:
    """Order-independent, idempotent replica: coverage=max, claims=newest, evidence=set."""

    def __init__(self) -> None:
        self._coverage: dict[tuple[int, int], float] = {}
        self._claims: dict[str, tuple[float, Any]] = {}
        self._evidence: dict[str, float] = {}
        self._version = 0

    def observe(self, item: Any) -> None:
        kind = item.get("kind") if isinstance(item, dict) else getattr(item, "kind", None)
        if kind == "coverage":
            x, y = float(item["x"]), float(item["y"])
            seen = float(item.get("last_seen", item.get("last_seen_t", time.time())))
            cell = _cell(x, y)
            self._coverage[cell] = max(self._coverage.get(cell, 0.0), seen)
        elif kind == "claim":
            key = str(item.get("key", item.get("task_id")))
            ts = float(item.get("t", time.time()))
            prev = self._claims.get(key)
            if prev is None or ts >= prev[0]:
                self._claims[key] = (ts, item.get("value", item.get("agent_id")))
        elif kind == "evidence":
            key = str(item.get("key", item.get("evidence_id")))
            self._evidence[key] = float(item.get("score", 0.0))
        self._version += 1

    def delta(self, since_version: int, byte_budget: int) -> bytes:
        payload = {
            "coverage": [[k[0], k[1], v] for k, v in self._coverage.items()],
            "claims": {k: [ts, val] for k, (ts, val) in self._claims.items()},
            "evidence": dict(self._evidence),
            "version": self._version,
            "since": since_version,
        }
        blob = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        if len(blob) <= byte_budget:
            return blob
        # newest-first: drop coverage cells with oldest last-seen
        ranked = sorted(self._coverage.items(), key=lambda kv: kv[1], reverse=True)
        for n in range(len(ranked), -1, -1):
            payload["coverage"] = [[k[0], k[1], v] for k, v in ranked[:n]]
            blob = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
            if len(blob) <= byte_budget:
                return blob
        return blob[:byte_budget]

    def merge(self, delta: bytes) -> None:
        if not delta:
            return
        payload = json.loads(delta.decode())
        for ix, iy, seen in payload.get("coverage", []):
            cell = (int(ix), int(iy))
            self._coverage[cell] = max(self._coverage.get(cell, 0.0), float(seen))
        for key, (ts, val) in payload.get("claims", {}).items():
            prev = self._claims.get(key)
            if prev is None or float(ts) >= prev[0]:
                self._claims[key] = (float(ts), val)
        for key, score in payload.get("evidence", {}).items():
            self._evidence[key] = float(score)
        self._version = max(self._version, int(payload.get("version", self._version))) + 1

    def query(
        self, kind: str, region: tuple[float, float, float, float] | None = None
    ) -> list[Any]:
        now = time.time()
        if kind == "coverage":
            items = []
            for (ix, iy), seen in self._coverage.items():
                x, y = ix * 2.0, iy * 2.0
                if region and not (region[0] <= x <= region[2] and region[1] <= y <= region[3]):
                    continue
                items.append({"x": x, "y": y, "last_seen": seen, "age": now - seen})
            return items
        if kind == "claim":
            return [
                {"key": k, "value": val, "t": ts, "age": now - ts}
                for k, (ts, val) in self._claims.items()
            ]
        if kind == "evidence":
            return [{"key": k, "score": s, "age": 0.0} for k, s in sorted(self._evidence.items())]
        return []

    @property
    def version(self) -> int:
        return self._version

    def evidence_score(self) -> float:
        return float(sum(self._evidence.values()))


def load_fleet_memory() -> FleetMemory:
    """Wrap C's FleetMemoryStore. Local replica only if that class cannot be imported."""
    import airtight.memory as mem

    cls = getattr(mem, "FleetMemoryStore", None)
    if not isinstance(cls, type):
        try:
            from airtight.memory import store as memory_store

            cls = getattr(memory_store, "FleetMemoryStore", None)
        except ImportError:
            cls = None
    if isinstance(cls, type):
        return cls()  # type: ignore[no-any-return]
    skip = {
        "FleetMemory",
        "Claim",
        "CoverageCell",
        "Evidence",
        "MemoryItem",
        "LocalFleetMemory",
    }
    for name in dir(mem):
        if name.startswith("_") or name in skip:
            continue
        candidate = getattr(mem, name)
        if isinstance(candidate, type) and candidate is not LocalFleetMemory:
            try:
                inst = candidate()
            except Exception:
                continue
            if all(
                hasattr(inst, attr) for attr in ("observe", "delta", "merge", "query", "version")
            ):
                return inst  # type: ignore[no-any-return]
    return LocalFleetMemory()


def _as_record(item: Any, now: float) -> dict[str, Any]:
    if isinstance(item, dict):
        rec = dict(item)
    else:
        dump = getattr(item, "model_dump", None)
        rec = dump() if callable(dump) else {"item": str(item)}
    if not isinstance(rec.get("age"), (int, float)):
        stamp = rec.get("last_seen", rec.get("last_seen_t", rec.get("t")))
        try:
            rec["age"] = now - float(stamp)
        except (TypeError, ValueError):
            rec["age"] = 0.0
    return rec


def format_recall(store: Any, query: str, kind: str = "coverage") -> str:
    """Render `query` results with age. Used by the module and by WalkModule."""
    region = None
    now = time.time()
    if query.strip() and query.strip() != "*":
        parts = [p.strip() for p in query.split(",")]
        if len(parts) == 4:
            region = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    records = getattr(store, "records", None)
    items = records(kind, region, now=now) if callable(records) else store.query(kind, region)
    if not items:
        return f"kind={kind} empty"
    rendered = []
    for item in items[:12]:
        rec = _as_record(item, now)
        age = rec.get("age")
        rendered.append(f"{rec} age={age:.1f}s" if isinstance(age, (int, float)) else str(rec))
    return "; ".join(rendered)


def remember_item(
    store: Any,
    *,
    kind: str = "coverage",
    x: float = 0.0,
    y: float = 0.0,
    key: str = "",
    value: str = "",
    t: float | None = None,
) -> None:
    now = time.time() if t is None else t
    if kind == "coverage":
        # Dict shape C's FleetMemoryStore.coerce accepts (x/y metres + last_seen).
        store.observe(
            {
                "kind": "coverage",
                "x": x,
                "y": y,
                "last_seen": now,
                "last_seen_t": now,
                "by": key or "unknown",
            }
        )
    elif kind == "claim":
        store.observe(
            {
                "kind": "claim",
                "key": key or "claim",
                "task_id": key or "claim",
                "value": value,
                "agent_id": value,
                "t": now,
            }
        )
    else:
        store.observe({"kind": "evidence", "key": key or "ev", "score": 1.0})


class FleetMemoryModule(Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.inner = load_fleet_memory()

    @skill
    def remember(
        self,
        kind: str = "coverage",
        x: float = 0.0,
        y: float = 0.0,
        key: str = "",
        value: str = "",
    ) -> str:
        """Record a coverage cell, claim, or evidence item."""
        remember_item(self.inner, kind=kind, x=x, y=y, key=key, value=value)
        return format_recall(self.inner, "*", kind)

    @skill
    def recall(self, query: str = "*", kind: str = "coverage") -> str:
        """Return fleet-memory entries for `kind`, each with age in seconds."""
        return format_recall(self.inner, query, kind)
