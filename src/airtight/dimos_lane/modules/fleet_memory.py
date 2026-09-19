"""Fleet-memory module. Wraps C's implementation when present, else a local CRDT."""

from __future__ import annotations

import json
import time
from typing import Any

from airtight.memory.interface import FleetMemory
from dimos.agents.annotation import skill
from dimos.core.module import Module


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
            seen = float(item.get("last_seen", time.time()))
            cell = _cell(x, y)
            self._coverage[cell] = max(self._coverage.get(cell, 0.0), seen)
        elif kind == "claim":
            key = str(item["key"])
            ts = float(item.get("t", time.time()))
            prev = self._claims.get(key)
            if prev is None or ts >= prev[0]:
                self._claims[key] = (ts, item.get("value"))
        elif kind == "evidence":
            key = str(item["key"])
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
            return [
                {"key": k, "score": s, "age": 0.0} for k, s in sorted(self._evidence.items())
            ]
        return []

    @property
    def version(self) -> int:
        return self._version

    def evidence_score(self) -> float:
        return float(sum(self._evidence.values()))


def load_fleet_memory() -> FleetMemory:
    """Prefer a concrete class shipped by lane C; otherwise the local replica."""
    import airtight.memory as mem

    for name in dir(mem):
        if name.startswith("_") or name in {"FleetMemory"}:
            continue
        candidate = getattr(mem, name)
        if isinstance(candidate, type) and candidate is not LocalFleetMemory:
            try:
                inst = candidate()
            except Exception:
                continue
            if all(hasattr(inst, attr) for attr in ("observe", "delta", "merge", "query", "version")):
                return inst  # type: ignore[no-any-return]
    return LocalFleetMemory()


class FleetMemoryModule(Module):
    def __init__(self, config_args: dict[str, object] | None = None) -> None:
        super().__init__(dict(config_args or {}))
        self.inner = load_fleet_memory()

    @skill
    def recall(self, query: str, kind: str = "coverage") -> str:
        """Return fleet-memory entries for `kind`, each with age in seconds."""
        region = None
        if query.strip() and query.strip() != "*":
            # "xmin,ymin,xmax,ymax" optional
            parts = [p.strip() for p in query.split(",")]
            if len(parts) == 4:
                region = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
        items = self.inner.query(kind, region)
        if not items:
            return f"kind={kind} empty"
        rendered = []
        for item in items[:12]:
            age = item.get("age") if isinstance(item, dict) else None
            rendered.append(f"{item} age={age:.1f}s" if isinstance(age, float) else str(item))
        return "; ".join(rendered)
