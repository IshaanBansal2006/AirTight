from __future__ import annotations

from typing import Any, Protocol


class FleetMemory(Protocol):
    """Contract between lane C (implements), lane B (calls in the sim) and lane A (wraps as a module)."""

    def observe(self, item: Any) -> None: ...

    def delta(self, since_version: int, byte_budget: int) -> bytes: ...

    def merge(self, delta: bytes) -> None: ...

    def query(
        self, kind: str, region: tuple[float, float, float, float] | None = None
    ) -> list[Any]: ...

    @property
    def version(self) -> int: ...
