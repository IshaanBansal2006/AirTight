"""Lane C: fleet memory library. A state-based CRDT with a byte-budgeted delta for degraded links."""

from airtight.memory.interface import FleetMemory
from airtight.memory.items import Claim, CoverageCell, Evidence, MemoryItem
from airtight.memory.store import FleetMemoryStore

__all__ = ["Claim", "CoverageCell", "Evidence", "FleetMemory", "FleetMemoryStore", "MemoryItem"]
