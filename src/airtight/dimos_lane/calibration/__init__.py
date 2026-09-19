"""Detection look cache and live capture against the Go2 camera."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from airtight.dimos_lane.site_io import load_example_site

if TYPE_CHECKING:
    from pathlib import Path

RANGE_BINS_M = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 30.0]
BEARINGS_DEG = [-30.0, 0.0, 30.0]
FRAMES_PER_CELL = 30
QUERY = "person"


@dataclass(frozen=True)
class LookRecord:
    range_m: float
    bearing_deg: float
    cls: str
    hit: bool
    image_hash: str
    n_boxes: int
    detector: str


def looks_path(root: Path) -> Path:
    return root / "calibration_looks.jsonl"


def append_look(path: Path, record: LookRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(asdict(record)) + "\n")


def read_looks(path: Path) -> list[LookRecord]:
    if not path.exists():
        return []
    out: list[LookRecord] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        out.append(LookRecord(**raw))
    return out


def image_bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def pose_for_look(
    range_m: float, bearing_deg: float, go2_xy: tuple[float, float] = (10.0, 70.0)
) -> tuple[float, float]:
    """Person standing `range_m` in front of a Go2 that faces +x, then yawed by bearing."""
    from math import cos, radians, sin

    yaw = radians(bearing_deg)
    gx, gy = go2_xy
    return gx + range_m * cos(yaw), gy + range_m * sin(yaw)


def synthetic_hit(range_m: float, cls: str, rng: Any) -> bool:
    """Fallback look generator when the live detector is not on this box.

    Logistic Pd for a person, low Pfa for benign classes. Used to keep the
    fit pipeline testable and to fill the hour-9 hand-off if capture is short.
    """
    import math

    if cls == "person":
        pd = 1.0 / (1.0 + math.exp(0.35 * (range_m - 14.0)))
        return bool(rng.random() < pd)
    pfa = 0.03 if cls == "vehicle" else 0.02
    return bool(rng.random() < pfa)


def record_synthetic_sweep(
    path: Path,
    *,
    seed: int = 0,
    frames_per_cell: int = FRAMES_PER_CELL,
    detector: str = "synthetic-logistic",
) -> list[LookRecord]:
    """Write a cached sweep covering every range bin, bearing, and class."""
    import random

    rng = random.Random(seed)
    site = load_example_site()
    classes = sorted({route.cls for route in site.benign_routes} | {"person"})
    records: list[LookRecord] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    for cls in classes:
        ranges = RANGE_BINS_M if cls == "person" else RANGE_BINS_M[:4]
        bearings = BEARINGS_DEG if cls == "person" else [0.0]
        for range_m in ranges:
            for bearing in bearings:
                for i in range(frames_per_cell):
                    hit = synthetic_hit(range_m, cls, rng)
                    rec = LookRecord(
                        range_m=range_m,
                        bearing_deg=bearing,
                        cls=cls,
                        hit=hit,
                        image_hash=f"{cls}:{range_m}:{bearing}:{i}",
                        n_boxes=int(hit),
                        detector=detector,
                    )
                    append_look(path, rec)
                    records.append(rec)
    return records
