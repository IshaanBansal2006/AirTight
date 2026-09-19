"""Live capture loop: park the Go2, place the person, run the detector.

The live path needs a running `unitree-go2-agentic` sim. Offline, call
`record_synthetic_sweep` and fit from the cache.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from airtight.dimos_lane.calibration import (
    BEARINGS_DEG,
    QUERY,
    RANGE_BINS_M,
    LookRecord,
    append_look,
    image_bytes_hash,
    pose_for_look,
)


def detect_image(
    image: Any, query: str = QUERY, detector_name: str = "owlv2"
) -> tuple[bool, int, str]:
    """Run a dimOS open-vocab detector. Prefer OWLv2 (local, cacheable)."""
    if detector_name == "owlv2":
        from dimos.perception.detection.detectors.owlv2 import Owlv2Detector

        det = Owlv2Detector()
        result = det.query_detections(image, queries=[query], threshold=0.1)
        n = len(result.detections)
        return n > 0, n, "owlv2"
    from dimos.models.vl.create import create

    model = create(detector_name)
    result = model.query_detections(image, query)
    n = len(result.detections)
    return n > 0, n, detector_name


def capture_look(
    image: Any,
    range_m: float,
    bearing_deg: float,
    cls: str,
    cache: Path,
    *,
    detector_name: str = "owlv2",
    query: str = QUERY,
) -> LookRecord:
    hit, n_boxes, detector = detect_image(image, query=query, detector_name=detector_name)
    raw = getattr(image, "data", None)
    digest = image_bytes_hash(bytes(raw) if raw is not None else repr(image).encode())
    record = LookRecord(
        range_m=range_m,
        bearing_deg=bearing_deg,
        cls=cls,
        hit=hit,
        image_hash=digest,
        n_boxes=n_boxes,
        detector=detector,
    )
    append_look(cache, record)
    return record


def planned_poses(
    go2_xy: tuple[float, float] = (10.0, 70.0),
) -> list[tuple[float, float, float, float]]:
    """(range_m, bearing_deg, person_x, person_y) for the person sweep."""
    out: list[tuple[float, float, float, float]] = []
    for range_m in RANGE_BINS_M:
        for bearing in BEARINGS_DEG:
            x, y = pose_for_look(range_m, bearing, go2_xy)
            out.append((range_m, bearing, x, y))
    return out
