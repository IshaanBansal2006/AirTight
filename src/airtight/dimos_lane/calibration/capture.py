"""Live capture loop: park the Go2, place the person, run the detector.

The live path talks to a running `airtight.airtight-go2` / `airtight-site`
daemon over MCP for snapshots. OWLv2 stays in this CLI process so the
MuJoCo daemon does not load another GPU model.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from airtight.dimos_lane.calibration import (
    BEARINGS_DEG,
    QUERY,
    RANGE_BINS_M,
    LookRecord,
    append_look,
    image_bytes_hash,
    pose_for_look,
)

if TYPE_CHECKING:
    from pathlib import Path

DetectFn = Callable[..., tuple[bool, int, str]]
GrabFn = Callable[[], Any]
PlaceFn = Callable[[float, float], None]

_OWL: Any | None = None
LIVE_FRAMES_PER_CELL = 3
LIVE_SETTLE_S = 0.5
VEHICLE_RANGES_M = RANGE_BINS_M[:4]


def _owlv2(device: str = "cpu") -> Any:
    global _OWL
    if _OWL is None:
        from dimos.perception.detection.detectors.owlv2 import Owlv2Detector

        _OWL = Owlv2Detector(device=device)
    return _OWL


def detect_image(
    image: Any, query: str = QUERY, detector_name: str = "owlv2"
) -> tuple[bool, int, str]:
    """Run a dimOS open-vocab detector. Prefer OWLv2 (local, cacheable)."""
    if detector_name == "owlv2":
        det = _owlv2()
        result = det.query_detections(image, queries=[query], threshold=0.2)
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
    detect: DetectFn | None = None,
) -> LookRecord:
    fn = detect or detect_image
    hit, n_boxes, detector = fn(image, query=query, detector_name=detector_name)
    raw = getattr(image, "data", None)
    if raw is None:
        digest = image_bytes_hash(repr(image).encode())
    else:
        try:
            digest = image_bytes_hash(bytes(raw))
        except (TypeError, ValueError):
            digest = image_bytes_hash(repr(raw).encode())
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


def grab_frame_via_mcp(path: Path) -> Any:
    """Ask the running WalkModule for a JPEG, then load it as a dimOS Image."""
    from dimos.agents.mcp.mcp_adapter import McpAdapter
    from dimos.msgs.sensor_msgs.Image import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    McpAdapter.from_run_entry(timeout=30).call_tool("snapshot_camera", {"path": str(path)})
    return Image.from_file(path)


def default_place_xy(x: float, y: float) -> None:
    from airtight.dimos_lane.person import INTRUDER_YAW_RAD, publish_person_pose

    publish_person_pose(x, y, z=0.0, yaw_rad=INTRUDER_YAW_RAD)


def run_live_sweep(
    cache: Path,
    *,
    grab_frame: GrabFn,
    place_xy: PlaceFn | None = None,
    detect: DetectFn | None = None,
    frames_per_cell: int = LIVE_FRAMES_PER_CELL,
    settle_s: float = LIVE_SETTLE_S,
    go2_xy: tuple[float, float] = (10.0, 70.0),
    classes: tuple[str, ...] = ("person", "vehicle"),
) -> list[LookRecord]:
    """Place the person, grab frames, detect. Writes `cache` from scratch.

    Person looks measure Pd. Vehicle looks query "vehicle" with the person
    still in view (cross-class false alarms). Detector stays in this process.
    """
    place = place_xy or default_place_xy
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("")
    records: list[LookRecord] = []
    query_for = {"person": "person", "vehicle": "vehicle"}

    def _cell(range_m: float, bearing: float, cls: str) -> None:
        px, py = pose_for_look(range_m, bearing, go2_xy)
        place(px, py)
        if settle_s > 0:
            time.sleep(settle_s)
        for _ in range(frames_per_cell):
            image = grab_frame()
            records.append(
                capture_look(
                    image,
                    range_m,
                    bearing,
                    cls,
                    cache,
                    query=query_for.get(cls, cls),
                    detect=detect,
                )
            )

    if "person" in classes:
        for range_m, bearing, _px, _py in planned_poses(go2_xy):
            _cell(range_m, bearing, "person")
    if "vehicle" in classes:
        for range_m in VEHICLE_RANGES_M:
            _cell(range_m, 0.0, "vehicle")
    return records


def run_live_from_mcp(
    cache: Path,
    snapshot_dir: Path,
    *,
    frames_per_cell: int = LIVE_FRAMES_PER_CELL,
    settle_s: float = LIVE_SETTLE_S,
) -> list[LookRecord]:
    """MCP snapshot_camera + /person_pose + OWLv2 in this process."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    n = 0

    def grab() -> Any:
        nonlocal n
        n += 1
        return grab_frame_via_mcp(snapshot_dir / f"look_{n:04d}.jpg")

    return run_live_sweep(
        cache,
        grab_frame=grab,
        frames_per_cell=frames_per_cell,
        settle_s=settle_s,
    )
