"""Controller trace and perceived map for the console's Perception page: what the patrol wants and what the swarm believes.

Two things the episode log does not carry are rebuilt here for one recorded episode.

The controller trace (targets, modes, headings, staleness, patrol weight, candidate sets) comes
from re-running the same episode with a probe. This reaches into lane B's PRIVATE internals
(`airtight.sim.episode._run_loop`, `_start_jitter_s`, `airtight.sim.actors`, `airtight.sim.fleet`,
`airtight.sim.adapt`, `airtight.sim.geometry.voronoi_mask`), none of which `airtight.sim.__init__`
lists as public. All of that use is isolated in `episode_trace`. Lane B should expose a public
episode-trace function (same arguments as `simulate`, returning per-step agent and controller
state); when it does, `episode_trace` should become a thin call to it and the private imports
should go. `check_trace_against_log` proves the re-run is the recorded episode: probe positions
must equal the logged positions to 1e-6.

The perceived map is dimOS's column-carving voxel map. `ColumnMap` feeds one synthesised sensed
frame per second into `dimos.mapping.voxels.impl.packed.PackedVoxels(voxel_size, carve_columns=True)`:
adding a frame deletes every stored voxel in any (x, y) column the frame touches and then inserts
the frame's voxels, so an object that has moved on is erased the next time its old column is swept
and survives as a ghost until then. The carving is dimOS's; this module only keeps a last-touched
time per column next to it. `dimos.mapping.voxels.grid.VoxelGrid` is not used because it imports
open3d, which does not load in this environment (missing libusb). For the same reason
`PointCloud2.from_numpy` cannot build a frame here, so `_frame` falls back to a minimal stand-in
that offers the one method `PackedVoxels.add_frame` reads (`points_f32`); with a working open3d
the real `PointCloud2` is used.

The map is carved once per SOURCE: "drones" feeds only the active drones' footprints, "swarm" feeds
every active agent and the fixed cameras. Both runs start at the beginning of the warm-up, so the
sweep history at t = 0 is each source's own. The warm-up poses are not offered by the probe (it is
only called for t >= 0), so `episode_trace` swaps `airtight.sim.episode.PatrolController` for a
recording subclass for the length of the re-run; the subclass only reads and then calls the parent.

Voxels are cubes, so height needs voxels smaller than a person: the voxel is 1.25 m and a display
column is 2 x 2 voxel columns (2.5 m) that are always sensed together, which keeps the payload at
80 x 48 columns for the yard.
"""

from __future__ import annotations

import base64
import inspect
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic

VOXEL_M = 1.25
COLUMN_VOXELS = 2
OBJECT_HEIGHT_M = {"vehicle": 2.5}
DEFAULT_HEIGHT_M = 1.8
MODES = ("patrol", "returning", "charging")
NEVER = 255
SOURCES = ("drones", "swarm")
GROUPS = ("drone", "ground", "fixed")

_open3d_ok: bool | None = None


class _Frame:
    """The one method PackedVoxels.add_frame reads from a PointCloud2."""

    def __init__(self, points: np.ndarray) -> None:
        self._points = points.astype(np.float32)

    def points_f32(self) -> np.ndarray:
        return self._points


def _frame(points: np.ndarray) -> Any:
    global _open3d_ok
    if _open3d_ok is not False:
        try:
            from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

            frame = PointCloud2.from_numpy(points.astype(np.float32))
            _open3d_ok = True
            return frame
        except (ImportError, OSError):
            _open3d_ok = False
    return _Frame(points)


def lattice(bounds: Sequence[float], size_m: float) -> np.ndarray:
    """(rows, cols, 2) cell centres as (x, y); the far edge rounds up, as in sim.geometry.Grid."""
    xmin, ymin, xmax, ymax = bounds
    rows = max(1, math.ceil((ymax - ymin) / size_m - 1e-9))
    cols = max(1, math.ceil((xmax - xmin) / size_m - 1e-9))
    xs = xmin + (np.arange(cols) + 0.5) * size_m
    ys = ymin + (np.arange(rows) + 0.5) * size_m
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy], axis=-1)


def footprint_mask(
    centres: np.ndarray, pos: Sequence[float], heading: float, fov_deg: float, radius_m: float
) -> np.ndarray:
    """Cells whose centre is inside the observer's range disc and field-of-view wedge."""
    dx = centres[..., 0] - pos[0]
    dy = centres[..., 1] - pos[1]
    inside = np.hypot(dx, dy) <= radius_m
    if fov_deg < 360.0:
        off = np.mod(np.arctan2(dy, dx) - heading + math.pi, 2.0 * math.pi) - math.pi
        inside &= (np.abs(off) <= math.radians(fov_deg) / 2.0) | ((dx == 0) & (dy == 0))
    return inside


def candidate_set(
    centres: np.ndarray,
    weight: np.ndarray,
    staleness: np.ndarray,
    region: np.ndarray,
    pos: Sequence[float],
    d0_m: float,
    top_fraction: float,
) -> tuple[np.ndarray, bool]:
    """The cells PatrolController.retarget would draw from now: flat indices, and whether it fell back to the whole grid.

    utility = staleness x weight / (1 + distance / d0_m) inside the agent's Voronoi region, the
    top `top_fraction` of the positive cells by a stable sort, exactly as in sim/fleet.py.
    """
    dist = np.hypot(centres[..., 0] - pos[0], centres[..., 1] - pos[1])
    base = staleness * weight / (1.0 + dist / d0_m)
    utility = base * region
    fell_back = not np.any(utility > 0)
    if fell_back:
        utility = base
    flat = utility.ravel()
    positive = np.flatnonzero(flat > 0)
    if not len(positive):
        return positive, fell_back
    k = max(1, int(top_fraction * len(positive)))
    return positive[np.argsort(-flat[positive], kind="stable")][:k], fell_back


@dataclass
class Sensed:
    """One observer at one instant, as the map needs it."""

    pos: tuple[float, float]
    heading: float
    fov_deg: float
    radius_m: float


class ColumnMap:
    """The swarm's shared map: dimOS PackedVoxels with column carving, plus a last-touched time per display column."""

    def __init__(self, bounds: Sequence[float], voxel_m: float = VOXEL_M) -> None:
        from dimos.mapping.voxels.impl.packed import PackedVoxels

        self.bounds = tuple(float(b) for b in bounds)
        self.voxel_m = voxel_m
        self.column_m = voxel_m * COLUMN_VOXELS
        self.centres = lattice(self.bounds, self.column_m)
        self.rows, self.cols = self.centres.shape[:2]
        self.last_touched = np.full((self.rows, self.cols), np.nan)
        self.made_by: dict[tuple[int, int], str] = {}
        self._voxels = PackedVoxels(voxel_m, carve_columns=True)
        offs = (np.arange(COLUMN_VOXELS) + 0.5) * voxel_m - self.column_m / 2.0
        self._sub = np.array([[ox, oy] for oy in offs for ox in offs])

    def column_of(self, x: float, y: float) -> tuple[int, int] | None:
        col = int((x - self.bounds[0]) // self.column_m)
        row = int((y - self.bounds[1]) // self.column_m)
        return (row, col) if 0 <= row < self.rows and 0 <= col < self.cols else None

    def footprint(self, observers: Sequence[Sensed]) -> np.ndarray:
        """Display columns whose centre is inside any of the observers' footprints."""
        touched = np.zeros((self.rows, self.cols), dtype=np.bool_)
        for o in observers:
            touched |= footprint_mask(self.centres, o.pos, o.heading, o.fov_deg, o.radius_m)
        return touched

    def sense(
        self,
        t: float,
        touched: np.ndarray,
        objects: Mapping[str, tuple[float, float, float]],
    ) -> list[str]:
        """Feed one frame: ground in every touched column, a stack of points on each object standing in one.

        objects maps id to (x, y, height_m). Returns the ids in view.
        """
        stacks: dict[tuple[int, int], float] = {}
        made_now: dict[tuple[int, int], str] = {}
        in_view = []
        for oid in sorted(objects):
            x, y, height = objects[oid]
            rc = self.column_of(x, y)
            if rc is not None and touched[rc]:
                in_view.append(oid)
                stacks[rc] = max(stacks.get(rc, 0.0), height)
                if oid == "intruder" or rc not in made_now:
                    made_now[rc] = oid
        for rc in [rc for rc in self.made_by if touched[rc]]:
            del self.made_by[rc]
        self.made_by.update(made_now)
        self._add(touched, stacks)
        self.last_touched[touched] = t
        return in_view

    def _add(self, touched: np.ndarray, stacks: Mapping[tuple[int, int], float]) -> None:
        if not touched.any():
            return
        ground = (self.centres[touched][:, None, :] + self._sub[None, :, :]).reshape(-1, 2)
        pts = [np.column_stack([ground, np.zeros(len(ground))])]
        for (row, col), height in stacks.items():
            zs = np.arange(self.voxel_m / 2.0, height + 1e-9, self.voxel_m / 2.0)
            xy = self.centres[row, col] + self._sub
            pts.append(np.column_stack([np.repeat(xy, len(zs), axis=0), np.tile(zs, len(xy))]))
        self._voxels.add_frame(_frame(np.vstack(pts)))

    def levels(self) -> np.ndarray:
        """(rows, cols) int: -1 unknown, 0 free ground, n > 0 occupied up to voxel n. Read back from dimOS."""
        out = np.full((self.rows, self.cols), -1, dtype=np.int64)
        pts = self._voxels.points()
        if not len(pts):
            return out
        col = np.floor((pts[:, 0] - self.bounds[0]) / self.column_m).astype(np.int64)
        row = np.floor((pts[:, 1] - self.bounds[1]) / self.column_m).astype(np.int64)
        lvl = np.floor(pts[:, 2] / self.voxel_m).astype(np.int64)
        np.maximum.at(out, (row, col), lvl)
        return out

    def ghosts(self, t: float, truth: Mapping[str, tuple[float, float, float]]) -> int:
        """Occupied columns not swept at t whose object is no longer there."""
        here = {self.column_of(x, y) for x, y, _ in truth.values()}
        levels = self.levels()
        stale = (levels > 0) & (self.last_touched < t)
        return sum(
            1 for rc in zip(*np.nonzero(stale), strict=True) if (int(rc[0]), int(rc[1])) not in here
        )


@dataclass
class Frame:
    t: float
    agents: dict[str, dict[str, Any]]
    last_seen: np.ndarray
    objects: dict[str, tuple[float, float, float]]
    candidates: dict[str, np.ndarray] = field(default_factory=dict)
    regions: dict[str, np.ndarray] = field(default_factory=dict)
    fell_back: dict[str, bool] = field(default_factory=dict)


@dataclass
class Trace:
    frames: list[Frame]
    weight: np.ndarray
    centres: np.ndarray
    cell_m: float
    never_seen_at: float
    warmup_s: float
    d0_m: float
    top_fraction: float
    agent_meta: dict[str, dict[str, Any]]
    fixed: list[dict[str, Any]]
    classes: dict[str, str]
    warm: list[tuple[float, dict[str, dict[str, Any]]]] = field(default_factory=list)

    def staleness(self, frame: Frame) -> np.ndarray:
        return np.where(self.weight > 0, np.maximum(frame.t - frame.last_seen, 0.0), 0.0)


def episode_trace(
    site: Site,
    fleet: FleetConfig,
    tactic: Tactic,
    curves: SensorCurves,
    seed: int,
    sample_s: float = 1.0,
) -> Trace:
    """Re-run one official episode with a probe, sampled every sample_s. Mirrors sim.episode.simulate.

    The only function here that touches lane B's private internals; see the module docstring.
    """
    from airtight.sim import adapt
    from airtight.sim import episode as episode_module
    from airtight.sim.actors import Intruder, make_decoy, spawn_benign
    from airtight.sim.constants import BENIGN_HORIZON_S, TIME_EPS
    from airtight.sim.episode import _run_loop, _start_jitter_s, check_setup, official_params
    from airtight.sim.fleet import PatrolController
    from airtight.sim.geometry import voronoi_mask
    from airtight.sim.sensing import make_fixed_observers

    params = official_params()
    check_setup(site, fleet, curves, params)
    stale_init_s = float(inspect.signature(PatrolController).parameters["stale_init_s"].default)
    intruder = Intruder(site, tactic)
    decoy = make_decoy(tactic)
    t_end = intruder.t_reach + params.task_time_s + params.tail_s
    benign = spawn_benign(site, 0.0, BENIGN_HORIZON_S, seed)
    objects = [intruder, *([decoy] if decoy is not None else []), *benign]
    t0_abs = adapt.tactic_phase(tactic) * adapt.reference_cycle_s(fleet) + _start_jitter_s(
        seed, params
    )
    classes = {o.object_id: str(o.kind) for o in objects}
    heights = {oid: OBJECT_HEIGHT_M.get(cls, DEFAULT_HEIGHT_M) for oid, cls in classes.items()}
    frames: list[Frame] = []
    shared: dict[str, Any] = {}
    warm: list[tuple[float, dict[str, dict[str, Any]]]] = []
    agent_ids = {spec.id for spec in fleet.agents}

    def on_sample(t: float) -> bool:
        return abs(t / sample_s - round(t / sample_s)) <= 1e-9

    class RecordingController(PatrolController):
        """Reads the warm-up poses the probe never sees, then does exactly what the parent does."""

        def mark_seen(self, observers: Any, t: float) -> None:
            if t < -TIME_EPS and on_sample(t):
                warm.append(
                    (
                        float(round(t / sample_s) * sample_s),
                        {
                            o.agent_id: {
                                "pos": (float(o.pos[0]), float(o.pos[1])),
                                "heading": float(o.heading),
                                "active": bool(o.active),
                            }
                            for o in observers
                            if o.agent_id in agent_ids
                        },
                    )
                )
            super().mark_seen(observers, t)

    def probe(t: float, agents: Any, controller: Any) -> None:
        if not on_sample(t):
            return
        shared.setdefault("controller", controller)
        frame = Frame(
            t=float(round(t / sample_s) * sample_s),
            agents={
                a.agent_id: {
                    "pos": (float(a.pos[0]), float(a.pos[1])),
                    "target": (float(a.target[0]), float(a.target[1])),
                    "heading": float(a.heading),
                    "mode": str(a.mode),
                    "active": bool(a.active),
                }
                for a in agents
            },
            last_seen=controller.last_seen.copy(),
            objects={
                o.object_id: (*map(float, o.position(t)), heights[o.object_id])
                for o in objects
                if o.alive(t)
            },
        )
        centres = controller.grid.cell_centers()
        stale = np.maximum(t - controller.last_seen, 0.0)
        patrolling = sorted((a for a in agents if a.patrolling), key=lambda a: a.index)
        for a in patrolling:
            peers = {b.index: b.pos for b in patrolling if b is not a}
            region = voronoi_mask(centres, a.pos, peers, a.index)
            cells, fell_back = candidate_set(
                centres,
                controller.weight,
                stale,
                region,
                a.pos,
                controller.d0_m,
                controller.top_fraction,
            )
            frame.candidates[a.agent_id] = cells
            frame.regions[a.agent_id] = region
            frame.fell_back[a.agent_id] = fell_back
        frames.append(frame)

    original = episode_module.PatrolController
    episode_module.PatrolController = RecordingController  # type: ignore[misc]
    try:
        _run_loop(site, fleet, curves, seed, params, objects, t_end, t0_abs, probe=probe)
    finally:
        episode_module.PatrolController = original  # type: ignore[misc]

    controller = shared["controller"]
    meta = {}
    for spec in fleet.agents:
        sensor = adapt.agent_sensor_type(fleet, spec.id)
        meta[spec.id] = {
            "type": spec.type,
            "sensor": sensor,
            "range": adapt.sensor_footprint_radius_m(curves, sensor),
            "fov": adapt.sensor_fov_deg(curves, sensor),
        }
    fixed = [
        {
            "id": o.agent_id,
            "pos": (float(o.pos[0]), float(o.pos[1])),
            "heading": float(o.heading),
            "range": float(o.footprint_radius_m),
            "fov": float(o.fov_deg),
        }
        for o in make_fixed_observers(site, curves)
    ]
    return Trace(
        frames=frames,
        weight=controller.weight.copy(),
        centres=np.asarray(controller.grid.cell_centers()),
        cell_m=float(controller.grid.cell_size),
        never_seen_at=-math.ceil(params.warmup_s / params.dt - TIME_EPS) * params.dt - stale_init_s,
        warmup_s=float(params.warmup_s),
        d0_m=float(controller.d0_m),
        top_fraction=float(controller.top_fraction),
        agent_meta=meta,
        fixed=fixed,
        classes=classes,
        warm=warm,
    )


def check_trace_against_log(
    trace: Trace, tracks: Mapping[str, Sequence[Sequence[float]]], tol: float = 1e-6
) -> int:
    """Every sampled agent position must equal the logged one. Returns how many were compared."""
    logged = {
        (oid, round(float(p[0]), 6)): (float(p[1]), float(p[2]))
        for oid, tr in tracks.items()
        for p in tr
    }
    n = 0
    for frame in trace.frames:
        for aid, a in frame.agents.items():
            hit = logged.get((aid, round(frame.t, 6)))
            if hit is None:
                raise ValueError(f"the log has no position for {aid} at t = {frame.t}")
            if math.hypot(hit[0] - a["pos"][0], hit[1] - a["pos"][1]) > tol:
                raise ValueError(
                    f"trace and log disagree for {aid} at t = {frame.t}: {a['pos']} vs {hit}; the re-run is not the recorded episode"
                )
            n += 1
    if not n:
        raise ValueError("nothing to compare: the trace has no frames")
    return n


def _sensed(trace: Trace, agents: Mapping[str, Mapping[str, Any]], group: str) -> list[Sensed]:
    """The active observers of one group: "drone", "ground" (Go2 and guard) or "fixed"."""
    if group == "fixed":
        return [Sensed(f["pos"], f["heading"], f["fov"], f["range"]) for f in trace.fixed]
    return [
        Sensed(a["pos"], a["heading"], trace.agent_meta[aid]["fov"], trace.agent_meta[aid]["range"])
        for aid, a in sorted(agents.items())
        if a["active"] and (trace.agent_meta[aid]["type"] == "drone") == (group == "drone")
    ]


def perceive(trace: Trace, bounds: Sequence[float]) -> dict[str, Any]:
    """Run the carving map over the warm-up and the trace, once per source, one sensed frame per sampled second.

    Returns the three footprint masks per recorded frame (they are shared by the sources: the
    swarm's touched set is the union of all three) and, per source, the age of every column just
    before t = 0 and what dimOS holds after each recorded frame.
    """
    maps = {src: ColumnMap(bounds) for src in SOURCES}
    groups_of = {"drones": ("drone",), "swarm": GROUPS}
    probe_map = maps["drones"]
    fixed_mask = probe_map.footprint(_sensed(trace, {}, "fixed"))

    def masks(agents: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
        return {
            "drone": probe_map.footprint(_sensed(trace, agents, "drone")),
            "ground": probe_map.footprint(_sensed(trace, agents, "ground")),
            "fixed": fixed_mask,
        }

    def union(m: Mapping[str, np.ndarray], src: str) -> np.ndarray:
        out = np.zeros_like(fixed_mask)
        for g in groups_of[src]:
            out |= m[g]
        return out

    for t, agents in trace.warm:
        m = masks(agents)
        for src, cmap in maps.items():
            cmap.sense(t, union(m, src), {})
    t0 = trace.frames[0].t
    out: dict[str, Any] = {
        "touch": {"drone": [], "ground": [], "fixed": fixed_mask},
        "src": {
            src: {
                "age0": np.where(
                    np.isnan(cmap.last_touched),
                    NEVER,
                    np.clip(t0 - cmap.last_touched, 0, NEVER - 1),
                ),
                "occ": [],
                "vis": [],
                "ghosts": [],
            }
            for src, cmap in maps.items()
        },
    }
    for frame in trace.frames:
        m = masks(frame.agents)
        out["touch"]["drone"].append(m["drone"])
        out["touch"]["ground"].append(m["ground"])
        for src, cmap in maps.items():
            in_view = cmap.sense(frame.t, union(m, src), frame.objects)
            levels = cmap.levels().ravel()
            rec = out["src"][src]
            rec["occ"].append(
                [
                    [
                        int(i),
                        int(levels[i]),
                        int(cmap.made_by.get(divmod(int(i), cmap.cols), "") == "intruder"),
                    ]
                    for i in np.flatnonzero(levels > 0)
                ]
            )
            rec["vis"].append(in_view)
            rec["ghosts"].append(cmap.ghosts(frame.t, frame.objects))
    return out


def pack_bits(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.ravel().astype(np.uint8)).tobytes()).decode()


def pack_bytes(values: np.ndarray) -> str:
    return base64.b64encode(np.clip(np.rint(values), 0, 255).astype(np.uint8).tobytes()).decode()


def pack_runs(mask: np.ndarray) -> str:
    """Runs of set cells in row-major order as little-endian uint16 pairs (start, length)."""
    flat = np.concatenate([[0], mask.ravel().astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(flat))
    runs = np.column_stack([edges[0::2], edges[1::2] - edges[0::2]])
    if runs.size and runs.max() > 65535:
        raise ValueError("the map is too large for 16-bit runs")
    return base64.b64encode(runs.astype("<u2").tobytes()).decode()


def unpack_runs(text: str, shape: tuple[int, int]) -> np.ndarray:
    runs = np.frombuffer(base64.b64decode(text), dtype="<u2").reshape(-1, 2)
    flat = np.zeros(shape[0] * shape[1], dtype=np.bool_)
    for start, length in runs:
        flat[int(start) : int(start) + int(length)] = True
    return flat.reshape(shape)


def _changes(values: Sequence[Any]) -> list[list[Any]]:
    """[[first frame, *value], ...]: a value is repeated until the next entry."""
    out: list[list[Any]] = []
    for i, v in enumerate(values):
        if not out or out[-1][1:] != list(v):
            out.append([i, *v])
    return out


def encode(trace: Trace, perceived: Mapping[str, Any]) -> dict:
    """The compact payload the page reads. Times are whole seconds; staleness is exact to 1 s."""
    frames = trace.frames
    rows, cols = trace.weight.shape
    wmax = float(trace.weight.max()) or 1.0
    first = frames[0]
    never = first.last_seen <= trace.never_seen_at + 1e-6
    age0 = np.where(never, NEVER, np.clip(np.rint(first.t - first.last_seen), 0, NEVER - 1))
    swept = [np.zeros((rows, cols), dtype=np.bool_)] + [
        b.last_seen > a.last_seen for a, b in zip(frames, frames[1:], strict=False)
    ]
    priority = np.stack([trace.staleness(f) * trace.weight for f in frames])
    live = priority[:, trace.weight > 0]
    pmax = float(max(10.0, math.ceil(np.percentile(live, 99) / 10.0) * 10.0)) if live.size else 10.0
    agents = {}
    for aid, meta in trace.agent_meta.items():
        states = [f.agents[aid] for f in frames]
        agents[aid] = {
            "type": meta["type"],
            "range": round(float(meta["range"]), 2),
            "fov": round(float(meta["fov"]), 1),
            "tg": _changes([[round(s["target"][0], 1), round(s["target"][1], 1)] for s in states]),
            "md": _changes([[MODES.index(s["mode"]) if s["mode"] in MODES else 0] for s in states]),
            "hd": [round(math.degrees(s["heading"])) % 360 for s in states],
            "cand": _changes([[[int(c) for c in f.candidates.get(aid, ())]] for f in frames]),
        }
    touch = perceived["touch"]
    shape = touch["fixed"].shape
    return {
        "dt": frames[1].t - frames[0].t if len(frames) > 1 else 1.0,
        "n": len(frames),
        "cell": trace.cell_m,
        "rows": rows,
        "cols": cols,
        "w": pack_bytes(trace.weight / wmax * 255.0),
        "wmax": round(wmax, 3),
        "age0": pack_bytes(age0),
        "never0": round(first.t - trace.never_seen_at, 2),
        "warm": round(trace.warmup_s),
        "swept": [pack_runs(m) for m in swept],
        "pmax": pmax,
        "d0": trace.d0_m,
        "top": trace.top_fraction,
        "agents": agents,
        "classes": {
            oid: cls for oid, cls in trace.classes.items() if any(oid in f.objects for f in frames)
        },
        "pm": {
            "col": VOXEL_M * COLUMN_VOXELS,
            "vox": VOXEL_M,
            "rows": int(shape[0]),
            "cols": int(shape[1]),
            "real_frames": bool(_open3d_ok),
            "touch": {
                "drone": [pack_runs(m) for m in touch["drone"]],
                "ground": [pack_runs(m) for m in touch["ground"]],
                "fixed": pack_runs(touch["fixed"]),
            },
            "src": {
                src: {
                    "age0": pack_bytes(rec["age0"]),
                    "occ": _changes([[o] for o in rec["occ"]]),
                    "vis": _changes([[v] for v in rec["vis"]]),
                    "ghosts": rec["ghosts"],
                }
                for src, rec in perceived["src"].items()
            },
        },
    }


def build_view(log_path: Path, fleet: FleetConfig, site: Site, curves: SensorCurves) -> dict | None:
    """Trace plus perceived maps for one recorded episode, or None when the log is not a v0 full log or dimOS is missing."""
    from airtight.contracts import PositionEvent, read_episode_log

    header, events = read_episode_log(log_path)
    tracks: dict[str, list[tuple[float, float, float]]] = {}
    for ev in events:
        if isinstance(ev, PositionEvent):
            tracks.setdefault(ev.object_id, []).append((ev.t, ev.position.x, ev.position.y))
    if not tracks or fleet.content_hash() != header.fleet_hash:
        return None
    try:
        import dimos.mapping.voxels.impl.packed  # noqa: F401
    except ImportError:
        return None
    trace = episode_trace(site, fleet, header.tactic, curves, header.seed)
    check_trace_against_log(trace, tracks)
    return encode(trace, perceive(trace, site.bounds))
