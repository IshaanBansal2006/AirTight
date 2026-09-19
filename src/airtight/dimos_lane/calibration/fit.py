"""Fit SensorCurves from cached looks. Only go2_camera is measured."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from airtight.contracts.sensor_curve import SensorCurve, SensorCurves
from airtight.dimos_lane.calibration import RANGE_BINS_M, LookRecord, read_looks
from airtight.dimos_lane.site_io import load_stub_sensor_curves

DRONE_RANGE_STRETCH = 1.2
EPS = 1e-3


def _bin_index(range_m: float, bins: list[float]) -> int:
    for i, edge in enumerate(bins):
        if range_m <= edge + 1e-9:
            return i
    return len(bins) - 1


def empirical_pd(looks: list[LookRecord], cls: str, bins: list[float]) -> list[float]:
    hits: dict[int, list[bool]] = defaultdict(list)
    for look in looks:
        if look.cls != cls:
            continue
        hits[_bin_index(look.range_m, bins)].append(look.hit)
    out: list[float] = []
    for i in range(len(bins)):
        trials = hits.get(i, [])
        if not trials:
            out.append(0.0 if i == len(bins) - 1 else 0.5)
            continue
        out.append(sum(trials) / len(trials))
    return out


def empirical_pfa(looks: list[LookRecord], cls: str) -> float:
    trials = [look.hit for look in looks if look.cls == cls]
    if not trials:
        return 0.02
    return sum(trials) / len(trials)


def _logit(p: float) -> float:
    q = min(1 - EPS, max(EPS, p))
    return math.log(q / (1 - q))


def fit_logistic(bins: list[float], pd: list[float]) -> tuple[float, float]:
    """Least-squares logit(pd) = a + b * range. Returns (k, r0) for 1/(1+exp(k*(r-r0)))."""
    xs: list[float] = []
    ys: list[float] = []
    for edge, p in zip(bins, pd, strict=True):
        if p <= EPS or p >= 1 - EPS:
            continue
        xs.append(edge)
        ys.append(_logit(p))
    if len(xs) < 2:
        return 0.35, 14.0
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x < 1e-12:
        return 0.35, 14.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    b = cov / var_x
    a = mean_y - b * mean_x
    # logit = a + b r = -k (r - r0) = -k r + k r0  => k = -b, r0 = a/k
    k = -b
    if abs(k) < 1e-9:
        return 0.35, 14.0
    r0 = a / k
    return k, r0


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _interpolate(src_bins: list[float], src_pd: list[float], query_m: float) -> float:
    if query_m <= src_bins[0]:
        return src_pd[0]
    if query_m >= src_bins[-1]:
        return src_pd[-1]
    for i in range(1, len(src_bins)):
        if query_m <= src_bins[i]:
            span = src_bins[i] - src_bins[i - 1]
            t = 0.0 if span == 0 else (query_m - src_bins[i - 1]) / span
            return src_pd[i - 1] + t * (src_pd[i] - src_pd[i - 1])
    return src_pd[-1]


def curves_from_looks(looks: list[LookRecord]) -> SensorCurves:
    stub = load_stub_sensor_curves()
    go2_stub = stub.curves["go2_camera"]
    person_pd = [_clamp01(p) for p in empirical_pd(looks, "person", RANGE_BINS_M)]
    k, r0 = fit_logistic(RANGE_BINS_M, person_pd)
    go2_pd = person_pd

    pfa_by_class = dict(go2_stub.pfa_per_look_by_class)
    for cls in {lk.cls for lk in looks if lk.cls != "person"}:
        pfa_by_class[cls] = _clamp01(empirical_pfa(looks, cls))

    detectors = sorted({lk.detector for lk in looks})
    n_person = sum(1 for lk in looks if lk.cls == "person")
    source = (
        f"Go2 camera only, {', '.join(detectors) or 'unknown'} detector, "
        f"{n_person} person looks across {len(RANGE_BINS_M)} range bins x 3 bearings; "
        f"logistic k={k:.3f} r0={r0:.1f} m; drone_camera is the same curve at "
        f"{DRONE_RANGE_STRETCH}x slant-range; human_eye and fixed_camera stay parametric."
    )

    go2 = SensorCurve(
        sensor_type="go2_camera",
        range_bins_m=list(RANGE_BINS_M),
        pd_per_look=go2_pd,
        pfa_per_look_by_class=pfa_by_class,
        fov_deg=go2_stub.fov_deg,
        look_rate_hz=go2_stub.look_rate_hz,
    )
    drone_stub = stub.curves["drone_camera"]
    drone_pd = [
        _clamp01(_interpolate(RANGE_BINS_M, go2_pd, edge / DRONE_RANGE_STRETCH))
        for edge in drone_stub.range_bins_m
    ]
    drone = drone_stub.model_copy(update={"pd_per_look": drone_pd})

    return SensorCurves(
        source=source,
        curves={
            "go2_camera": go2,
            "drone_camera": drone,
            "human_eye": stub.curves["human_eye"],
            "fixed_camera": stub.curves["fixed_camera"],
        },
    )


def fit_cache(cache: Path, out: Path) -> SensorCurves:
    looks = read_looks(cache)
    if not looks:
        raise FileNotFoundError(f"no looks in {cache}; run a capture or synthetic sweep first")
    curves = curves_from_looks(looks)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(curves.model_dump_json(indent=2) + "\n")
    return curves
