from __future__ import annotations

from airtight.dimos_lane.calibration import RANGE_BINS_M, record_synthetic_sweep
from airtight.dimos_lane.calibration.fit import curves_from_looks, fit_cache, fit_logistic


def test_fit_writes_valid_sensor_curves(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cache = tmp_path / "looks.jsonl"
    record_synthetic_sweep(cache, seed=1, frames_per_cell=20)
    out = tmp_path / "sensor_curve.json"
    curves = fit_cache(cache, out)
    assert out.exists()
    assert set(curves.curves) == {"go2_camera", "drone_camera", "human_eye", "fixed_camera"}
    go2 = curves.curves["go2_camera"]
    assert go2.range_bins_m == RANGE_BINS_M
    assert go2.pd_per_look[0] > go2.pd_per_look[-1]
    assert "Go2 camera only" in curves.source
    assert len(curves.content_hash()) == 12


def test_logistic_recovers_decreasing_pd() -> None:
    bins = [2.0, 8.0, 16.0, 30.0]
    pd = [0.95, 0.8, 0.4, 0.1]
    k, r0 = fit_logistic(bins, pd)
    assert k > 0
    assert 2.0 < r0 < 30.0


def test_curves_from_looks_rescales_drone_camera(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cache = tmp_path / "looks.jsonl"
    looks = record_synthetic_sweep(cache, seed=2, frames_per_cell=15)
    curves = curves_from_looks(looks)
    drone = curves.curves["drone_camera"]
    assert drone.pd_per_look[0] >= drone.pd_per_look[-1]
    assert "person" in drone.pfa_per_look_by_class
