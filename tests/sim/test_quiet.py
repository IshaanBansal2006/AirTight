from __future__ import annotations

import pytest

from airtight.sim import adapt, constants, scenarios
from airtight.sim.episode import EpisodeParams, QuietScores, official_params, simulate_quiet
from airtight.sim.runner import _run_v0


def _world():  # type: ignore[no-untyped-def]
    return scenarios.load_site(), scenarios.load_fleet("2drones"), scenarios.load_sensor_curves()


def test_official_params_are_what_run_episode_uses_and_simulate_default_is_not() -> None:
    assert official_params() == EpisodeParams(battery=True)
    assert EpisodeParams().battery is False
    assert (
        "official_params" in _run_v0.__code__.co_names
    )  # the runner asks, it does not build its own


def test_engine_version_and_investigate_threshold() -> None:
    assert isinstance(constants.ENGINE_VERSION, str) and constants.ENGINE_VERSION
    assert constants.TAU_INVESTIGATE == 1.5
    assert constants.TAU_INVESTIGATE < constants.TAU_REF


def test_quiet_run_is_deterministic_and_reports_its_exposure() -> None:
    site, fleet, curves = _world()
    busy = site.model_copy(
        update={
            "benign_routes": [
                r.model_copy(update={"arrival_rate_per_hour": 120.0}) for r in site.benign_routes
            ]
        }
    )
    a = simulate_quiet(busy, fleet, curves, 1000, duration_s=600.0)
    assert a == simulate_quiet(busy, fleet, curves, 1000, duration_s=600.0)
    assert a != simulate_quiet(busy, fleet, curves, 1001, duration_s=600.0)
    assert isinstance(a, QuietScores) and a.seed == 1000
    assert a.duration_s == 600.0 and a.sim_hours == 600.0 / 3600.0
    assert a.benign_peaks and all(k.split("-")[0] in {"fox", "tarp"} for k in a.benign_peaks)
    assert "intruder" not in a.benign_peaks and "decoy" not in a.benign_peaks


def test_default_duration_is_one_reference_cycle_and_params_are_official() -> None:
    site, fleet, curves = _world()
    default = simulate_quiet(site, fleet, curves, 1000)
    assert default.duration_s == adapt.reference_cycle_s(fleet) == 3600.0
    assert default.sim_hours == 1.0
    assert default == simulate_quiet(site, fleet, curves, 1000, 3600.0, official_params())
    # the params argument is honoured: a fleet that never docks flies a different night
    always_up = simulate_quiet(site, fleet, curves, 1000, params=EpisodeParams(battery=False))
    assert always_up != default and always_up.duration_s == default.duration_s


def test_no_benign_routes_gives_an_empty_result() -> None:
    site, fleet, curves = _world()
    empty = simulate_quiet(site.model_copy(update={"benign_routes": []}), fleet, curves, 1, 300.0)
    assert empty.benign_peaks == {} and empty.sim_hours == 300.0 / 3600.0


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan")])
def test_non_positive_duration_is_rejected(bad: float) -> None:
    site, fleet, curves = _world()
    with pytest.raises(ValueError, match="duration_s must be positive"):
        simulate_quiet(site, fleet, curves, 1, duration_s=bad)


def test_weight_mode_override_changes_official_params_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataclasses

    from airtight.sim.episode import WEIGHT_MODE_ENV

    monkeypatch.delenv(WEIGHT_MODE_ENV, raising=False)
    default = official_params()
    assert default == EpisodeParams(battery=True) and default.weight_mode == "asset"
    monkeypatch.setenv(WEIGHT_MODE_ENV, "")
    assert official_params() == default  # empty counts as unset
    for mode in ("asset", "uniform", "band"):
        monkeypatch.setenv(WEIGHT_MODE_ENV, mode)
        overridden = official_params()
        assert overridden == dataclasses.replace(default, weight_mode=mode)
        changed = {
            k
            for k, v in dataclasses.asdict(overridden).items()
            if v != dataclasses.asdict(default)[k]
        }
        assert changed == (set() if mode == "asset" else {"weight_mode"})
    assert EpisodeParams() == EpisodeParams(battery=False)  # simulate's own default never reads it
    monkeypatch.setenv(WEIGHT_MODE_ENV, "spiral")
    with pytest.raises(ValueError, match="spiral"):
        official_params()
