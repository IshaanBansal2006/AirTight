from __future__ import annotations

import re
from pathlib import Path

import pytest

from airtight.score import quick
from airtight.score.config_score import ConfigScore, score_config
from airtight.score.quiet import run_quiet
from airtight.score.roc import FLAG_FAR_LIMITED
from airtight.sim import scenarios
from airtight.sim.constants import NEVER_SEEN, TAU_INVESTIGATE
from airtight.sim.coverage import uncovered_intervals
from airtight.sim.episode import EpisodeScores, QuietScores, official_params, simulate_quiet

SEEDS_FILE = Path(__file__).parents[2] / "data" / "seeds.json"


def _episode(seed: int, peak: float, benign: dict[str, float] | None = None) -> EpisodeScores:
    return EpisodeScores(seed, peak, None, benign or {}, None, 0.02, 52.0, 27.0, 62.0, 10)


def _quiet(seed: int, peaks: dict[str, float], hours: float = 1.0) -> QuietScores:
    return QuietScores(seed, peaks, hours, hours * 3600.0)


def test_score_config_on_a_hand_built_case() -> None:
    seeds = [11, 12, 13, 14]
    episodes = {
        "easy": [_episode(s, p) for s, p in zip(seeds, [9.0, 8.0, 7.0, 2.0], strict=True)],
        "hard": [_episode(s, p) for s, p in zip(seeds, [9.0, NEVER_SEEN, 1.0, 2.0], strict=True)],
    }
    quiet = [_quiet(1, {"fox-0": 5.0, "fox-1": 3.0}), _quiet(2, {"tarp-0": 1.6}, hours=2.0)]
    score = score_config(episodes, quiet, n_boot=200)
    assert isinstance(score, ConfigScore)
    # three benign objects in 3 hours: 1.0 per hour already at tau_min, so nothing to filter
    assert (score.tau, score.flag, score.far) == (TAU_INVESTIGATE, FLAG_FAR_LIMITED, 1.0)
    assert score.pd_by_tactic == {"easy": 1.0, "hard": 0.5} and score.pd == 0.75
    assert (score.worst_tactic_id, score.worst_tactic_pd) == ("hard", 0.5)
    assert score.human_decisions_per_hour == 1.0 and score.raw_alerts_per_hour == 1.0
    assert score.n_seeds == 4 and score.quiet_hours == 3.0
    assert score.roc[0].tau == TAU_INVESTIGATE and any(r.tau == score.tau for r in score.roc)
    assert all(r.pd_ci[0] <= r.pd <= r.pd_ci[1] for r in score.roc)


def test_raw_alerts_differ_from_human_decisions_when_the_score_filters() -> None:
    seeds = list(range(8))
    episodes = {"t": [_episode(s, 9.0) for s in seeds]}
    quiet = [_quiet(1, {f"fox-{i}": 2.0 + 0.1 * i for i in range(6)}, hours=2.0)]
    score = score_config(episodes, quiet, n_boot=100)
    assert score.raw_alerts_per_hour == 3.0  # six objects over tau_min in two hours
    assert score.human_decisions_per_hour == 1.0 and score.tau == pytest.approx(2.4)
    assert score.pd == 1.0


def test_misaligned_seeds_are_rejected_by_name() -> None:
    good = [_episode(s, 5.0) for s in (1, 2, 3)]
    quiet = [_quiet(1, {})]
    with pytest.raises(ValueError, match="'shuffled'"):
        score_config({"a": good, "shuffled": [good[1], good[0], good[2]]}, quiet)
    with pytest.raises(ValueError, match="'short'.*2 episodes against 3"):
        score_config({"a": good, "short": good[:2]}, quiet)
    with pytest.raises(ValueError, match="'other'"):
        score_config({"a": good, "other": [_episode(s, 5.0) for s in (1, 2, 4)]}, quiet)
    with pytest.raises(ValueError, match="quiet run"):
        score_config({"a": good}, [])


def test_the_decoy_must_never_appear_among_benign_peaks() -> None:
    quiet = [_quiet(1, {})]
    with pytest.raises(AssertionError, match="decoy"):
        score_config({"a": [_episode(1, 5.0, {"decoy": 9.0})]}, quiet)
    with pytest.raises(AssertionError, match="decoy"):
        score_config({"a": [_episode(1, 5.0)]}, [_quiet(1, {"decoy": 9.0})])


def test_run_quiet_keeps_seed_order_and_matches_serial() -> None:
    site, curves = scenarios.load_site(), scenarios.load_sensor_curves()
    fleet = scenarios.load_fleet("2drones")
    seeds = [1003, 1001, 1002]
    pooled = run_quiet(site, fleet, curves, seeds, workers=2)
    assert [r.seed for r in pooled] == seeds
    assert pooled == [
        simulate_quiet(site, fleet, curves, s, params=official_params()) for s in seeds
    ]
    assert pooled == run_quiet(site, fleet, curves, seeds, workers=1)


def test_end_to_end_the_worst_tactic_is_the_one_inside_the_charging_window() -> None:
    (gap,) = uncovered_intervals(scenarios.load_fleet("2drones"))  # picked from the arithmetic
    inside = round((gap.start_phase + gap.end_phase) / 2.0, 3)
    tactics = quick.phase_variants("yard_night", ["jog"], [0.2, inside])
    assert [t.id for t in tactics] == ["jog@0.2", f"jog@{inside:g}"]
    seeds, quiet_seeds = quick.split_seeds(SEEDS_FILE, 12, 3)
    assert not set(seeds) & set(quiet_seeds)
    score = quick.score_fleet("yard_night", "2drones", tactics, seeds, quiet_seeds, workers=1)
    assert score.worst_tactic_id == f"jog@{inside:g}" and score.worst_tactic_pd == 0.0
    assert score.pd_by_tactic["jog@0.2"] > 0.0
    assert score.n_seeds == 12 and score.quiet_hours == 3.0
    assert "operating point" in quick.format_score("2drones", score)


def test_split_seeds_refuses_to_overlap() -> None:
    with pytest.raises(ValueError, match="fewer than"):
        quick.split_seeds(SEEDS_FILE, 190, 20)


def test_score_imports_only_what_sim_allows() -> None:
    allowed = {
        "airtight.sim.episode": {
            "simulate",
            "simulate_quiet",
            "EpisodeScores",
            "QuietScores",
            "EpisodeParams",
            "official_params",
        },
        "airtight.sim.coverage": {
            "coverage_profile",
            "uncovered_intervals",
            "uncovered_s_per_hour",
        },
    }
    whole_modules = {"airtight.sim.constants", "airtight.sim.scenarios"}
    score_dir = Path(__file__).parents[2] / "src" / "airtight" / "score"
    found = 0
    for path in sorted(score_dir.glob("*.py")):
        text = path.read_text()
        assert not re.search(r"^\s*import airtight\.sim", text, re.M), path.name
        for module, names in re.findall(
            r"^\s*from (airtight\.sim[\w.]*) import ([^\n(]+|\([^)]*\))", text, re.M
        ):
            imported = {
                n.strip() for n in names.strip("()").replace("\n", " ").split(",") if n.strip()
            }
            found += len(imported)
            if module == "airtight.sim":
                assert imported <= {"scenarios", "constants"}, (path.name, imported)
            elif module in whole_modules:
                continue
            else:
                assert module in allowed, (path.name, module)
                assert imported <= allowed[module], (path.name, module, imported - allowed[module])
    assert found >= 6
