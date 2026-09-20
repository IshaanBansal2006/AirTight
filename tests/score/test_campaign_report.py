"""The campaign report, from a small synthetic results dict. Nothing here simulates."""

from __future__ import annotations

import copy
import json
import re
from typing import TYPE_CHECKING, Any

import pytest

from airtight.score import campaign_report as cr

if TYPE_CHECKING:
    from pathlib import Path

Row = dict[str, Any]

REC = "best:d3_swap_cams"
TACTICS = (
    "grid-main_gate-0.0000",
    "grid-main_gate-0.5000",
    "slow-grid-main_gate-0.0000",
    "gap-rear_fence_gap-0.2972",
)
STRICT_STAGES = ("A", "final_standin", "final_strong", "final_strong_nocams", "audit")
FREE = "best:d1_std_nocams"


def _row(
    label: str,
    hardware: str,
    cost: float,
    pd: float,
    worst: float,
    adversary: str = "stand-in",
    role: str = "final",
    n_seeds: int = 40,
    heldout: float | None = None,
    assumption: Row | None = None,
) -> Row:
    row: Row = {
        "label": label,
        "config": f"{hardware}__cfg",
        "hardware": hardware,
        "cost_per_hour": cost,
        "adversary": adversary,
        "seed_role": role,
        "n_seeds": n_seeds,
        "n_quiet": 12,
        "n_tactics": len(TACTICS),
        "strict": assumption is None,
        "assumption": assumption or {},
        "tau": 2.125,
        "flag": "ok",
        "tau_on_floor": False,
        "far_per_hour": 0.9,
        "pd": pd,
        "pd_ci": [pd - 0.0312, min(pd + 0.0287, 1.0)],
        "worst_naive": {"tactic": TACTICS[1], "pd": worst, "ci": [max(worst - 0.0511, 0.0), worst]},
        "objective": 0.5 * pd + 0.5 * worst,
        "pd_by_tactic": {t: (worst if i == 1 else pd) for i, t in enumerate(TACTICS)},
    }
    if heldout is not None:
        row["worst_heldout"] = {
            "tactic": TACTICS[3],
            "pd": heldout,
            "ci": [heldout - 0.0433, min(heldout + 0.0219, 1.0)],
            "n_seeds": n_seeds,
            "tau_single_column": 2.0,
        }
    return row


def _delta(a: str, pd_delta: float, worst_delta: float, n_seeds: int = 40) -> Row:
    return {
        "a": a,
        "b": "baseline",
        "n_seeds": n_seeds,
        "pd_delta": pd_delta,
        "pd_delta_ci": [pd_delta - 0.0213, pd_delta + 0.0214],
        "worst_delta": worst_delta,
        "worst_delta_ci": [worst_delta - 0.0315, worst_delta + 0.0316],
    }


def _entries(rows: list[Row]) -> list[Row]:
    policy = {"asset_gain": 0.25, "offsets": {"drone_1": 12.5}, "docks": {}, "origin": "random"}
    return [
        {
            "label": r["label"],
            "hardware": r["hardware"],
            "cost_per_hour": r["cost_per_hour"],
            "policy": None if r["label"] == "baseline" else policy,
            "config": r["config"],
        }
        for r in rows
    ]


def make_results(rec_pd: float = 0.9713, rec_worst: float = 0.9622) -> Row:
    """Every stage complete. With the defaults the target is reached; lower them for a ceiling."""
    d2, d3, d1 = "d2_std_nocams", "d3_swap_cams", "d1_std_nocams"
    standin = [
        _row("baseline", d2, 55.0, 0.4931, 0.0521),
        _row("confirmed_fix", d2, 55.0, 0.5417, 0.1042),
        _row("best:d1_std_nocams", d1, 48.0, 0.4861, 0.0313, heldout=0.2517),
        _row("best_free_policy", d2, 55.0, 0.6069, 0.2083, heldout=0.2519),
        _row(REC, d3, 67.38, rec_pd, rec_worst, heldout=rec_worst + 0.0111),
        _row("ingredient:weight", d2, 55.0, 0.5625, 0.0625, heldout=0.0731),
        _row("ingredient:schedule", d2, 55.0, 0.5486, 0.0417, heldout=0.2523),
    ]
    strong = [
        _row("baseline", d2, 55.0, 0.4474, 0.0123, "strong", n_seeds=30, heldout=0.0217),
        _row("confirmed_fix", d2, 55.0, 0.5351, 0.0641, "strong", n_seeds=30, heldout=0.1667),
        _row("best_free_policy", d2, 55.0, 0.5526, 0.0712, "strong", n_seeds=30, heldout=0.1669),
        _row(
            REC,
            d3,
            67.38,
            rec_pd - 0.0104,
            rec_worst - 0.0093,
            "strong",
            n_seeds=30,
            heldout=rec_worst,
        ),
    ]
    assumed = [
        _row(r["label"], r["hardware"], r["cost_per_hour"], 0.8714, 0.7136, heldout=h, assumption=a)
        for r, h, a in (
            (standin[0], None, {"task_time_s": 60.0}),
            (standin[3], 0.7341, {"task_time_s": 60.0}),
            (standin[4], 0.7343, {"task_time_s": 60.0}),
        )
    ]
    sens = []
    for label, hw, cost, base in (
        ("baseline", d2, 55.0, 0.4028),
        ("best_free_policy", d2, 55.0, 0.5278),
        (REC, d3, 67.38, 0.9444),
    ):
        for task, resp in ((0.0, 20.0), (60.0, 20.0), (0.0, 30.0)):
            strict = task == 0.0 and resp == 20.0
            row = _row(
                label,
                hw,
                cost,
                base if strict else base - 0.0777 + task / 400,
                0.0135,
                n_seeds=20,
                assumption=None if strict else {"task_time_s": task, "response_time_s": resp},
            )
            row["task_time_s"], row["response_time_s"] = task, resp
            sens.append(row)
    chosen = {
        hw: {**_row(f"{hw}:abc", hw, cost, pd, worst, role="validation", n_seeds=25), "digest": "x"}
        for hw, cost, pd, worst in (
            (d1, 48.0, 0.6331, 0.0456),
            (d2, 55.0, 0.6431, 0.2345),
            (d3, 67.38, 0.9876, 0.9765),
        )
    }
    a_rows = [copy.deepcopy(strong[0]), copy.deepcopy(strong[1])]
    stages: Row = {
        "probe": {"rate_sim_s_per_wall_s": 11000.0, "episodes_per_s": 50.0, "wall_s": 1.5},
        "A": {
            "complete": True,
            "entries": _entries(a_rows),
            "validation": [],
            "final": a_rows,
            "paired_vs_baseline": [_delta("confirmed_fix", 0.0877, 0.0518, 30)],
            "wall_s": 800.0,
        },
        "search": {"n0": 10, "specs": {}, "wall_s": 7000.0},
        "validation": {
            "complete": True,
            "n_seeds": 25,
            "candidates": {},
            "chosen": chosen,
            "recommended": d3,
            "recommended_rule": "the cheapest configuration reaching the target on validation",
            "picks": [d3],
            "frontier_labels": [REC, "best_free_policy"],
            "wall_s": 900.0,
        },
        "final_standin": {
            "complete": True,
            "entries": _entries(standin),
            "rows": standin,
            "paired_vs_baseline": [
                _delta(r["label"], r["pd"] - 0.4931, r["worst_naive"]["pd"] - 0.0521)
                for r in standin[1:]
            ],
            "upgrades": [],
            "task_time_s": 0.0,
            "wall_s": 1000.0,
        },
        "assumption60": {
            "complete": True,
            "entries": _entries(assumed),
            "rows": assumed,
            "paired_vs_baseline": [_delta(REC, 0.0, 0.0)],
            "upgrades": [],
            "task_time_s": 60.0,
            "wall_s": 400.0,
        },
        "final_strong": {
            "complete": True,
            "entries": _entries(strong),
            "validation": [],
            "final": strong,
            "paired_vs_baseline": [
                _delta(r["label"], r["pd"] - 0.4474, r["worst_naive"]["pd"] - 0.0123, 30)
                for r in strong[1:]
            ],
            "wall_s": 900.0,
        },
        "sensitivity": {"complete": True, "rows": sens, "recommended_label": REC, "wall_s": 300.0},
        "audit": {
            "rerun": {"checked": 10, "mismatches": []},
            "threshold_flags": [
                {
                    "label": "confirmed_fix",
                    "adversary": "strong",
                    "tau": 1.5,
                    "flag": "far_limited",
                    "tau_on_floor": True,
                }
            ],
            "too_good": [],
            "upgrades_that_lower_detection": [],
            "in_sample_vs_held_out": [],
            "wall_s": 100.0,
        },
    }
    results: Row = {
        "checkpoint": {
            "launch_commit": "abc123",
            "commits": ["abc123"],
            "engine_version": "1",
            "scenario_source_commit": "def456",
            "episodes": 1000,
            "quiet_nights": 50,
            "wall_s": 3600.0,
            "plans": {"final_standin": {"n": 40}},
            "stages_done": list(stages),
        },
        "stages": stages,
        "seed_split": {"final intrusion": "indices 600-1799"},
        "costs": {"entry_camera_usd_per_hour": 0.4},
        "engine_ignores": ["occlusion and altitude"],
        "shares": {},
    }
    results["targets"] = make_targets(results)
    return results


def make_targets(results: Row) -> Row:
    """What campaign._targets writes, rebuilt by hand so the test imports nothing that simulates."""
    out: Row = {}
    for name, stage, key in (
        ("strict", "final_standin", "rows"),
        ("task_time_60s_assumption", "assumption60", "rows"),
        ("strict_strong_finalists", "final_strong", "final"),
        ("strict_strong_without_cameras", "final_strong_nocams", "final"),
    ):
        rows = results["stages"].get(stage, {}).get(key, [])
        if not rows:
            continue
        out[name] = {}
        for target, metric in (
            ("overall", "pd"),
            ("worst_tactic_naive", "worst_naive"),
            ("worst_tactic_heldout", "worst_heldout"),
        ):
            known = [
                (r, r["pd"] if metric == "pd" else r[metric]["pd"])
                for r in rows
                if metric == "pd" or metric in r
            ]
            reached = [(r, v) for r, v in known if v >= cr.TARGET]
            if reached:
                row, v = min(reached, key=lambda rv: rv[0]["cost_per_hour"])
                cell = {"reached": True, "value": v}
            else:
                row, v = max(known, key=lambda rv: rv[1])
                cell = {"reached": False, "ceiling": v}
            out[name][target] = {
                **cell,
                "label": row["label"],
                "cost_per_hour": row["cost_per_hour"],
            }
    return out


def without_strong(results: Row) -> Row:
    out = copy.deepcopy(results)
    del out["stages"]["final_strong"]
    out["targets"] = make_targets(out)
    return out


def with_incomplete(results: Row) -> Row:
    out = copy.deepcopy(results)
    out["stages"]["sensitivity"] = {
        "complete": False,
        "reason": "cut before any seed finished",
        "wall_s": 5.0,
    }
    return out


def with_failed(results: Row) -> Row:
    out = copy.deepcopy(results)
    del out["stages"]["assumption60"]
    out["checkpoint"]["failed_stages"] = {
        "assumption60": "Traceback (most recent call last):\n  File x\nKeyError: 'boom'\n"
    }
    out["targets"] = make_targets(out)
    return out


def with_new_fields(results: Row, confirm: bool) -> Row:
    """The rescored output: constants, confirmed targets, reference rows, duty, far_check."""
    out = copy.deepcopy(results)
    out["constants"] = {
        "target": 0.95,
        "knee": 0.02,
        "far_target_per_hour": 1.0,
        "assumed_task_time_s": 60.0,
        "strict_response_time_s": 20.0,
        "intruder_speed_min_mps": 0.8,
        "intruder_speed_cap_mps": 2.0,
    }
    out["finished_at"], out["deadline_passed"] = 1789873123.0, True
    for group in out["targets"].values():
        for cell in group.values():
            cell["note"] = "NOTE-FROM-RESULTS same final seeds"
            cell["n_seeds"] = 40
            if cell["reached"]:
                cell["confirmed_label"] = cell["label"] if confirm else None
                cell["confirmed_cost_per_hour"] = cell["cost_per_hour"] if confirm else None
            else:
                cell["worst_tactic_of_ceiling"] = TACTICS[2]
    stages = out["stages"]
    stages["validation"]["reference"] = {
        "baseline": _row("baseline", "d2_std_nocams", 55.0, 0.5333, 0.0222, role="validation"),
    }
    for stage in ("A", "final_strong"):
        for row in stages[stage]["final"]:
            row["far_check"] = {
                "tau_from_first_half": 3.25,
                "far_on_second_half": 2.0513,
                "nights_each": [6, 6],
            }
            row["validation_worst_on_final"] = {TACTICS[0]: 0.3021, TACTICS[3]: 0.1011}
            row["validation_worst_on_final_n_seeds"] = 120
        for delta in stages[stage]["paired_vs_baseline"]:
            delta["worst_delta_kind"] = "naive: KIND-FROM-RESULTS"
    for stage in ("A", "final_standin", "final_strong", "assumption60"):
        for entry in stages[stage]["entries"]:
            entry["duty"] = {
                "drone_1": {"in_window": 0.5357, "steady_state": 0.3846},
                "go2_1": {"in_window": 0.6434, "steady_state": 0.6},
            }
    stages["audit"]["threshold_flags"][0]["stage"] = "final_strong"
    for row in stages["final_standin"]["rows"]:
        if row["label"].startswith("ingredient:"):
            del row["worst_heldout"]
    return out


def test_new_fields_are_used(tmp_path: Path) -> None:
    text, records = cr.build_report(with_new_fields(make_results(), confirm=True), {}, tmp_path)
    last = summary(text)[5]
    assert f"overall confirmed by {REC} at $67.38/h" in last
    assert "confirmed (interval lower end at or above target)" in section(text, "## Cheapest")
    assert "same final seeds so optimistic" in last
    assert "NOTE-FROM-RESULTS" in section(text, "## Cheapest configuration")
    assert "KIND-FROM-RESULTS" in text
    assert "2.051 OVER THE BUDGET" in text
    assert "scored on 120 final seeds" in text
    assert "in stage final_strong" in text
    assert "Deadline passed before the end: yes" in text
    assert "intruder speed 0.8 to 2 m/s" in text
    assert cr.NO_HELDOUT_INGREDIENT in text
    duty = section(text, "## Attack window and duty shares")
    assert cr.CAVEAT_WINDOW in duty
    assert "| baseline | drone_1 | 0.536 | 0.385 | +0.151 | FLAG" in duty
    assert "| baseline | go2_1 | 0.643 | 0.600 | +0.043 | - |" in duty
    assert "d5 and d6" in duty
    assert cr.CAVEAT_CAMERA_BOUND in text
    assert cr.CAVEAT_RESPONSE_QUIET in text
    points = [r for r in records if r["interval"] is None]
    assert {r["n_seeds"] for r in points} == {120}
    reference = [r for r in records if r["value"] == 0.5333]
    assert reference
    assert all(cr.SELECTION in r["label"] for r in reference)


def test_reached_is_never_claimed_without_confirmation(tmp_path: Path) -> None:
    results = with_new_fields(make_results(), confirm=False)
    last = summary(cr.build_report(results, {}, tmp_path)[0])[5]
    assert "reached on the point estimate only, not confirmed" in last
    assert "confirmed by" not in last
    assert not cr.target_is_confirmed(results, "strict", "overall", "final_standin")
    old = make_results(rec_pd=0.9991, rec_worst=0.9893)
    assert "confirmed_label" not in old["targets"]["strict"]["overall"]
    assert cr.target_is_confirmed(old, "strict", "overall", "final_standin")
    assert not cr.target_is_confirmed(make_results(), "strict", "overall", "final_standin")


def test_the_ceiling_names_the_tactic_with_its_speed(tmp_path: Path) -> None:
    results = with_new_fields(make_results(rec_pd=0.9013, rec_worst=0.8122), confirm=False)
    text, _ = cr.build_report(results, {}, tmp_path)
    assert "NOT reached" in summary(text)[5].split("Without entry cameras")[0]
    binds = section(text, "## Cheapest configuration")
    assert f"`{TACTICS[2]}`" in binds
    assert "minimum speed (0.8 m/s)" in binds
    assert "speed cap (2 m/s)" in binds


def with_nocams(results: Row) -> Row:
    """The stage that scores the recommendation without entry cameras."""
    out = copy.deepcopy(results)
    rows = [
        _row("baseline", "d2_std_nocams", 55.0, 0.4999, 0.0127, "strong", n_seeds=30, heldout=0.02),
        _row(FREE, "d1_std_nocams", 48.0, 0.4649, 0.0311, "strong", n_seeds=30, heldout=0.3333),
    ]
    out["stages"]["final_strong_nocams"] = {
        "complete": True,
        "entries": _entries(rows),
        "validation": [],
        "final": rows,
        "paired_vs_baseline": [_delta(FREE, -0.0351, 0.0184, 30)],
        "recommended_without_cameras": "d1_std_nocams",
        "recommended_without_cameras_label": FREE,
        "rule": "RULE-TEXT restricted to hardware without entry cameras",
        "wall_s": 12.0,
    }
    out["targets"] = make_targets(out)
    return out


def test_the_recommendation_without_cameras(tmp_path: Path) -> None:
    text, records = cr.build_report(with_nocams(make_results()), {}, tmp_path)
    line = summary(text)[2]
    assert line.startswith(f"3. Without entry cameras: {FREE} (1 drone, standard charging docks")
    assert "$48.00/h" in line
    assert "0.465 [0.434, 0.494]" in line
    assert "strict, strong adversary, 30 final seeds" in line
    assert "paired against the baseline, overall -0.035 [-0.056, -0.014]" in line
    part = section(text, "## Recommended without entry cameras")
    assert "RULE-TEXT" in part
    assert f"| {FREE} | $48.00/h | 0.465" in part
    assert f"Policy parameters of {FREE}" in part
    assert "Dock map:" in part
    assert any(r["stage"] == "final_strong_nocams" and r["strict"] for r in records)
    assert "the baseline and the recommendation without entry cameras" in text


def test_no_cameras_line_degrades(tmp_path: Path) -> None:
    text, _ = cr.build_report(make_results(), {}, tmp_path)
    assert "not evaluated against the strong adversary" in summary(text)[2]
    assert "final_strong_nocams is missing" in summary(text)[2]
    assert not re.search(r"\d\.\d", summary(text)[2])
    assert "Stage final_strong_nocams is MISSING" in text
    results = with_nocams(make_results())
    results["stages"]["final_strong_nocams"] = {"complete": False, "reason": "cut short"}
    text, _ = cr.build_report(results, {}, tmp_path)
    assert "final_strong_nocams is incomplete: cut short" in summary(text)[2]
    results = make_results()
    results["stages"]["validation"]["recommended"] = "d2_std_nocams"
    lines = summary(cr.build_report(results, {}, tmp_path)[0])
    assert "the main recommendation already has none" in lines[2]
    assert cr.CAMERA_WARNING not in lines[0]


def test_cameras_are_told_apart_and_the_camera_free_target_is_computed(tmp_path: Path) -> None:
    text, _ = cr.build_report(make_results(), {}, tmp_path)
    frontier = section(text, "## The frontier")
    assert f"| {REC} | yes (upper bound) | $67.38/h" in frontier
    assert "| baseline | no | $55.00/h" in frontier
    assert "WITHOUT entry cameras only (4 of 5 rows)" in frontier
    free_table = frontier.split("WITHOUT entry cameras only")[1]
    assert REC not in free_table
    free = summary(text)[5].split("Without entry cameras")[1]
    assert (
        "overall NOT reached, the ceiling is 0.607 [0.576, 0.636] set by best_free_policy" in free
    )
    results = make_results()
    row = results["stages"]["final_standin"]["rows"][3]
    row["pd"], row["pd_ci"] = 0.9612, [0.9312, 0.9899]
    free = summary(cr.build_report(results, {}, tmp_path)[0])[5].split("Without entry cameras")[1]
    assert "overall reached on the point estimate only, not confirmed" in free
    row["pd_ci"] = [0.9533, 0.9899]
    free = summary(cr.build_report(results, {}, tmp_path)[0])[5].split("Without entry cameras")[1]
    assert "overall confirmed by best_free_policy at $55.00/h" in free
    assert cr.has_cameras("d3_swap_cams")
    assert not cr.has_cameras("d2_std_nocams")


def summary(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()][:6]


def section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + 1)
    return text[start : end if end >= 0 else len(text)]


def all_numbers(node: Any) -> set[str]:
    """Every number in the input, formatted the ways the report formats numbers."""
    found: set[str] = set()
    if isinstance(node, dict):
        for value in node.values():
            found |= all_numbers(value)
    elif isinstance(node, list):
        for value in node:
            found |= all_numbers(value)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        found |= {cr.fmt(float(node)), f"{float(node):.2f}"}
    return found


VARIANTS = {
    "complete": lambda: make_results(),
    "rescored": lambda: with_new_fields(make_results(), confirm=True),
    "nocams": lambda: with_nocams(make_results()),
    "ceiling": lambda: make_results(rec_pd=0.9013, rec_worst=0.8122),
    "no_strong": lambda: without_strong(make_results()),
    "incomplete": lambda: with_incomplete(make_results()),
    "failed": lambda: with_failed(make_results()),
}


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_the_five_lines_name_the_recommendation_and_the_baseline(name: str, tmp_path: Path) -> None:
    text, _ = cr.build_report(VARIANTS[name](), {}, tmp_path)
    lines = summary(text)
    assert [line[:2] for line in lines] == ["1.", "2.", "3.", "4.", "5.", "6."]
    assert REC in lines[0]
    assert "3 drones" in lines[0]
    assert cr.CAMERA_WARNING in lines[0]
    assert "strict" in lines[1].lower()
    assert lines[2].startswith("3. Without entry cameras:")
    assert "ASSUMPTION" in lines[3]
    assert "baseline" in lines[4].lower()
    assert "95 percent" in lines[5]
    assert "Without entry cameras (stand-in adversary):" in lines[5]


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_every_float_in_the_five_lines_comes_from_the_input(name: str, tmp_path: Path) -> None:
    results = VARIANTS[name]()
    text, _ = cr.build_report(results, {}, tmp_path)
    known = all_numbers(results)
    floats = re.findall(r"-?\d+\.\d+", "\n".join(summary(text)))
    assert floats
    for token in floats:
        assert token in known, f"{token} is not a formatted value of results.json"


def test_the_headline_prefers_the_strong_adversary_and_falls_back(tmp_path: Path) -> None:
    text, _ = cr.build_report(make_results(), {}, tmp_path)
    assert "strong adversary, 30 final seeds" in summary(text)[1]
    assert "same adversary" in summary(text)[4]
    text, _ = cr.build_report(without_strong(make_results()), {}, tmp_path)
    assert "stand-in adversary, 40 final seeds" in summary(text)[1]
    assert "final_strong is MISSING" in text


def test_target_reached_against_ceiling(tmp_path: Path) -> None:
    text, _ = cr.build_report(make_results(), {}, tmp_path)
    assert "NOT reached" not in summary(text)[5].split("Without entry cameras")[0]
    text, _ = cr.build_report(make_results(rec_pd=0.9013, rec_worst=0.8122), {}, tmp_path)
    last = summary(text)[5]
    assert "NOT reached" in last.split("Without entry cameras")[0]
    assert "the ceiling is 0.901" in last
    assert REC in last
    assert "What binds" in section(text, "## Cheapest configuration")


def test_assumption_rows_never_enter_the_strict_frontier(tmp_path: Path) -> None:
    results = make_results()
    text, _ = cr.build_report(results, {}, tmp_path)
    frontier = section(text, "## The frontier")
    assert "ASSUMPTION" not in frontier
    assert cr.fmt(0.8714) not in frontier
    assert cr.fmt(0.8714) in section(text, "## ASSUMPTIONS")
    results["stages"]["final_standin"]["rows"][0]["strict"] = False
    with pytest.raises(ValueError, match="does not belong in a strict table"):
        cr.build_report(results, {}, tmp_path)


def test_no_table_mixes_strict_and_assumption_rows(tmp_path: Path) -> None:
    text, _ = cr.build_report(make_results(), {}, tmp_path)
    for block in re.split(r"\n\s*\n", text):
        body = [line for line in block.splitlines() if line.startswith("|")]
        if not any("| conditions |" in line for line in body):
            continue
        kinds = {"ASSUMPTION" in line for line in body[2:]}
        assert len(kinds) == 1


def test_validation_numbers_are_labelled_selection(tmp_path: Path) -> None:
    text, records = cr.build_report(make_results(), {}, tmp_path)
    for line in text.splitlines():
        if cr.fmt(0.9876) in line:
            assert cr.SELECTION in line
    chosen = [r for r in records if r["seed_role"] == "validation"]
    assert chosen
    assert all(cr.SELECTION in r["label"] for r in chosen)
    assert all(cr.SELECTION not in r["label"] for r in records if r["seed_role"] == "final")


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_number_records_are_complete_and_consistent(name: str, tmp_path: Path) -> None:
    _, records = cr.build_report(VARIANTS[name](), {}, tmp_path)
    assert records
    for record in records:
        assert set(cr.REQUIRED_RECORD_KEYS) <= set(record)
        assert isinstance(record["strict"], bool)
        assert record["strict"] == (record["assumption"] == {})
        if record["stage"] in STRICT_STAGES:
            assert record["strict"]
        if record["stage"] == "assumption60":
            assert not record["strict"]
            assert record["assumption"] == {"task_time_s": 60.0}
        assert record["interval"] is None or len(record["interval"]) == 2
        assert record["n_seeds"] is not None
        assert record["adversary"]


def test_broken_stages_are_said_plainly(tmp_path: Path) -> None:
    text, _ = cr.build_report(with_incomplete(make_results()), {}, tmp_path)
    assert "Stage sensitivity is INCOMPLETE: cut before any seed finished" in text
    text, _ = cr.build_report(with_failed(make_results()), {}, tmp_path)
    assert "assumption60 is FAILED" in text or "assumption60 is failed" in text
    assert "KeyError: 'boom'" in text
    assert "Stages without a complete result: assumption60" in text


def test_a_run_with_nothing_but_a_probe_still_reports(tmp_path: Path) -> None:
    results = make_results()
    results["stages"] = {"probe": results["stages"]["probe"]}
    results["targets"] = {}
    text, records = cr.build_report(results, {}, tmp_path)
    assert len(summary(text)) == 6
    assert "No recommended configuration" in summary(text)[0]
    assert records == []
    paths = cr.write_charts(results, tmp_path / "charts")
    assert all(p.stat().st_size > 0 for p in paths)


def test_notes_are_shown(tmp_path: Path) -> None:
    notes = {
        "needs_your_call": ["verify the swap dock price"],
        "merges": ["merged origin/main at 1234abc, no conflicts"],
        "check_pins": {"start": "clean", "end": "clean at the end"},
        "audit_actions": {"threshold_flags": "reported as is; thresholds sit on the floor"},
        "extra": ["the machine slept once"],
    }
    text, _ = cr.build_report(make_results(), notes, tmp_path)
    assert "1. verify the swap dock price" in text
    assert "merged origin/main at 1234abc" in text
    assert "clean at the end" in text
    assert "reported as is; thresholds sit on the floor" in text
    assert "the machine slept once" in text
    assert cr.CAVEAT_CAMERAS in text
    assert cr.CAVEAT_WORST in text


def test_the_cli_writes_the_report_the_numbers_and_the_charts(tmp_path: Path) -> None:
    out = tmp_path / "campaign"
    out.mkdir()
    (out / "results.json").write_text(json.dumps(make_results()))
    (out / "notes.json").write_text(json.dumps({"needs_your_call": ["look at this"]}))
    report_dir = tmp_path / "report"
    assert cr.main(["--out", str(out), "--report-dir", str(report_dir)]) == 0
    assert "look at this" in (report_dir / "REPORT.md").read_text()
    records = json.loads((report_dir / "numbers.json").read_text())
    assert isinstance(records, list)
    assert records
    for name in ("frontier", "phase", "ingredients", "sensitivity"):
        assert (report_dir / "charts" / f"{name}.png").stat().st_size > 2000
    assert cr.main(["--out", str(tmp_path / "nowhere")]) == 1


def test_tactic_ids_parse() -> None:
    assert cr.parse_tactic("slow-grid-rear_fence_gap-0.1250") == {
        "family": "slow-grid",
        "entry": "rear_fence_gap",
        "phase": 0.125,
        "speed": "minimum speed",
    }
    parsed = cr.parse_tactic("gap-main_gate-0.2972")
    assert parsed is not None
    assert (parsed["family"], parsed["entry"]) == ("gap", "main_gate")
    assert cr.parse_tactic("blind_spot-709026275") is None
    assert cr.parse_tactic("strong-main_gate-0.8-0.5000") is None
