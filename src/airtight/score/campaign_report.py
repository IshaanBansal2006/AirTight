"""The campaign's report: REPORT.md, numbers.json and four charts, from results.json alone.

    python -m airtight.score.campaign_report --out /abs/path/data/campaign

Nothing here simulates. Every number printed comes from results.json (or is a constant of
campaign.py); every detection number is printed with its interval and recorded in numbers.json
with its seed count, adversary, seed role and conditions. Strict rows and assumption rows never
share a table: a table refuses rows of the wrong kind. Validation-seed numbers are selection
numbers and are labelled so wherever they appear. A stage that is missing, incomplete or failed
is reported as such and the rest of the report still renders.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from airtight.score.campaign import (  # noqa: E402
    ASSUMED_TASK_TIME_S,
    BASELINE_LABEL,
    BEST_FREE_LABEL,
    FIX_LABEL,
    INGREDIENT_PREFIX,
    KNEE,
    STAGES,
    TARGET,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from matplotlib.axes import Axes

Row = dict[str, Any]

SELECTION = "selection, not a result"
CAVEAT_CAMERAS = (
    "Entry cameras sit on the entry point where every tactic starts, so they see the intruder "
    "at the moment of entry by construction; the contract's tactic model cannot express a "
    "breach away from a listed entry."
)
CAVEAT_WORST = (
    "Worst-tactic 'naive' is the minimum over many noisy per-tactic estimates on the final "
    "seeds and reads low; 'held-out' scores the tactic that was worst on validation seeds and "
    "reads at or above the true minimum. The truth lies between."
)
ROW_KEYS = {"A": "final", "final_strong": "final"}
REQUIRED_RECORD_KEYS = (
    "label",
    "value",
    "interval",
    "units",
    "n_seeds",
    "n_quiet",
    "adversary",
    "seed_role",
    "strict",
    "assumption",
    "configuration",
    "cost_per_hour",
    "stage",
)
HARDWARE_RE = re.compile(r"^d(\d+)_(std|swap)_(nocams|cams)$")
TACTIC_RE = re.compile(r"^(slow-grid|grid|gap)-(.+)-(\d+\.\d+)$")

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
CONTEXT = "#a3a29b"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


CAVEAT_CAMERA_BOUND = (
    "Camera configurations: every tactic starts at range 0 from an entry camera, association is "
    "by truth, there is no occlusion and only listed entries are attackable. Read camera rows "
    "as an upper bound under those assumptions; the fresh-random-tactic audit cannot break "
    "this, because its tactics start on the cameras too."
)
CAVEAT_RESPONSE_QUIET = (
    "Response-time sensitivity rows are scored with quiet nights simulated at that response "
    "time, because the band ring of the patrol weight scales with it."
)
CAVEAT_WINDOW = (
    "Tactic.phase spans one reference cycle from absolute time 0 (lane C's convention), and so "
    "do quiet nights. A fleet's size and its charge offsets change what falls inside that "
    "window, so an agent can be on duty more (or less) inside the window than in steady state; "
    "a charge period pushed outside the window is never attacked."
)
DEFAULT_TARGET_NOTE = (
    "chosen and reported on the same final seeds, so the value flatters the choice; "
    "'confirmed' asks the lower end of the interval to reach the target as well"
)
DUTY_FLAG = 0.10
NO_HELDOUT_INGREDIENT = "n/a (never on validation seeds)"


@dataclass(frozen=True)
class Consts:
    """results["constants"] when present, else campaign.py's constants (older results)."""

    target: float
    knee: float
    task_time_s: float
    far_target_per_hour: float | None
    strict_response_time_s: float | None
    speed_min_mps: float | None
    speed_cap_mps: float | None
    from_results: bool


def consts(results: Row) -> Consts:
    found = results.get("constants")
    c: Row = found if isinstance(found, dict) else {}

    def opt(key: str) -> float | None:
        return float(c[key]) if c.get(key) is not None else None

    return Consts(
        target=float(c.get("target", TARGET)),
        knee=float(c.get("knee", KNEE)),
        task_time_s=float(c.get("assumed_task_time_s", ASSUMED_TASK_TIME_S)),
        far_target_per_hour=opt("far_target_per_hour"),
        strict_response_time_s=opt("strict_response_time_s"),
        speed_min_mps=opt("intruder_speed_min_mps"),
        speed_cap_mps=opt("intruder_speed_cap_mps"),
        from_results=bool(c),
    )


# ---- formatting ------------------------------------------------------------------------------


def fmt(value: float) -> str:
    return f"{value:.3f}"


def fmt_ci(ci: Sequence[float] | None) -> str:
    return "[no interval]" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def fmt_delta(value: float) -> str:
    return f"{value:+.3f}"


def money(value: float) -> str:
    return f"${value:.2f}/h"


def percent(value: float) -> str:
    return f"{value * 100:.0f} percent"


def seconds(value: float) -> str:
    return f"{value:.0f} s"


def hardware_words(name: str) -> str:
    """A hardware name of hardware.py in plain words; the name itself when it does not parse."""
    match = HARDWARE_RE.match(name)
    if match is None:
        return name
    n, dock, cams = int(match.group(1)), match.group(2), match.group(3)
    drones = f"{n} drone" + ("" if n == 1 else "s")
    docks = "battery-swap docks" if dock == "swap" else "standard charging docks"
    cameras = (
        "a fixed camera at every uncovered entry" if cams == "cams" else "no added entry cameras"
    )
    return f"{drones}, {docks}, {cameras}"


def parse_tactic(tactic_id: str) -> dict[str, Any] | None:
    """Family, entry, phase and speed class of a stand-in tactic id; None for any other id."""
    match = TACTIC_RE.match(tactic_id)
    if match is None:
        return None
    family = match.group(1)
    return {
        "family": family,
        "entry": match.group(2),
        "phase": float(match.group(3)),
        "speed": "minimum speed" if family == "slow-grid" else "speed cap",
    }


def tactic_words(tactic_id: str, k: Consts | None = None) -> str:
    parsed = parse_tactic(tactic_id)
    if parsed is None:
        return f"`{tactic_id}` (not a grid or gap tactic: a lane C, random or strong-grid tactic)"
    kind = {
        "grid": "straight run on the phase grid",
        "slow-grid": "slow straight run on the phase grid",
        "gap": "straight run timed into a coverage gap",
    }[parsed["family"]]
    mps = None
    if k is not None:
        mps = k.speed_min_mps if parsed["family"] == "slow-grid" else k.speed_cap_mps
    speed = parsed["speed"] + (f" ({mps:g} m/s)" if mps is not None else "")
    return (
        f"`{tactic_id}` ({kind}, entry {parsed['entry']}, at the {speed}, "
        f"schedule phase {parsed['phase']:.4f})"
    )


# ---- reading results -------------------------------------------------------------------------


def stage_data(results: Row, stage: str) -> Row:
    data = results.get("stages", {}).get(stage)
    return data if isinstance(data, dict) else {}


def stage_status(results: Row, stage: str) -> tuple[str, str]:
    """("complete" | "incomplete" | "failed" | "missing", detail)."""
    data = results.get("stages", {}).get(stage)
    failed = results.get("checkpoint", {}).get("failed_stages", {})
    if isinstance(data, dict) and data.get("complete", True):
        return "complete", ""
    if isinstance(data, dict):
        return "incomplete", str(data.get("reason", "no reason recorded"))
    if stage in failed:
        lines = [line for line in str(failed[stage]).strip().splitlines() if line.strip()]
        return "failed", lines[-1].strip() if lines else "no traceback recorded"
    return "missing", "no result on record (not run, skipped, or past the deadline)"


def stage_rows(results: Row, stage: str) -> list[Row]:
    if stage_status(results, stage)[0] != "complete":
        return []
    rows = stage_data(results, stage).get(ROW_KEYS.get(stage, "rows"), [])
    return [r for r in rows if isinstance(r, dict)]


def find(rows: Iterable[Row], label: str | None) -> Row | None:
    return next((r for r in rows if r.get("label") == label), None)


def recommended_label(results: Row) -> str | None:
    validation = stage_data(results, "validation")
    name = validation.get("recommended") if validation.get("complete") else None
    if name is None:
        return None
    return BEST_FREE_LABEL if name == "d2_std_nocams" else f"best:{name}"


def is_ingredient(row: Row) -> bool:
    return str(row.get("label", "")).startswith(INGREDIENT_PREFIX)


# ---- the book of numbers ---------------------------------------------------------------------

METRIC_WORDS = {
    "pd": "overall detection probability",
    "worst_naive": "worst-tactic detection probability, naive (minimum on the scored seeds)",
    "worst_heldout": "worst-tactic detection probability, held-out (tactic picked on validation)",
}


def conditions(row: Row) -> str:
    assumption = row.get("assumption") or {}
    if row.get("strict", False):
        return "strict"
    parts = [f"{k} = {v:g}" for k, v in sorted(assumption.items())]
    return "ASSUMPTION " + ", ".join(parts)


@dataclass
class Book:
    """Every detection number the report prints, recorded once with its provenance."""

    records: list[Row] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    def _add(self, record: Row) -> None:
        key = json.dumps(record, sort_keys=True)
        if key not in self._seen:
            self._seen.add(key)
            self.records.append(record)

    def _base(self, row: Row, stage: str) -> Row:
        return {
            "n_seeds": row.get("n_seeds"),
            "n_quiet": row.get("n_quiet"),
            "adversary": row.get("adversary"),
            "seed_role": row.get("seed_role"),
            "strict": bool(row.get("strict", False)),
            "assumption": dict(row.get("assumption") or {}),
            "configuration": row.get("label"),
            "cost_per_hour": row.get("cost_per_hour"),
            "stage": stage,
        }

    def pd(self, row: Row | None, stage: str, metric: str = "pd") -> str:
        """The number with its interval, recorded. "n/a" when the row does not hold it."""
        if row is None:
            return "n/a"
        if metric == "pd":
            value, ci, n_seeds = row.get("pd"), row.get("pd_ci"), row.get("n_seeds")
        else:
            cell = row.get(metric)
            if not isinstance(cell, dict):
                ingredient = metric == "worst_heldout" and is_ingredient(row)
                return NO_HELDOUT_INGREDIENT if ingredient else "n/a"
            value, ci = cell.get("pd"), cell.get("ci")
            n_seeds = cell.get("n_seeds", row.get("n_seeds"))
        if value is None:
            return "n/a"
        role = row.get("seed_role")
        label = (
            f"{METRIC_WORDS[metric]} of {row.get('label')} against the {row.get('adversary')} "
            f"adversary on {n_seeds} {role} seeds, {conditions(row)}"
        )
        if role != "final":
            label += f" ({SELECTION})"
        record = self._base(row, stage)
        record.update(
            {
                "label": label,
                "value": float(value),
                "interval": [float(ci[0]), float(ci[1])] if ci else None,
                "units": "probability",
                "n_seeds": n_seeds,
            }
        )
        self._add(record)
        return f"{fmt(float(value))} {fmt_ci(ci)}"

    def point(self, value: float, words: str, row: Row, stage: str, n_seeds: Any) -> str:
        """A detection number results.json holds without an interval, recorded as such."""
        record = self._base(row, stage)
        record.update(
            {
                "label": f"{words} ({conditions(row)}; no interval on record)",
                "value": float(value),
                "interval": None,
                "units": "probability",
                "n_seeds": n_seeds,
            }
        )
        self._add(record)
        return f"{fmt(float(value))} [no interval]"

    def delta(self, delta: Row, which: str, a_row: Row | None, stage: str) -> str:
        """A paired difference (a minus b) with its paired interval, recorded."""
        value, ci = delta.get(f"{which}_delta"), delta.get(f"{which}_delta_ci")
        if value is None:
            return "n/a"
        context = a_row or {}
        what = (
            "overall detection probability"
            if which == "pd"
            else "worst-tactic detection probability (naive, each side's own worst tactic)"
        )
        record = self._base(context, stage)
        record.update(
            {
                "label": (
                    f"paired difference in {what}, {delta.get('a')} minus {delta.get('b')}, "
                    f"against the {context.get('adversary')} adversary on {delta.get('n_seeds')} "
                    f"{context.get('seed_role')} seeds, {conditions(context)}"
                ),
                "value": float(value),
                "interval": [float(ci[0]), float(ci[1])] if ci else None,
                "units": "probability difference",
                "n_seeds": delta.get("n_seeds"),
                "configuration": delta.get("a"),
            }
        )
        self._add(record)
        return f"{fmt_delta(float(value))} {fmt_ci(ci)}"


# ---- tables ----------------------------------------------------------------------------------


def table(header: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    if not body:
        return ["(no rows)", ""]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return [*lines, ""]


def require_kind(rows: Sequence[Row], strict: bool, where: str) -> None:
    """Strict rows and assumption rows never share a table."""
    for row in rows:
        if bool(row.get("strict", False)) is not strict:
            kind = "strict" if strict else "assumption"
            raise ValueError(f"{where}: row {row.get('label')} does not belong in a {kind} table")


def seeds_words(row: Row) -> str:
    return (
        f"{row.get('adversary')} adversary, {row.get('n_seeds')} {row.get('seed_role')} seeds, "
        f"{row.get('n_quiet')} quiet nights, {row.get('n_tactics')} tactics"
    )


def tau_words(row: Row) -> str:
    flag = str(row.get("flag"))
    floor = ", on the floor" if row.get("tau_on_floor") else ""
    return f"{float(row.get('tau', float('nan'))):.2f} ({flag}{floor})"


def detection_table(book: Book, rows: Sequence[Row], stage: str, strict: bool) -> list[str]:
    require_kind(rows, strict, stage)
    ordered = sorted(rows, key=lambda r: (r.get("cost_per_hour", 0.0), str(r.get("label"))))
    body = [
        [
            str(r.get("label")),
            money(float(r.get("cost_per_hour", float("nan")))),
            book.pd(r, stage, "pd"),
            book.pd(r, stage, "worst_naive"),
            book.pd(r, stage, "worst_heldout"),
            tau_words(r),
            conditions(r),
            str(r.get("n_seeds")),
        ]
        for r in ordered
    ]
    header = [
        "configuration",
        "cost",
        "overall pd [95% CI]",
        "worst naive [CI]",
        "worst held-out [CI]",
        "tau (flag)",
        "conditions",
        "seeds",
    ]
    return table(header, body)


# ---- sections --------------------------------------------------------------------------------


def headline_stage(results: Row, rec: str | None) -> str | None:
    """final_strong when it holds the recommended configuration, else final_standin."""
    if rec is not None and find(stage_rows(results, "final_strong"), rec) is not None:
        return "final_strong"
    if rec is not None and find(stage_rows(results, "final_standin"), rec) is not None:
        return "final_standin"
    return None


def _worst_clause(book: Book, row: Row, stage: str) -> str:
    held = row.get("worst_heldout")
    naive = book.pd(row, stage, "worst_naive")
    if not isinstance(held, dict):
        return f"worst tactic {naive} naive only (no held-out value for this row)"
    return (
        f"worst tactic {book.pd(row, stage, 'worst_heldout')} held-out on "
        f"{held.get('n_seeds')} seeds (naive {naive})"
    )


TARGET_METRICS = {
    "overall": "pd",
    "worst_tactic_naive": "worst_naive",
    "worst_tactic_heldout": "worst_heldout",
}


def _confirmed_row(results: Row, target: Row, stage: str, metric: str) -> Row | None:
    """The cheapest configuration whose interval lower end reaches the target: the campaign's
    own answer when it wrote one, else the same rule applied to the rows (older results)."""
    rows = [r for r in stage_rows(results, stage) if not is_ingredient(r)]
    if "confirmed_label" in target:
        return find(rows, target.get("confirmed_label"))
    goal = consts(results).target
    good = []
    for row in rows:
        got = _metric(row, metric)
        if got is not None and got[1] and float(got[1][0]) >= goal:
            good.append(row)
    return min(good, key=lambda r: (float(r["cost_per_hour"]), str(r["label"])), default=None)


def target_is_confirmed(results: Row, group: str, key: str, stage: str) -> bool:
    target = results.get("targets", {}).get(group, {}).get(key)
    if not isinstance(target, dict) or not target.get("reached"):
        return False
    return _confirmed_row(results, target, stage, TARGET_METRICS[key]) is not None


def _target_clause(book: Book, results: Row, group: str, key: str, stage: str) -> str:
    target = results.get("targets", {}).get(group, {}).get(key)
    if not isinstance(target, dict):
        return "no result"
    metric = TARGET_METRICS[key]
    if target.get("label") is None:
        return "no row holds this number"
    row = find(stage_rows(results, stage), target.get("label"))
    number = book.pd(row, stage, metric)
    where = f"{target.get('label')} at {money(float(target.get('cost_per_hour', 0.0)))}"
    if not target.get("reached"):
        return f"NOT reached, the ceiling is {number} set by {where}"
    confirmed = _confirmed_row(results, target, stage, metric)
    if confirmed is None:
        return (
            f"reached on the point estimate only, not confirmed (cheapest is {where} with "
            f"{number}; no interval lower end reaches the target)"
        )
    sure = (
        f"confirmed (interval lower end at or above target), cheapest confirmed is "
        f"{confirmed.get('label')} at {money(float(confirmed.get('cost_per_hour', 0.0)))} with "
        f"{book.pd(confirmed, stage, metric)}"
    )
    if confirmed.get("label") != target.get("label"):
        sure += (
            f"; reached on the point estimate, not confirmed, by the cheaper {where} with {number}"
        )
    return sure


def target_note(results: Row, group: str) -> str:
    cells = results.get("targets", {}).get(group, {})
    notes = [c.get("note") for c in cells.values() if isinstance(c, dict) and c.get("note")]
    return str(notes[0]) if notes else DEFAULT_TARGET_NOTE


def summary_lines(book: Book, results: Row) -> list[str]:
    rec = recommended_label(results)
    stage = headline_stage(results, rec)
    if rec is None or stage is None:
        status, detail = stage_status(results, "validation")
        why = (
            f"the validation stage is {status} ({detail})"
            if status != "complete"
            else "no final stage holds the recommended configuration"
        )
        base_stage = "final_standin" if stage_rows(results, "final_standin") else "A"
        base = find(stage_rows(results, base_stage), BASELINE_LABEL)
        base_text = (
            f"{book.pd(base, base_stage, 'pd')} overall, {seeds_words(base)}, strict"
            if base is not None
            else "no final-seed baseline row either"
        )
        return [
            f"1. No recommended configuration can be named: {why}.",
            "2. No strict detection numbers for a recommendation; see the stage status below.",
            "3. ASSUMPTION numbers for a recommendation: none.",
            f"4. Baseline (`{BASELINE_LABEL}`): {base_text}.",
            f"5. Whether {percent(consts(results).target)} was reached cannot be said from this run.",
        ]
    rows = stage_rows(results, stage)
    row = find(rows, rec) or {}
    hardware = str(row.get("hardware"))
    bought = (
        "the baseline's own hardware with a retuned patrol policy, nothing to buy"
        if rec == BEST_FREE_LABEL
        else hardware_words(hardware)
    )
    line1 = (
        f"1. Recommended: {rec} (hardware `{hardware}`: {bought}) at "
        f"{money(float(row.get('cost_per_hour', 0.0)))}."
    )
    line2 = (
        f"2. Strict, {seeds_words(row)}: overall {book.pd(row, stage, 'pd')}; "
        f"{_worst_clause(book, row, stage)}; held-out scores the one tactic picked on validation "
        "seeds, naive is the minimum over all tactics on the final seeds and reads low."
    )
    assumed = find(stage_rows(results, "assumption60"), rec)
    if assumed is None:
        status, detail = stage_status(results, "assumption60")
        line3 = (
            f"3. ASSUMPTION ({seconds(consts(results).task_time_s)} task time): no number for {rec}; "
            f"stage assumption60 is {status}{' (' + detail + ')' if detail else ''}."
        )
    else:
        line3 = (
            f"3. ASSUMPTION ({seconds(consts(results).task_time_s)} task time, not a strict result), "
            f"{seeds_words(assumed)}: overall {book.pd(assumed, 'assumption60', 'pd')}; "
            f"{_worst_clause(book, assumed, 'assumption60')}."
        )
    base = find(rows, BASELINE_LABEL)
    if base is None:
        line4 = f"4. Baseline (`{BASELINE_LABEL}`): no row in {stage}."
    else:
        line4 = (
            f"4. Baseline (`{BASELINE_LABEL}`, the scenario's own fleet and policy) at {money(float(base.get('cost_per_hour', 0.0)))}, "
            f"same adversary and seeds: overall {book.pd(base, stage, 'pd')}; "
            f"{_worst_clause(book, base, stage)}."
        )
    group = "strict" if stage_rows(results, "final_standin") else "strict_strong_finalists"
    group_stage = "final_standin" if group == "strict" else "final_strong"
    scope = (
        "every hardware configuration, stand-in adversary"
        if group == "strict"
        else "finalists only, strong adversary"
    )
    line5 = (
        f"5. {percent(consts(results).target)}, strict, {scope} (chosen and reported on the same "
        f"final seeds, so optimistic): overall "
        f"{_target_clause(book, results, group, 'overall', group_stage)}; worst tactic held-out "
        f"{_target_clause(book, results, group, 'worst_tactic_heldout', group_stage)}; "
        f"worst tactic naive "
        f"{_target_clause(book, results, group, 'worst_tactic_naive', group_stage)}."
    )
    return [line1, line2, line3, line4, line5]


def section_status(results: Row) -> list[str]:
    body = []
    for stage in STAGES:
        status, detail = stage_status(results, stage)
        wall = stage_data(results, stage).get("wall_s")
        plan = results.get("checkpoint", {}).get("plans", {}).get(stage)
        body.append(
            [
                stage,
                status.upper() if status != "complete" else status,
                detail or "-",
                f"{float(wall) / 60:.1f} min" if wall is not None else "-",
                json.dumps(plan, sort_keys=True) if plan is not None else "-",
            ]
        )
    lines = ["## Stage status, wall times and plans", ""]
    lines += table(["stage", "status", "detail", "wall time", "plan (sizes chosen)"], body)
    broken = [row[0] for row in body if row[1] != "complete"]
    if broken:
        lines += [
            f"**Stages without a complete result: {', '.join(broken)}.** Every section that "
            "depends on them says so instead of showing numbers.",
            "",
        ]
    return lines


def section_how_to_read(results: Row) -> list[str]:
    split = results.get("seed_split", {})
    lines = [
        "## How to read this",
        "",
        "- **Seed roles.** Search seeds pick policies. Validation seeds pick one policy per "
        "hardware configuration, the finalists, the recommended configuration and each "
        f"configuration's worst tactic: every validation number is a {SELECTION}. Final seeds "
        "only report; nothing was chosen on them.",
        "- **Naive and held-out worst tactic.** " + CAVEAT_WORST,
        "- **Strict** means the scenario exactly as lane C exported it: site, sensor curve, "
        "intruder limits, response time and benign traffic untouched, zero task time. A row "
        "marked ASSUMPTION changes the task time or the response time; it is a what-if, never a "
        "result, and never shares a table with strict rows.",
        "- Every number is `value [2.5th, 97.5th percentile]` from a bootstrap that resamples "
        "seeds (and quiet nights), never episodes. Thresholds (tau) are set per configuration "
        "from its own quiet nights at the false-alarm budget; `tau_on_floor` means the "
        "threshold sits on the investigate floor.",
        f"- The recommended configuration follows this rule ({SELECTION}): "
        f"{stage_data(results, 'validation').get('recommended_rule', 'rule not on record')}.",
        "",
    ]
    if split:
        lines += ["Seed splits:", ""]
        lines += [f"- {name}: {text}" for name, text in sorted(split.items())]
        lines.append("")
    return lines


def _by_value(metric: str) -> Callable[[Row], tuple[float, float]]:
    def key(row: Row) -> tuple[float, float]:
        value = row["pd"] if metric == "pd" else row[metric]["pd"]
        return float(value), -float(row["cost_per_hour"])

    return key


def _by_label_then(field_name: str) -> Callable[[Row], tuple[str, float]]:
    def key(row: Row) -> tuple[str, float]:
        return str(row["label"]), float(row[field_name])

    return key


def _running_best(rows: Sequence[Row], book: Book, stage: str) -> list[str]:
    costs = sorted({float(r["cost_per_hour"]) for r in rows})
    body = []
    for cost in costs:
        upto = [r for r in rows if float(r["cost_per_hour"]) <= cost]
        cells = [money(cost)]
        for metric in ("pd", "worst_naive", "worst_heldout"):
            have = [r for r in upto if metric == "pd" or isinstance(r.get(metric), dict)]
            if not have:
                cells.append("n/a")
                continue
            best = max(
                have,
                key=_by_value(metric),
            )
            cells.append(f"{book.pd(best, stage, metric)} ({best.get('label')})")
        body.append(cells)
    header = [
        "spend up to",
        "best overall pd so far",
        "best worst-tactic naive so far",
        "best worst-tactic held-out so far",
    ]
    return table(header, body)


def section_frontier(book: Book, results: Row) -> list[str]:
    lines = ["## The frontier (strict, stand-in adversary, final seeds)", ""]
    status, detail = stage_status(results, "final_standin")
    if status != "complete":
        return [*lines, f"Stage final_standin is {status.upper()}: {detail}. No frontier.", ""]
    rows = [r for r in stage_rows(results, "final_standin") if not is_ingredient(r)]
    require_kind(rows, True, "frontier")
    strong = stage_rows(results, "final_strong")
    ordered = sorted(rows, key=lambda r: (float(r["cost_per_hour"]), str(r["label"])))
    body = []
    for r in ordered:
        s = find(strong, r.get("label"))
        body.append(
            [
                str(r.get("label")),
                money(float(r["cost_per_hour"])),
                book.pd(r, "final_standin", "pd"),
                book.pd(r, "final_standin", "worst_naive"),
                book.pd(r, "final_standin", "worst_heldout"),
                tau_words(r),
                book.pd(s, "final_strong", "pd") if s else "not a finalist",
                book.pd(s, "final_strong", "worst_heldout") if s else "-",
            ]
        )
    first = ordered[0]
    lines += [f"All rows strict: {seeds_words(first)}."]
    if strong:
        require_kind(strong, True, "frontier strong columns")
        lines += [f"Strong columns (finalists only), strict: {seeds_words(strong[0])}."]
    else:
        status, detail = stage_status(results, "final_strong")
        lines += [f"Strong columns are empty: stage final_strong is {status.upper()} ({detail})."]
    lacking = [str(r.get("label")) for r in ordered if not isinstance(r.get("worst_heldout"), dict)]
    lines += [
        f"{len(lacking)} of {len(ordered)} rows lack a held-out worst tactic"
        + (f" ({', '.join(lacking)})" if lacking else "")
        + ". A row lacks one when its validation-worst tactic is a gap tactic that is absent "
        "from the final union set of tactics, or, in results written before the validation "
        f"reference rows existed, when it is {BASELINE_LABEL} or {FIX_LABEL}. Ingredient rows "
        "(section B) deliberately have none: they were never on validation seeds.",
        "",
    ]
    header = [
        "configuration",
        "cost",
        "overall pd [95% CI]",
        "worst naive [CI]",
        "worst held-out [CI]",
        "tau (flag)",
        "STRONG overall pd [CI]",
        "STRONG worst held-out [CI]",
    ]
    lines += table(header, body)
    lines += ["Best so far at each cost level (same rows, same seeds):", ""]
    lines += _running_best(ordered, book, "final_standin")
    lines += ["Chart: `charts/frontier.png`.", ""]
    return lines


def _what_binds(results: Row, stage: str, target: Row) -> str:
    k = consts(results)
    row = find(stage_rows(results, stage), target.get("label"))
    if row is None:
        return "the ceiling row is not on record, so what binds cannot be said"
    worst = str(target.get("worst_tactic_of_ceiling") or row["worst_naive"]["tactic"])
    parts = [f"its naive worst tactic is {tactic_words(worst, k)}"]
    held = row.get("worst_heldout")
    if isinstance(held, dict):
        parts.append(
            f"its validation-picked worst tactic is {tactic_words(str(held['tactic']), k)}"
        )
    per_tactic = row.get("pd_by_tactic", {})
    below = sorted(t for t, v in per_tactic.items() if float(v) < k.target)
    if below:
        families: dict[str, int] = {}
        for tactic in below:
            parsed = parse_tactic(tactic)
            key = f"{parsed['family']} at {parsed['entry']}" if parsed else "other"
            families[key] = families.get(key, 0) + 1
        counted = ", ".join(f"{k}: {n}" for k, n in sorted(families.items()))
        parts.append(
            f"{len(below)} of {len(per_tactic)} tactics sit below the target "
            f"(count by family and entry: {counted})"
        )
    return "; ".join(parts)


def section_targets(book: Book, results: Row) -> list[str]:
    k = consts(results)
    lines = [
        f"## Cheapest configuration to {percent(k.target)}",
        "",
        "Two readings. **Reached on the point estimate**: the value is at or above the target. "
        "**Confirmed**: the lower end of the interval is at or above the target as well. Only "
        "'confirmed' should be read as reaching the target. Ingredient rows are excluded.",
        "",
    ]
    groups = (
        ("strict", "final_standin", "STRICT, stand-in adversary, every configuration"),
        (
            "task_time_60s_assumption",
            "assumption60",
            f"ASSUMPTION, {seconds(k.task_time_s)} task time, stand-in adversary, "
            "frontier configurations only",
        ),
        ("strict_strong_finalists", "final_strong", "STRICT, strong adversary, finalists only"),
    )
    for group, stage, title in groups:
        lines += [f"### {title}", ""]
        rows = stage_rows(results, stage)
        if group not in results.get("targets", {}) or not rows:
            status, detail = stage_status(results, stage)
            lines += [f"No answer: stage {stage} is {status.upper()} ({detail or 'no rows'}).", ""]
            continue
        lines += [
            f"Rows: {seeds_words(rows[0])}, {conditions(rows[0])}. Note: "
            f"{target_note(results, group)}.",
            "",
        ]
        for key, words in (
            ("overall", "Overall"),
            ("worst_tactic_heldout", "Worst tactic, held-out"),
            ("worst_tactic_naive", "Worst tactic, naive"),
        ):
            target = results["targets"][group].get(key, {})
            text = f"- **{words}:** {_target_clause(book, results, group, key, stage)}."
            if not target.get("reached") and target.get("label") is not None:
                text += f" What binds: {_what_binds(results, stage, target)}."
            lines.append(text)
        lines.append("")
    return lines


def _paired_lines(
    book: Book, results: Row, stage: str, labels: Sequence[str] | None = None
) -> list[str]:
    rows = stage_rows(results, stage)
    deltas = stage_data(results, stage).get("paired_vs_baseline", []) if rows else []
    body = []
    for d in sorted(deltas, key=lambda d: d.get("a") == BEST_FREE_LABEL and labels is not None):
        if labels is not None and d.get("a") not in labels:
            continue
        a_row = find(rows, d.get("a"))
        body.append(
            [
                f"{d.get('a')} minus {d.get('b')}",
                book.delta(d, "pd", a_row, stage),
                book.delta(d, "worst", a_row, stage),
                str(d.get("n_seeds")),
            ]
        )
    header = [
        "paired difference",
        "overall pd delta [paired CI]",
        "worst naive delta [paired CI]",
        "seeds",
    ]
    kinds = sorted({str(d["worst_delta_kind"]) for d in deltas if d.get("worst_delta_kind")})
    kind = "; ".join(kinds) or "naive (each side's own worst tactic; kind not on record)"
    lines = table(header, body)
    if body:
        lines += [f"Worst-tactic delta kind: {kind}.", ""]
    return lines


def _far_cell(value: Any, budget: float | None) -> str:
    if value is None:
        return "n/a"
    over = budget is not None and float(value) > budget
    return f"{float(value):.3f}" + (" OVER THE BUDGET" if over else "")


def _strong_extras(book: Book, results: Row, stage: str) -> list[str]:
    """The out-of-sample false alarm check and every validation-worst tactic on final seeds."""
    rows = stage_rows(results, stage)
    lines: list[str] = []
    checked = [r for r in rows if isinstance(r.get("far_check"), dict)]
    if checked:
        k = consts(results)
        budget = (
            f" The false alarm budget is {k.far_target_per_hour:g} per hour."
            if k.far_target_per_hour is not None
            else ""
        )
        lines += [
            "False alarm rate, in-sample and the out-of-sample check (threshold set on the first "
            f"half of the quiet nights, false alarms counted on the second half).{budget}",
            "",
        ]
        body = [
            [
                str(r.get("label")),
                f"{float(r.get('far_per_hour', float('nan'))):.3f}",
                f"{float(r['far_check'].get('tau_from_first_half', float('nan'))):.2f}",
                _far_cell(r["far_check"].get("far_on_second_half"), k.far_target_per_hour),
                " and ".join(str(n) for n in r["far_check"].get("nights_each", [])),
            ]
            for r in checked
        ]
        header = [
            "configuration",
            "in-sample false alarms per hour",
            "tau from first half",
            "OUT-OF-SAMPLE false alarms per hour (second half)",
            "quiet nights in each half",
        ]
        lines += table(header, body)
    else:
        lines += ["No out-of-sample false alarm check on record for these rows.", ""]
    for r in rows:
        worst = r.get("validation_worst_on_final")
        if not isinstance(worst, dict) or not worst:
            continue
        n = r.get("validation_worst_on_final_n_seeds", r.get("n_seeds"))
        lines += [
            f"{r.get('label')}: every tactic that was among its worst on validation seeds, "
            f"scored on {n} final seeds (strict, {r.get('adversary')} adversary):",
            "",
        ]
        body = [
            [
                f"`{tactic}`",
                book.point(
                    float(value),
                    f"detection probability of {r.get('label')} against validation-worst tactic "
                    f"{tactic} on {n} final seeds, {r.get('adversary')} adversary",
                    r,
                    stage,
                    n,
                ),
            ]
            for tactic, value in sorted(worst.items(), key=lambda tv: (float(tv[1]), tv[0]))
        ]
        lines += table(["validation-worst tactic", "pd on final seeds"], body)
    return lines


def section_finalists(book: Book, results: Row) -> list[str]:
    lines = ["## Finalists against the strong adversary (strict)", ""]
    status, detail = stage_status(results, "final_strong")
    if status != "complete":
        return [*lines, f"Stage final_strong is {status.upper()}: {detail}.", ""]
    rows = stage_rows(results, "final_strong")
    lines += [f"All rows strict: {seeds_words(rows[0])}.", ""]
    lines += detection_table(book, rows, "final_strong", True)
    lines += ["Paired against the baseline, same seeds and quiet nights:", ""]
    lines += _paired_lines(book, results, "final_strong")
    lines += _strong_extras(book, results, "final_strong")
    return lines


def _entry_records(results: Row) -> dict[str, Row]:
    found: dict[str, Row] = {}
    for stage in ("final_standin", "final_strong", "A", "assumption60"):
        for entry in stage_data(results, stage).get("entries", []):
            if isinstance(entry, dict) and entry.get("label") is not None:
                found.setdefault(str(entry["label"]), entry)
    return found


def section_duty(results: Row) -> list[str]:
    lines = ["## Attack window and duty shares", "", CAVEAT_WINDOW, ""]
    entries = _entry_records(results)
    wanted = [BASELINE_LABEL, BEST_FREE_LABEL, recommended_label(results)]
    wanted += [str(e.get("label")) for e in stage_data(results, "final_strong").get("entries", [])]
    labels = [x for x in dict.fromkeys(wanted) if x is not None and x in entries]
    body = []
    for label in labels:
        duty = entries[label].get("duty")
        if not isinstance(duty, dict):
            continue
        for agent, shares in sorted(duty.items()):
            inside, steady = float(shares["in_window"]), float(shares["steady_state"])
            gap = inside - steady
            flag = f"FLAG: differs by more than {DUTY_FLAG:.2f}" if abs(gap) > DUTY_FLAG else "-"
            body.append([label, agent, fmt(inside), fmt(steady), fmt_delta(gap), flag])
    if body:
        lines += [
            "Share of time each charging agent is on duty inside the attack window, next to its "
            "steady-state share (shares of time, not detection numbers):",
            "",
        ]
        header = [
            "configuration",
            "agent",
            "on duty inside the window",
            "on duty in steady state",
            "difference",
            "flag",
        ]
        lines += table(header, body)
        flagged = sum(1 for cells in body if cells[5] != "-")
        lines += [f"{flagged} of {len(body)} agent rows are flagged.", ""]
    else:
        lines += ["No duty shares on record (results written before the `duty` field).", ""]
    lines += ["Dock maps of the chosen policies:", ""]
    for label, entry in sorted(entries.items()):
        if label.startswith(INGREDIENT_PREFIX):
            continue
        policy = entry.get("policy")
        docks = policy.get("docks") if isinstance(policy, dict) else None
        text = json.dumps(docks, sort_keys=True) if docks else "the scenario's own dock assignment"
        lines.append(f"- {label} (`{entry.get('hardware')}`): {text}")
    lines += [
        "",
        "The engine ignores dock capacity (every agent has its own pad), and the d5 and d6 fleets "
        "exceed the site's dock capacity with no extra dock priced, so their cost reads low.",
        "",
    ]
    return lines


def section_a(book: Book, results: Row) -> list[str]:
    lines = ["## A. Baseline and confirmed fix against the strong adversary (strict)", ""]
    status, detail = stage_status(results, "A")
    if status != "complete":
        return [*lines, f"Stage A is {status.upper()}: {detail}.", ""]
    rows = stage_rows(results, "A")
    lines += [f"All rows strict: {seeds_words(rows[0])}.", ""]
    lines += detection_table(book, rows, "A", True)
    lines += ["Paired against the baseline, same seeds and quiet nights:", ""]
    lines += _paired_lines(book, results, "A")
    lines += _strong_extras(book, results, "A")
    lines += [
        f"Stage A also holds validation-seed rows; they only picked each configuration's worst "
        f"tactic and are a {SELECTION}, so they are not shown.",
        "",
    ]
    return lines


def section_b(book: Book, results: Row) -> list[str]:
    lines = ["## B. The best free policy and its ingredients (strict)", ""]
    status, detail = stage_status(results, "final_standin")
    if status != "complete":
        return [*lines, f"Stage final_standin is {status.upper()}: {detail}.", ""]
    data = stage_data(results, "final_standin")
    entry = find(data.get("entries", []), BEST_FREE_LABEL)
    if entry is None:
        return [*lines, f"No {BEST_FREE_LABEL} entry in final_standin.", ""]
    lines += [
        "Same agents, same hardware and same cost as the baseline; only the patrol policy, "
        "charge offsets and dock assignment differ. Its parameters:",
        "",
        "```json",
        json.dumps(entry.get("policy"), indent=1, sort_keys=True),
        "```",
        "",
    ]
    for stage in ("final_standin", "final_strong"):
        rows = stage_rows(results, stage)
        if find(rows, BEST_FREE_LABEL) is None:
            status, detail = stage_status(results, stage)
            lines += [f"No {BEST_FREE_LABEL} row in {stage} (stage {status}). {detail}".strip(), ""]
            continue
        lines += [f"Strict, {seeds_words(rows[0])}:", ""]
        keep = [r for r in rows if r.get("label") in (BASELINE_LABEL, BEST_FREE_LABEL)]
        lines += detection_table(book, keep, stage, True)
        lines += _paired_lines(book, results, stage, [BEST_FREE_LABEL])
    ingredients = [r for r in stage_rows(results, "final_standin") if is_ingredient(r)]
    lines += ["### Ingredient breakdown", ""]
    if not ingredients:
        return [*lines, "No ingredient rows on record.", ""]
    lines += [
        "Each row is the baseline's policy with one group of the best policy's fields swapped "
        f"in. Strict, {seeds_words(ingredients[0])}.",
        "",
    ]
    lines += detection_table(book, ingredients, "final_standin", True)
    labels = [str(r["label"]) for r in ingredients]
    lines += _paired_lines(book, results, "final_standin", [*labels, BEST_FREE_LABEL])
    lines += ["Chart: `charts/ingredients.png`.", ""]
    return lines


def section_selection(book: Book, results: Row) -> list[str]:
    lines = [f"## How the configurations were chosen ({SELECTION})", ""]
    validation = stage_data(results, "validation")
    status, detail = stage_status(results, "validation")
    if status != "complete" or not validation.get("chosen"):
        return [*lines, f"Stage validation is {status.upper()}: {detail}.", ""]
    chosen = list(validation["chosen"].values())
    lines += [
        f"**Every number in this table is a {SELECTION}.** These validation-seed numbers chose "
        "the policy per hardware configuration, the finalists "
        f"({', '.join(validation.get('picks', []))}) and the recommended configuration "
        f"({validation.get('recommended')}); they are optimistic by construction. "
        f"{seeds_words(chosen[0])}. Knee: {consts(results).knee}.",
        "",
    ]
    ordered = sorted(chosen, key=lambda r: (float(r["cost_per_hour"]), str(r["hardware"])))
    body = [
        [
            str(r.get("hardware")),
            money(float(r["cost_per_hour"])),
            f"{book.pd(r, 'validation', 'pd')} ({SELECTION})",
            f"{book.pd(r, 'validation', 'worst_naive')} ({SELECTION})",
        ]
        for r in ordered
    ]
    lines += table(["hardware", "cost", "overall pd [CI]", "worst naive [CI]"], body)
    reference = validation.get("reference")
    if isinstance(reference, dict) and reference:
        lines += [
            f"Reference rows on the same validation seeds ({SELECTION}); they only pick the "
            f"held-out worst tactic of {BASELINE_LABEL} and {FIX_LABEL}:",
            "",
        ]
        body = [
            [
                str(r.get("label")),
                money(float(r["cost_per_hour"])),
                f"{book.pd(r, 'validation', 'pd')} ({SELECTION})",
                f"{book.pd(r, 'validation', 'worst_naive')} ({SELECTION})",
                f"`{r['worst_naive']['tactic']}`",
            ]
            for _, r in sorted(reference.items())
        ]
        header = ["configuration", "cost", "overall pd [CI]", "worst naive [CI]", "worst tactic"]
        lines += table(header, body)
    return lines


def _sensitivity_axes(rows: Sequence[Row]) -> tuple[float | None, list[Row], list[Row]]:
    """The strict response time, the task-time axis rows and the response-time axis rows."""
    anchor = next((r for r in rows if r.get("strict")), None)
    if anchor is None:
        return None, [], []
    resp = float(anchor["response_time_s"])
    task_axis = [r for r in rows if float(r["response_time_s"]) == resp]
    resp_axis = [r for r in rows if float(r["task_time_s"]) == 0.0]
    return resp, task_axis, resp_axis


def section_assumptions(book: Book, results: Row) -> list[str]:
    lines = [
        "## ASSUMPTIONS: task-time and response-time what-ifs",
        "",
        "> **Everything between this heading and the next is an ASSUMPTION, not a result.** "
        "The strict scenario has zero task time and the exported response time. These rows ask "
        "what detection would be if that were different. None of them is a headline.",
        "",
        f"### ASSUMPTION: {seconds(consts(results).task_time_s)} task time, frontier configurations",
        "",
    ]
    rows = stage_rows(results, "assumption60")
    if rows:
        lines += [f"{seeds_words(rows[0])}, {conditions(rows[0])}.", ""]
        lines += detection_table(book, rows, "assumption60", False)
        lines += ["Paired against the baseline under the same assumption:", ""]
        lines += _paired_lines(book, results, "assumption60")
    else:
        status, detail = stage_status(results, "assumption60")
        lines += [f"Stage assumption60 is {status.upper()}: {detail}.", ""]
    lines += ["### ASSUMPTION: sensitivity axes", ""]
    rows = stage_rows(results, "sensitivity")
    if not rows:
        status, detail = stage_status(results, "sensitivity")
        return [*lines, f"Stage sensitivity is {status.upper()}: {detail}.", "", "> End.", ""]
    resp, task_axis, resp_axis = _sensitivity_axes(rows)
    anchors = [r for r in rows if r.get("strict")]
    lines += [
        f"Three configurations ({', '.join(sorted({str(r['label']) for r in rows}))}), two axes "
        f"through the strict cell. {seeds_words(rows[0])}.",
        "",
        "The strict anchor cells on these seeds, in a table of their own (strict; fewer seeds "
        "than the frontier, so prefer the frontier's numbers):",
        "",
    ]
    lines += detection_table(book, anchors, "sensitivity", True)
    for title, axis, key in (
        (f"ASSUMPTION: task time varied, response time {resp:g} s", task_axis, "task_time_s"),
        ("ASSUMPTION: response time varied, zero task time", resp_axis, "response_time_s"),
    ):
        assumed = [r for r in axis if not r.get("strict")]
        require_kind(assumed, False, title)
        ordered = sorted(assumed, key=_by_label_then(key))
        body = [
            [
                str(r["label"]),
                f"{float(r[key]):g} s",
                book.pd(r, "sensitivity", "pd"),
                book.pd(r, "sensitivity", "worst_naive"),
                tau_words(r),
                conditions(r),
            ]
            for r in ordered
        ]
        lines += [f"#### {title}", ""]
        header = [
            "configuration",
            key,
            "overall pd [CI]",
            "worst naive [CI]",
            "tau (flag)",
            "conditions",
        ]
        lines += table(header, body)
    lines += ["Chart: `charts/sensitivity.png`.", "", "> End of the ASSUMPTIONS section.", ""]
    return lines


def section_audit(book: Book, results: Row, notes: Row) -> list[str]:
    lines = ["## Audit flags and what was done about them", ""]
    status, detail = stage_status(results, "audit")
    if status != "complete":
        return [*lines, f"Stage audit is {status.upper()}: {detail}. Nothing was audited.", ""]
    audit = stage_data(results, "audit")
    actions = notes.get("audit_actions", {})

    def action(key: str) -> str:
        return str(actions.get(key, "no action recorded in notes.json"))

    rerun = audit.get("rerun", {})
    mismatches = rerun.get("mismatches", [])
    lines += [
        f"- **rerun** (cached rows re-simulated, must be bit-identical): {rerun.get('checked')} "
        f"checked, {len(mismatches)} mismatches"
        + (f" {json.dumps(mismatches)}" if mismatches else "")
        + f". Action: {action('rerun') if mismatches or 'rerun' in actions else 'none needed'}.",
    ]
    flags = audit.get("threshold_flags", [])
    lines.append(
        f"- **threshold_flags** (threshold not ok, or on the floor): {len(flags)} rows. "
        f"Action: {action('threshold_flags') if flags else 'none needed'}."
    )
    lines += [
        f"    - {f.get('label')} against {f.get('adversary')}"
        f"{' in stage ' + str(f['stage']) if f.get('stage') else ''}: tau {float(f.get('tau', 0)):.2f}, "
        f"flag {f.get('flag')}, on floor {f.get('tau_on_floor')}"
        for f in flags
    ]
    too_good = audit.get("too_good", [])
    lines.append(
        f"- **too_good** (detection above 0.99, re-tested against fresh random tactics on final "
        f"seeds): {len(too_good)} configurations. "
        f"Action: {action('too_good') if too_good else 'none needed'}."
    )
    for item in too_good:
        row = {
            "label": item.get("label"),
            "pd": item.get("pd"),
            "pd_ci": item.get("pd_ci"),
            "worst_naive": item.get("worst"),
            "n_seeds": item.get("n_seeds"),
            "adversary": "fresh random",
            "seed_role": "final",
            "strict": True,
            "assumption": {},
        }
        lines.append(
            f"    - {item.get('label')}, strict, {item.get('n_tactics')} fresh random tactics, "
            f"{item.get('n_seeds')} final seeds: overall {book.pd(row, 'audit', 'pd')}, worst "
            f"naive {book.pd(row, 'audit', 'worst_naive')}, "
            f"{item.get('n_tactics_below_target')} tactics below the target"
        )
    lower = audit.get("upgrades_that_lower_detection", [])
    lines.append(
        f"- **upgrades_that_lower_detection** (one purchase more, paired, point estimate lower): "
        f"{len(lower)}. Action: {action('upgrades_that_lower_detection') if lower else 'none needed'}."
    )
    standin = stage_rows(results, "final_standin")
    for u in lower:
        a_row = find(standin, u.get("a"))
        lines.append(
            f"    - {u.get('from')} to {u.get('to')}, strict, stand-in: overall "
            f"{book.delta(u, 'pd', a_row, 'final_standin')}, worst naive "
            f"{book.delta(u, 'worst', a_row, 'final_standin')}, {u.get('n_seeds')} seeds"
        )
    gap = audit.get("in_sample_vs_held_out", [])
    lines.append(
        f"- **in_sample_vs_held_out** (validation and final numbers further apart than their "
        f"intervals allow): {len(gap)}. "
        f"Action: {action('in_sample_vs_held_out') if gap else 'none needed'}."
    )
    chosen = stage_data(results, "validation").get("chosen", {})
    for g in gap:
        final_row = find(standin, g.get("label"))
        metric = "pd" if g.get("metric") == "pd" else "worst_naive"
        val_row = chosen.get(final_row.get("hardware")) if final_row else None
        lines.append(
            f"    - {g.get('label')}, {g.get('metric')}: validation "
            f"{book.pd(val_row, 'validation', metric)} ({SELECTION}) against final "
            f"{book.pd(final_row, 'final_standin', metric)}"
        )
    extra = sorted(set(actions) - set(audit))
    lines += [f"- note for `{key}`: {actions[key]}" for key in extra]
    lines.append("")
    return lines


def section_caveats(results: Row) -> list[str]:
    lines = ["## Standing caveats", "", f"- {CAVEAT_CAMERAS}", f"- {CAVEAT_WORST}"]
    lines += [
        f"- {CAVEAT_CAMERA_BOUND}",
        f"- {CAVEAT_RESPONSE_QUIET}",
        "- 'Reached' for the target is a point estimate chosen and reported on the same final "
        "seeds; only 'confirmed' (interval lower end at or above the target) should be relied on.",
        "- Paired worst-tactic differences compare each side's own naive worst tactic, so they "
        "inherit the naive reading's bias on both sides.",
        "- Per-tactic detection (the phase chart) is a point estimate per tactic; results.json "
        "holds an interval only for the worst tactic.",
        "",
        "## What the engine ignores",
        "",
    ]
    ignores = results.get("engine_ignores", [])
    lines += [f"- {item}" for item in ignores] or ["- not on record"]
    lines.append("")
    return lines


def _run_end_lines(results: Row) -> list[str]:
    k = consts(results)
    lines = []
    finished = results.get("finished_at")
    if finished is not None:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(float(finished)))
        lines.append(f"- Finished at {stamp} (local time)")
    if "deadline_passed" in results:
        passed = "yes, later stages may be missing" if results["deadline_passed"] else "no"
        lines.append(f"- Deadline passed before the end: {passed}")
    if k.from_results:
        speeds = (
            f"{k.speed_min_mps:g} to {k.speed_cap_mps:g} m/s"
            if k.speed_min_mps is not None and k.speed_cap_mps is not None
            else "not on record"
        )
        far = f"{k.far_target_per_hour:g}" if k.far_target_per_hour is not None else "unknown"
        resp = (
            seconds(k.strict_response_time_s)
            if k.strict_response_time_s is not None
            else "not on record"
        )
        lines.append(
            f"- Constants of the run: target {k.target:g}, knee {k.knee:g}, false alarm budget "
            f"{far} per hour, strict response time {resp}, intruder speed {speeds}, assumed "
            f"task time {seconds(k.task_time_s)}"
        )
    else:
        lines.append(
            "- results.json holds no `constants` block (written before the rescore); the target, "
            "knee and assumed task time are campaign.py's, and intruder speeds are not shown"
        )
    return lines


def section_provenance(results: Row, notes: Row, out: Path) -> list[str]:
    checkpoint = results.get("checkpoint", {})
    lines = ["## Provenance", ""]
    lines += [
        f"- Scenario source commit: `{checkpoint.get('scenario_source_commit', 'unknown')}`",
        f"- Launch commit: `{checkpoint.get('launch_commit', 'unknown')}`; every commit that "
        f"ran: {', '.join(f'`{c}`' for c in checkpoint.get('commits', [])) or 'unknown'}",
        f"- Engine version: {checkpoint.get('engine_version', 'unknown')}",
        *_run_end_lines(results),
        f"- Total: {checkpoint.get('episodes', 'unknown')} episodes, "
        f"{checkpoint.get('quiet_nights', 'unknown')} quiet nights, "
        f"{float(checkpoint.get('wall_s', 0.0)) / 3600:.2f} h of wall time",
    ]
    if checkpoint.get("cache_full"):
        lines.append("- **The run stopped early because the cache was full.**")
    lines += [f"- Lane C tactic note: {n}" for n in checkpoint.get("lane_c_tactic_notes", [])]
    lines += ["", "Hardware prices used (USD per hour):", ""]
    statuses: Row = {}
    for folder in (out, out.parent):
        path = folder / "costs.json"
        if path.is_file():
            try:
                statuses = json.loads(path.read_text())
            except (OSError, ValueError):
                statuses = {}
            break
    for key, value in sorted(results.get("costs", {}).items()):
        entry = statuses.get(key)
        status = entry.get("status", "unknown") if isinstance(entry, dict) else "not read"
        lines.append(f"- `{key}`: {value} (status in costs.json: {status})")
    lines += [
        "",
        "Both purchasable prices are marked `verify` in `costs.json` until someone checks them "
        "against a quote; the cost axis of every table moves with them.",
        "",
        "Merges of main during the run:",
        "",
    ]
    lines += [f"- {m}" for m in notes.get("merges", [])] or ["- none recorded in notes.json"]
    pins = notes.get("check_pins", {})
    lines += [
        "",
        f"- check_pins at the start: {pins.get('start', 'not recorded in notes.json')}",
        f"- check_pins at the end: {pins.get('end', 'not recorded in notes.json')}",
        "",
    ]
    lines += [f"- {text}" for text in notes.get("extra", [])]
    if notes.get("extra"):
        lines.append("")
    return lines


def section_needs_call(notes: Row) -> list[str]:
    items = notes.get("needs_your_call", [])
    lines = ["## Needs your call", ""]
    lines += [f"{i}. {text}" for i, text in enumerate(items, 1)] or ["Nothing recorded."]
    lines.append("")
    return lines


def build_report(results: Row, notes: Row, out: Path) -> tuple[str, list[Row]]:
    book = Book()
    lines = [*summary_lines(book, results), ""]
    lines += [
        "# Campaign report",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{out / 'results.json'}`. Every "
        "number is read from that file and listed in `numbers.json`.",
        "",
    ]
    lines += section_needs_call(notes)
    lines += section_how_to_read(results)
    lines += section_frontier(book, results)
    lines += section_targets(book, results)
    lines += section_a(book, results)
    lines += section_finalists(book, results)
    lines += section_b(book, results)
    lines += section_duty(results)
    lines += section_selection(book, results)
    lines += section_assumptions(book, results)
    lines += section_audit(book, results, notes)
    lines += section_caveats(results)
    lines += section_status(results)
    lines += section_provenance(results, notes, out)
    return "\n".join(lines).rstrip() + "\n", book.records


# ---- charts ----------------------------------------------------------------------------------


def _style(ax: Axes, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=11)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=11)


def _pd_axis(ax: Axes) -> None:
    ax.set_ylim(-0.04, 1.08)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])


def _target_line(ax: Axes, target: float) -> None:
    ax.axhline(target, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate(
        f"target {target:.2f}",
        (1.0, target),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 3),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK_2,
    )


def _err(value: float, ci: Sequence[float] | None) -> list[list[float]]:
    if not ci:
        return [[0.0], [0.0]]
    return [[max(value - float(ci[0]), 0.0)], [max(float(ci[1]) - value, 0.0)]]


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _titles(fig: Any, title: str, subtitle: str, top: float = 0.985) -> None:
    fig.text(0.01, top, title, ha="left", va="top", fontsize=14, color=INK, fontweight="bold")
    fig.text(
        0.01,
        top - 0.3 / fig.get_figheight(),
        subtitle,
        ha="left",
        va="top",
        fontsize=10.5,
        color=INK_2,
    )


def _placeholder(path: Path, title: str, why: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", fontsize=14, color=INK, transform=ax.transAxes)
    ax.text(0.5, 0.35, why, ha="center", fontsize=11, color=INK_2, transform=ax.transAxes)
    _save(fig, path)


def _metric(row: Row, metric: str) -> tuple[float, Sequence[float] | None] | None:
    if metric == "pd":
        return float(row["pd"]), row.get("pd_ci")
    cell = row.get(metric)
    return (float(cell["pd"]), cell.get("ci")) if isinstance(cell, dict) else None


def chart_frontier(results: Row, path: Path) -> None:
    rows = [r for r in stage_rows(results, "final_standin") if not is_ingredient(r)]
    if not rows:
        status, detail = stage_status(results, "final_standin")
        _placeholder(path, "No frontier", f"stage final_standin is {status}: {detail}")
        return
    rec = recommended_label(results)
    costs = [float(r["cost_per_hour"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), sharey=True)
    panels = (("pd", "Overall detection"), ("worst_heldout", "Worst tactic, held-out"))
    for ax, (metric, title) in zip(axes, panels, strict=True):
        _style(ax, "cost, USD per hour", "detection probability" if metric == "pd" else "")
        _pd_axis(ax)
        _target_line(ax, consts(results).target)
        best = -1.0
        for row in sorted(rows, key=lambda r: (float(r["cost_per_hour"]), str(r["label"]))):
            got, hollow = _metric(row, metric), False
            if got is None:
                got, hollow = _metric(row, "worst_naive"), True
            if got is None:
                continue
            value, ci = got
            label = str(row["label"])
            special = label in (BASELINE_LABEL, rec)
            colour = ORANGE if label == BASELINE_LABEL else BLUE if label == rec else CONTEXT
            ax.errorbar(
                [float(row["cost_per_hour"])],
                [value],
                yerr=_err(value, ci),
                fmt="o",
                markersize=9 if special else 7,
                color=colour,
                markerfacecolor=SURFACE if hollow else colour,
                markeredgewidth=2,
                elinewidth=2 if special else 1.2,
                capsize=3,
                zorder=3 if special else 2,
            )
            improves = value > best + 1e-9 and label != FIX_LABEL
            best = max(best, value)
            if special or improves:
                name = (
                    "baseline"
                    if label == BASELINE_LABEL
                    else f"{row['hardware']} (recommended)"
                    if label == rec
                    else str(row["hardware"])
                    if label != BEST_FREE_LABEL
                    else "best free policy"
                )
                right_half = float(row["cost_per_hour"]) > (min(costs) + max(costs)) / 2
                note = ax.annotate(
                    name,
                    (float(row["cost_per_hour"]), value),
                    xytext=(-9 if right_half else 9, -15 if label == BASELINE_LABEL else 9),
                    textcoords="offset points",
                    ha="right" if right_half else "left",
                    fontsize=10,
                    color=INK,
                    fontweight="bold" if special else "normal",
                    zorder=4,
                )
                note.set_in_layout(False)
        ax.set_title(title, loc="left", fontsize=13, color=INK)
        ax.margins(x=0.12)
    first = rows[0]
    _titles(
        fig,
        "Detection against cost",
        f"Strict, {seeds_words(first)}. Bars are 95% intervals.",
    )
    fig.text(
        0.01,
        0.01,
        "Hollow marker: naive worst tactic, shown where no held-out value exists (baseline, "
        "confirmed fix). Grey: other configurations; named where they raise the best so far.",
        fontsize=9,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.88))
    _save(fig, path)


def phase_series(row: Row) -> dict[str, list[tuple[float, float]]]:
    """Mean detection over entries at each phase, per tactic family."""
    cells: dict[str, dict[float, list[float]]] = {}
    for tactic, value in row.get("pd_by_tactic", {}).items():
        parsed = parse_tactic(tactic)
        if parsed is not None:
            cells.setdefault(parsed["family"], {}).setdefault(parsed["phase"], []).append(
                float(value)
            )
    return {
        family: [(phase, sum(v) / len(v)) for phase, v in sorted(by_phase.items())]
        for family, by_phase in cells.items()
    }


PHASE_FAMILIES = (
    ("grid", "fast, on the grid", BLUE, "o"),
    ("slow-grid", "slow, on the grid", ORANGE, "s"),
    ("gap", "fast, into a coverage gap", AQUA, "D"),
)


def chart_phase(results: Row, path: Path) -> None:
    rows = stage_rows(results, "final_standin")
    rec = recommended_label(results)
    picked = [(name, find(rows, name)) for name in dict.fromkeys((BASELINE_LABEL, rec)) if name]
    panels = [(name, row) for name, row in picked if row is not None]
    if not panels:
        status, detail = stage_status(results, "final_standin")
        _placeholder(path, "No phase chart", f"stage final_standin is {status}: {detail}")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.4), sharey=True)
    axes_list = [axes] if len(panels) == 1 else list(axes)
    for index, (ax, (name, row)) in enumerate(zip(axes_list, panels, strict=True)):
        _style(ax, "intruder entry phase (fraction of the schedule cycle)", "")
        _pd_axis(ax)
        ax.set_xlim(-0.03, 1.03)
        series = phase_series(row)
        for family, words, colour, marker in PHASE_FAMILIES:
            points = series.get(family, [])
            if not points:
                continue
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            ax.plot(
                xs,
                ys,
                color=colour,
                linewidth=2 if family != "gap" else 0,
                marker=marker,
                markersize=8,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=words,
                zorder=3,
            )
        shown = f"{name} (recommended)" if name == rec else str(name)
        ax.set_title(shown, loc="left", fontsize=13, color=INK, pad=30)
        if index == 0:
            ax.set_ylabel("detection probability, mean over entries", color=INK_2, fontsize=11)
            legend = ax.legend(
                loc="lower left",
                bbox_to_anchor=(0.0, 1.005),
                ncol=3,
                frameon=False,
                fontsize=10,
                labelcolor=INK,
                borderaxespad=0.0,
            )
            legend.set_in_layout(False)
    first = panels[0][1]
    _titles(
        fig,
        "Detection by intruder entry phase",
        f"Strict, {seeds_words(first)}. Mean over entries of per-tactic point estimates "
        "(no per-tactic intervals on record).",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    _save(fig, path)


def chart_ingredients(results: Row, path: Path) -> None:
    rows = stage_rows(results, "final_standin")
    deltas = stage_data(results, "final_standin").get("paired_vs_baseline", []) if rows else []
    wanted = [
        d
        for d in deltas
        if str(d.get("a", "")).startswith(INGREDIENT_PREFIX) or d.get("a") == BEST_FREE_LABEL
    ]
    if not wanted:
        status, detail = stage_status(results, "final_standin")
        _placeholder(path, "No ingredient chart", f"final_standin is {status}; {detail or 'none'}")
        return
    wanted.sort(key=lambda d: (d.get("a") == BEST_FREE_LABEL, str(d.get("a"))))
    names = [
        "full policy" if d["a"] == BEST_FREE_LABEL else str(d["a"])[len(INGREDIENT_PREFIX) :]
        for d in wanted
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 2.2 + 0.7 * len(wanted)), sharey=True, sharex=True)
    reach = 0.1
    for d in wanted:
        for which in ("pd", "worst"):
            reach = max([reach, *(abs(float(v)) for v in d[f"{which}_delta_ci"])])
            reach = max(reach, abs(float(d[f"{which}_delta"])))
    panels = (("pd", "Overall detection"), ("worst", "Worst tactic (naive)"))
    for ax, (which, title) in zip(axes, panels, strict=True):
        _style(ax, "paired change against the baseline", "")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        positions = list(range(len(wanted)))
        values = [float(d[f"{which}_delta"]) for d in wanted]
        lows = [
            max(v - float(d[f"{which}_delta_ci"][0]), 0.0)
            for v, d in zip(values, wanted, strict=True)
        ]
        highs = [
            max(float(d[f"{which}_delta_ci"][1]) - v, 0.0)
            for v, d in zip(values, wanted, strict=True)
        ]
        ax.barh(positions, values, height=0.5, color=BLUE, zorder=2)
        ax.errorbar(
            values,
            positions,
            xerr=[lows, highs],
            fmt="none",
            ecolor=INK,
            elinewidth=1.5,
            capsize=4,
            zorder=3,
        )
        ax.axvline(0.0, color=INK_2, linewidth=1.0)
        ax.set_yticks(positions, names, fontsize=11, color=INK)
        ax.set_xlim(-reach * 1.15, reach * 1.15)
        ax.set_title(title, loc="left", fontsize=13, color=INK)
    axes[0].invert_yaxis()
    first = find(rows, str(wanted[0]["a"])) or {}
    _titles(
        fig,
        "What each ingredient of the best free policy adds over the baseline",
        f"Strict, {seeds_words(first)}. Whiskers are paired 95% intervals; same hardware and cost.",
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    _save(fig, path)


def chart_sensitivity(results: Row, path: Path) -> None:
    rows = stage_rows(results, "sensitivity")
    resp, task_axis, resp_axis = _sensitivity_axes(rows)
    if resp is None:
        status, detail = stage_status(results, "sensitivity")
        _placeholder(path, "ASSUMPTIONS: no sensitivity chart", f"stage is {status}: {detail}")
        return
    rec = stage_data(results, "sensitivity").get("recommended_label")
    labels = list(dict.fromkeys(str(r["label"]) for r in rows))
    colours = {BASELINE_LABEL: ORANGE, BEST_FREE_LABEL: AQUA}
    markers = {BASELINE_LABEL: "s", BEST_FREE_LABEL: "D"}
    if rec is not None:
        colours[str(rec)], markers[str(rec)] = BLUE, "o"
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), sharey=True)
    columns = (
        (task_axis, "task_time_s", f"task time, s (response time {resp:g} s)"),
        (resp_axis, "response_time_s", "response time, s (zero task time)"),
    )
    metrics = (("pd", "overall detection"), ("worst_naive", "worst tactic, naive"))
    for r_index, (metric, words) in enumerate(metrics):
        for c_index, (axis_rows, key, xlabel) in enumerate(columns):
            ax = axes[r_index][c_index]
            _style(ax, xlabel if r_index == 1 else "", words if c_index == 0 else "")
            _pd_axis(ax)
            for offset, label in enumerate(labels):
                mine = sorted(
                    (r for r in axis_rows if r["label"] == label), key=_by_label_then(key)
                )
                if not mine:
                    continue
                span = max(float(r[key]) for r in axis_rows) - min(float(r[key]) for r in axis_rows)
                nudge = (offset - (len(labels) - 1) / 2) * 0.012 * (span or 1.0)
                xs = [float(r[key]) + nudge for r in mine]
                got = [_metric(r, metric) or (float("nan"), None) for r in mine]
                ys = [g[0] for g in got]
                colour = colours.get(label, CONTEXT)
                ax.plot(xs, ys, color=colour, linewidth=2, zorder=2)
                for x, (y, ci), r in zip(xs, got, mine, strict=True):
                    ax.errorbar(
                        [x],
                        [y],
                        yerr=_err(y, ci),
                        fmt=markers.get(label, "o"),
                        markersize=9,
                        color=colour,
                        markerfacecolor=SURFACE if r.get("strict") else colour,
                        markeredgewidth=2,
                        elinewidth=1.5,
                        capsize=3,
                        zorder=3,
                    )
                if r_index == 0 and c_index == 0:
                    ax.plot(
                        [],
                        [],
                        color=colour,
                        marker=markers.get(label, "o"),
                        linewidth=2,
                        markersize=8,
                        label=label + (" (recommended)" if label == rec else ""),
                    )
            ax.set_xticks(sorted({float(r[key]) for r in axis_rows}))
            ax.margins(x=0.08)
    legend = axes[0][0].legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.04),
        ncol=3,
        frameon=False,
        fontsize=10,
        labelcolor=INK,
        borderaxespad=0.0,
    )
    legend.set_in_layout(False)
    _titles(
        fig,
        "ASSUMPTIONS, not results: task-time and response-time what-ifs",
        f"{seeds_words(rows[0])}. Only the hollow markers are the strict scenario.",
    )
    fig.text(
        0.01,
        0.01,
        "Hollow marker: the strict cell (the only point that is not an assumption). Bars are "
        "95% intervals. Series are nudged sideways so overlapping intervals stay visible.",
        fontsize=9,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    _save(fig, path)


def write_charts(results: Row, folder: Path) -> list[Path]:
    jobs = (
        ("frontier.png", chart_frontier),
        ("phase.png", chart_phase),
        ("ingredients.png", chart_ingredients),
        ("sensitivity.png", chart_sensitivity),
    )
    written = []
    for name, draw in jobs:
        path = folder / name
        try:
            draw(results, path)
        except (KeyError, TypeError, ValueError, IndexError) as err:
            plt.close("all")
            _placeholder(path, f"{name} could not be drawn", f"{type(err).__name__}: {err}")
        written.append(path)
    return written


# ---- driver ----------------------------------------------------------------------------------


def generate(results: Row, notes: Row, out: Path, report_dir: Path) -> list[Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    text, records = build_report(results, notes, out)
    report = report_dir / "REPORT.md"
    report.write_text(text)
    numbers = report_dir / "numbers.json"
    numbers.write_text(json.dumps(records, indent=1, sort_keys=True))
    return [report, numbers, *write_charts(results, report_dir / "charts")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="absolute path to data/campaign")
    parser.add_argument("--results", type=Path, default=None, help="default <out>/results.json")
    parser.add_argument("--report-dir", type=Path, default=None, help="default <out>")
    args = parser.parse_args(argv)
    if not args.out.is_absolute():
        parser.error("--out must be an absolute path")
    results_path = args.results or args.out / "results.json"
    if not results_path.is_file():
        print(f"no results at {results_path}; the campaign has not written them yet")
        return 1
    results = json.loads(results_path.read_text())
    notes_path = args.out / "notes.json"
    notes = json.loads(notes_path.read_text()) if notes_path.is_file() else {}
    for path in generate(results, notes, args.out, args.report_dir or args.out):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
