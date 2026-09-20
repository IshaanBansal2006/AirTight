"""Assemble the eight-slide deck as Marp markdown from numbers.json, token_numbers.json and the rendered charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def pretty_fleet(name: str) -> str:
    """d3_go2_guard_stagger -> '3 drones, Go2, guard, staggered'; older run names with +drone suffixes count the suffixes."""
    import re

    base, _, extra = name.partition("+")
    m = re.match(r"d(\d+)_(go2|nogo2)_(guard|noguard)_(sync|stagger)", base)
    if not m:
        return name.replace("_", " ")
    n = int(m.group(1)) + extra.count("drone")
    bits = [f"{n} drone" + ("s" if n != 1 else "")]
    if m.group(2) == "go2" or "go2" in extra:
        bits.append("Go2")
    if m.group(3) == "guard" or "guard" in extra:
        bits.append("guard")
    return ", ".join(bits) + (", staggered" if m.group(4) == "stagger" else ", synchronized")


def blind_line(ba: dict) -> str:
    """The same worst tactics against an adversary that does not know the charge schedule, when the report has it."""
    pair = ba.get("worst_tactic_pd_schedule_blind") or [None, None]
    if pair[0] is None or pair[1] is None:
        return ""
    return f"; the same tactics without the charge schedule {_fmt(pair[0])} to {_fmt(pair[1])}"


def _fmt(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}"


def load_clip_facts(clips_dir: Path) -> dict | None:
    """Lane C's clips.json, or lane A's miss.json and catch.json sidecars folded into the same shape."""
    for name in ("clips_c.json", "clips.json"):
        candidate = clips_dir / name
        if candidate.exists():
            data = json.loads(candidate.read_text())
            if {"tactic_id", "seed", "baseline", "fixed", "t_cdp"} <= set(data):
                return data
    miss, catch = clips_dir / "miss.json", clips_dir / "catch.json"
    if not (miss.exists() and catch.exists()):
        return None
    m, c = json.loads(miss.read_text()), json.loads(catch.read_text())
    return {
        "tactic_id": c.get("tactic_id", m.get("tactic_id", "?")),
        "seed": c.get("seed", m.get("seed", "?")),
        "baseline": m.get("fleet", "baseline fleet"),
        "fixed": c.get("fleet", "fixed fleet"),
        "miss_t_alarm": m.get("t_alarm"),
        "catch_t_alarm": c.get("t_alarm") if c.get("t_alarm") is not None else float("nan"),
        "t_cdp": c.get("t_cdp", m.get("t_cdp", float("nan"))),
    }


def clip_line(clips: dict | None, which: str) -> str:
    if not clips:
        return ""
    if which == "miss":
        return f"\n\nClip A, seed {clips['seed']}: {clips['baseline']} never raised a timely alarm against `{clips['tactic_id']}` (deadline {clips['t_cdp']:.0f} s)."
    return f"\n\nClip B, same seed and tactic: {clips['fixed']} alarmed at {clips['catch_t_alarm']:.0f} s, before the {clips['t_cdp']:.0f} s deadline."


def build(
    numbers: dict, tokens: dict, charts_rel: str, example: bool, clips: dict | None = None
) -> str:
    cond = numbers["conditions"]
    conditions = (
        f"Operating point {cond['far_per_hour_operating_point']:g} false alarm/h · {cond['n_seeds']} seeds · "
        f"adversary: {cond['adversary_knowledge']} · sensor: {cond['sensor_calibration']} · {cond['detection_model_note']}"
    )
    watermark = "\n\n**EXAMPLE DATA, NOT A RESULT**" if example else ""
    ba = numbers.get("before_after")
    worst = numbers.get("worst_tactics", {})
    cw = worst.get("charging_window")
    cmp = tokens["comparison"]
    mm = numbers.get("minmax") or []
    minmax_slide = (
        (
            "# Attack, fix, re-attack\n\nEach round the red team searches the current fleet, every affordable fix is scored against what it found, the best worst case wins, and the red team attacks again.\n\n"
            "| Round | Fleet | $/h | Worst case after re-attack | Fix chosen |\n|---|---|---|---|---|\n"
            + "\n".join(
                f"| {i['k']} | {pretty_fleet(i['fleet'])} | {i['cost']:.0f} | {i['worst_pd']:.2f} | {(i.get('move') or 'stop').replace('_', ' ')} |"
                for i in mm
            )
            + "\n\nThe re-attack number stays near zero: with the charge schedule in hand and no reaction from the fleet, the adversary finds a new hole after every fix. Hide the schedule and the same tactics land far less often, which is the number a buyer can act on. The score reports the typical intruder and the worst case side by side, and the loop is how a site finds the next hole before an intruder does."
            + watermark
        )
        if mm
        else None
    )
    sb = numbers.get("schedule_blind") or {}
    schedule_slide = (
        f"# What hiding the schedule is worth\n\n![height:440px]({charts_rel}/schedule_blind.png)\n\nSame fleets, same worst tactics. The orange point assumes the adversary has the charge schedule to the second; the blue point gives it the site and nothing else. The gap is what a site buys by keeping its schedule private, and the score reports both.{watermark}"
        if sb and ba
        else None
    )
    slides = [
        f"# Airtight\n\n## How secure is this site, and what should you buy?\n\nA security score and a vulnerability map for building owners, insurers and security firms, before any robot is purchased, and a re-score after.{watermark}",
        f"# Site twin and mixed fleet\n\n![height:480px]({charts_rel}/vulnerability_map.png)\n\nDrones, a ground robot and guards bid in one auction. Batteries and docks make coverage continuity real.{watermark}",
        "# The red team\n\nFour tactic families: charging window, decoy, blind spot, comms cut.\n\nAn LLM proposes; search attacks. Every tactic passes one validator.\n\n"
        + "\n".join(
            f"- **{fam}**: entry `{t['entry']}`, phase {t['phase']:.2f}, {t['speed_mps']:.1f} m/s, origin {t['origin']}"
            for fam, t in worst.items()
        )
        + watermark,
        (
            f"# What the adversary found\n\nCharging-window attack: enter `{cw['entry']}` at phase {cw['phase']:.2f} of the charge cycle at {cw['speed_mps']:.1f} m/s.\n\n*Replay clip A: the miss.*{clip_line(clips, 'miss')}{watermark}"
            if cw
            else f"# What the adversary found\n\n*Run the search to populate this slide.*{watermark}"
        ),
        f"# The score\n\n![height:470px]({charts_rel}/cost_vs_detection.png){watermark}",
        (
            f"# The fix and the re-attack\n\n![height:400px]({charts_rel}/before_after.png)\n\n{ba['baseline']} to {ba['fixed']}: detection {_fmt(ba['pd'][0])} to {_fmt(ba['pd'][1])}; against the re-attacking worst tactic {_fmt(ba['worst_tactic_pd'][0])} to {_fmt(ba['worst_tactic_pd'][1])}{blind_line(ba)}.\n\n*Replay clip B: the catch.*{clip_line(clips, 'catch')}{watermark}"
            if ba
            else f"# The fix and the re-attack\n\n*Needs a report with at least two configurations.*{watermark}"
        ),
        (
            f"# Human attention is a cost\n\nHuman decisions per hour: {_fmt(ba['human_decisions_per_hour'][0])} to {_fmt(ba['human_decisions_per_hour'][1])}. Coverage gap: {ba['coverage_gap_s_per_hour'][0]:.0f} to {ba['coverage_gap_s_per_hour'][1]:.0f} s/h.\n\n![height:300px]({charts_rel}/token_cost.png)\n\nAdversary cost per scored configuration: {cmp['ratio']:,.0f}x cheaper than an LLM planning every episode ({cmp['basis']}).{watermark}"
            if ba
            else f"# Human attention is a cost\n\n![height:300px]({charts_rel}/token_cost.png)\n\n{cmp['ratio']:,.0f}x cheaper than an LLM planning every episode ({cmp['basis']}).{watermark}"
        ),
        *([schedule_slide] if schedule_slide else []),
        *([minmax_slide] if minmax_slide else []),
        f"# What we sell, and what is next\n\n- The score, the vulnerability map, and a re-score after purchase\n- Next: learned adversary, calibrated sensors on more platforms, fleet memory under link loss\n\n<small>{conditions}</small>{watermark}",
    ]
    header = (
        "---\nmarp: true\ntheme: default\npaginate: true\n"
        "style: |\n"
        "  section { background: #fcfcfb; color: #0b0b0b; font-family: ui-sans-serif, system-ui, sans-serif; padding: 48px 64px; }\n"
        "  h1 { color: #0b0b0b; font-size: 1.7em; margin-bottom: 0.2em; }\n"
        "  h2 { color: #52514e; font-weight: 500; font-size: 1.1em; }\n"
        "  a, strong { color: #2a78d6; }\n"
        "  small { color: #898781; font-size: 0.6em; }\n"
        "  code { background: #f0efe9; color: #0b0b0b; }\n"
        "  section::after { color: #898781; }\n"
        "  img { display: block; margin: 0 auto; }\n"
        "---\n\n"
    )
    return header + "\n\n---\n\n".join(slides) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", type=Path, default=REPO / "pitch" / "charts")
    ap.add_argument("--out", type=Path, default=REPO / "pitch" / "deck.md")
    ap.add_argument("--clips-dir", type=Path, default=REPO / "pitch" / "clips")
    args = ap.parse_args(argv)
    numbers_path = args.charts / "numbers.json"
    tokens_path = args.charts / "token_numbers.json"
    if not numbers_path.exists() or not tokens_path.exists():
        raise SystemExit(
            f"missing {numbers_path.name} or {tokens_path.name} in {args.charts}; run make_charts.py and make_token_chart.py first"
        )
    numbers = json.loads(numbers_path.read_text())
    tokens = json.loads(tokens_path.read_text())
    example = "examples" in numbers.get("report", "")
    rel = (
        Path(args.charts).resolve().relative_to(args.out.resolve().parent)
        if args.charts.resolve().is_relative_to(args.out.resolve().parent)
        else args.charts
    )
    clips = load_clip_facts(args.clips_dir)
    args.out.write_text(build(numbers, tokens, str(rel), example, clips))
    print(
        f"deck written to {args.out}"
        + (" (EXAMPLE DATA watermark on every slide)" if example else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
