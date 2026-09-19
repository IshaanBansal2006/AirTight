"""Assemble the eight-slide deck as Marp markdown from numbers.json, token_numbers.json and the rendered charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _fmt(v: float, digits: int = 2) -> str:
    return f"{v:.{digits}f}"


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
        f"# The score\n\n![height:440px]({charts_rel}/cost_vs_detection.png)\n\n<small>{conditions}</small>{watermark}",
        (
            f"# The fix and the re-attack\n\n![height:400px]({charts_rel}/before_after.png)\n\n{ba['baseline']} to {ba['fixed']}: detection {_fmt(ba['pd'][0])} to {_fmt(ba['pd'][1])}; against the re-attacking worst tactic {_fmt(ba['worst_tactic_pd'][0])} to {_fmt(ba['worst_tactic_pd'][1])}.\n\n*Replay clip B: the catch.*{clip_line(clips, 'catch')}{watermark}"
            if ba
            else f"# The fix and the re-attack\n\n*Needs a report with at least two configurations.*{watermark}"
        ),
        (
            f"# Human attention is a cost\n\nHuman decisions per hour: {_fmt(ba['human_decisions_per_hour'][0])} to {_fmt(ba['human_decisions_per_hour'][1])}. Coverage gap: {ba['coverage_gap_s_per_hour'][0]:.0f} to {ba['coverage_gap_s_per_hour'][1]:.0f} s/h.\n\n![height:300px]({charts_rel}/token_cost.png)\n\nAdversary cost per scored configuration: {cmp['ratio']:,.0f}x cheaper than an LLM planning every episode ({cmp['basis']}).{watermark}"
            if ba
            else f"# Human attention is a cost\n\n![height:300px]({charts_rel}/token_cost.png)\n\n{cmp['ratio']:,.0f}x cheaper than an LLM planning every episode ({cmp['basis']}).{watermark}"
        ),
        f"# What we sell, and what is next\n\n- The score, the vulnerability map, and a re-score after purchase\n- Next: learned adversary, calibrated sensors on more platforms, fleet memory under link loss\n\n<small>{conditions}</small>{watermark}",
    ]
    header = "---\nmarp: true\ntheme: default\npaginate: true\n---\n\n"
    return header + "\n\n---\n\n".join(slides) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", type=Path, default=REPO / "pitch" / "charts")
    ap.add_argument("--out", type=Path, default=REPO / "pitch" / "deck.md")
    ap.add_argument("--clips", type=Path, default=REPO / "pitch" / "clips" / "clips.json")
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
    clips = json.loads(args.clips.read_text()) if args.clips.exists() else None
    args.out.write_text(build(numbers, tokens, str(rel), example, clips))
    print(
        f"deck written to {args.out}"
        + (" (EXAMPLE DATA watermark on every slide)" if example else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
