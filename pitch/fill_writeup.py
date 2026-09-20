"""Fill the bracketed slots in writeup.md from numbers.json and token_numbers.json; writes writeup_filled.md."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _blind(ba: dict, i: int) -> str:
    pair = ba.get("worst_tactic_pd_schedule_blind") or [None, None]
    return f"{pair[i]:.2f}" if pair[i] is not None else "?"


def values(numbers: dict, tokens: dict) -> dict[str, str]:
    ba = numbers.get("before_after") or {}
    cvd = numbers.get("cost_vs_detection") or {}
    base_name = ba.get("baseline")
    cost = cvd.get(base_name, {}).get("cost_per_hour") if base_name else None
    worst = numbers.get("worst_tactics", {}).get("charging_window", {})
    return {
        "ratio": f"{tokens['comparison']['ratio']:,.0f}",
        "pd_baseline": f"{ba['pd'][0]:.2f}" if ba else "?",
        "pd_fixed": f"{ba['pd'][1]:.2f}" if ba else "?",
        "worst_baseline": f"{ba['worst_tactic_pd'][0]:.2f}" if ba else "?",
        "worst_fixed": f"{ba['worst_tactic_pd'][1]:.2f}" if ba else "?",
        "blind_baseline": _blind(ba, 0),
        "blind_fixed": _blind(ba, 1),
        "decisions_baseline": f"{ba['human_decisions_per_hour'][0]:.1f}" if ba else "?",
        "decisions_fixed": f"{ba['human_decisions_per_hour'][1]:.1f}" if ba else "?",
        "cost": f"{cost:.0f}" if cost is not None else "?",
        "n_seeds": str(numbers["conditions"]["n_seeds"]),
        "fixed_config": ba.get("fixed", "?"),
        "baseline_config": ba.get("baseline", "?"),
        "pd_before": "?",
        "pd_sync": "?",
        "worst_entry": worst.get("entry", "?"),
        "worst_phase": f"{worst['phase']:.2f}" if worst else "?",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", type=Path, default=REPO / "pitch" / "charts")
    ap.add_argument("--src", type=Path, default=REPO / "pitch" / "writeup.md")
    ap.add_argument("--out", type=Path, default=REPO / "pitch" / "writeup_filled.md")
    args = ap.parse_args(argv)
    numbers = json.loads((args.charts / "numbers.json").read_text())
    tokens = json.loads((args.charts / "token_numbers.json").read_text())
    table = values(numbers, tokens)
    text = args.src.read_text()
    unfilled: list[str] = []

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key in table and table[key] != "?":
            return table[key]
        unfilled.append(key)
        return m.group(0)

    filled = re.sub(r"\[([a-z_]+)\]", sub, text)
    example = "examples" in numbers.get("report", "")
    if example:
        filled = (
            "> EXAMPLE DATA, NOT A RESULT: numbers below come from the packaged example report.\n\n"
            + filled
        )
    args.out.write_text(filled)
    print(f"wrote {args.out}; unfilled slots: {sorted(set(unfilled)) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
