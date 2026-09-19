"""Cost of an LLM that plans every episode versus one that proposes once per configuration, from the real ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from palette import DARK, LIGHT

from airtight.redteam.accounting import compare_planners, ledger_from, summarize_ledger
from airtight.redteam.config import RedTeamConfig

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, default=REPO / "data" / "llm_calls.jsonl")
    ap.add_argument("--episodes-per-config", type=int, default=200)
    ap.add_argument("--n-configs", type=int, default=12)
    ap.add_argument("--out", type=Path, default=REPO / "pitch" / "charts")
    ap.add_argument("--dark", action="store_true")
    args = ap.parse_args(argv)
    cfg = RedTeamConfig().llm
    calls = ledger_from(args.ledger, cfg) if args.ledger.exists() else []
    cmp = compare_planners(cfg, calls, args.episodes_per_config, args.n_configs)
    p = DARK if args.dark else LIGHT
    plt.rcParams.update(
        {
            "figure.facecolor": p.surface,
            "axes.facecolor": p.surface,
            "text.color": p.ink,
            "axes.edgecolor": p.axis,
            "xtick.color": p.muted,
            "ytick.color": p.muted,
            "savefig.dpi": 200,
            "savefig.facecolor": p.surface,
        }
    )
    fig, ax = plt.subplots(figsize=(7, 3))
    labels = ["LLM plans every episode", "LLM proposes, search attacks"]
    values = [cmp.naive_usd, cmp.propose_then_search_usd]
    ax.barh(labels, values, color=[p.muted, p.series[0]], height=0.5)
    for y, v in enumerate(values):
        ax.annotate(
            f"${v:,.2f}" if v >= 0.01 else f"${v:.4f}",
            (v, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=p.ink,
        )
    ax.set_xscale("log")
    ax.set_xlabel(
        f"USD to score {cmp.n_configs} configurations at {cmp.episodes_per_config} episodes each (log scale)",
        color=p.ink_secondary,
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=p.grid, linewidth=0.6)
    ax.set_title(
        f"{cmp.ratio:,.0f}x cheaper: adversary cost per scored configuration",
        loc="left",
        fontweight="bold",
        color=p.ink,
    )
    fig.text(
        0.01,
        0.01,
        f"Per-call cost ${cmp.usd_per_call:.5f} ({cmp.basis}); model {cfg.model}.",
        fontsize=7,
        color=p.muted,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "token_cost.png")
    (args.out / "token_numbers.json").write_text(
        json.dumps(
            {"comparison": cmp.model_dump(), "ledger": summarize_ledger(calls).model_dump()},
            indent=2,
        )
    )
    print(
        f"token chart written; naive ${cmp.naive_usd:.2f} vs propose-then-search ${cmp.propose_then_search_usd:.4f} ({cmp.basis})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
