from __future__ import annotations

import argparse
import json
import logging
import sys
from importlib import resources
from pathlib import Path

import numpy as np

from airtight.contracts import FleetConfig, SensorCurves, Site, Tactic
from airtight.redteam.config import RedTeamConfig
from airtight.redteam.coverage import GeometryCoverage
from airtight.redteam.families import FAMILIES, sample_tactic
from airtight.redteam.search import SearchResult, load_seeds, search_all
from airtight.redteam.validate import validate

EXAMPLES = resources.files("airtight.contracts.examples")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _example(name: str) -> Path:
    return Path(str(EXAMPLES.joinpath(name)))


def _add_scene_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--site", type=Path, default=_example("site.json"))
    p.add_argument("--fleet", type=Path, default=_example("fleet_config.json"))
    p.add_argument("--curves", type=Path, default=_example("sensor_curve.json"))
    p.add_argument(
        "--config", type=Path, default=None, help="RedTeamConfig JSON; defaults apply when omitted"
    )


def _load_scene(args: argparse.Namespace) -> tuple[Site, FleetConfig, SensorCurves, RedTeamConfig]:
    site = Site.model_validate_json(args.site.read_text())
    fleet = FleetConfig.model_validate_json(args.fleet.read_text())
    curves = SensorCurves.model_validate_json(args.curves.read_text())
    cfg = (
        RedTeamConfig.model_validate_json(args.config.read_text())
        if args.config
        else RedTeamConfig()
    )
    return site, fleet, curves, cfg


def cmd_sample(args: argparse.Namespace) -> int:
    site, fleet, curves, cfg = _load_scene(args)
    rng = np.random.default_rng(args.seed)
    tactics = [sample_tactic(args.family, site, fleet, curves, rng, cfg) for _ in range(args.n)]
    print(json.dumps([t.model_dump(mode="json") for t in tactics], indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    site, _, _, cfg = _load_scene(args)
    status = 0
    for path in args.tactic:
        errors = validate(Tactic.model_validate_json(path.read_text()), site, cfg)
        print(f"{path}: {'valid' if not errors else '; '.join(errors)}")
        status |= int(bool(errors))
    return status


def cmd_coverage(args: argparse.Namespace) -> int:
    site, _, curves, cfg = _load_scene(args)
    cm = GeometryCoverage(cfg.coverage_cell_m, cfg.dock_halo_m).coverage(site, curves)
    low = cm.low_cells(site)
    if args.out:
        args.out.write_text(cm.model_dump_json())
    print(
        f"{cm.nx}x{cm.ny} cells of {cm.cell_m} m; {len(low)} unwatched cells inside the perimeter; source={cm.source}"
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from airtight.sim.runner import run_episode

    site, fleet, curves, cfg = _load_scene(args)
    if args.n_random:
        cfg.search.n_random = args.n_random
    if args.n_rounds is not None:
        cfg.search.n_rounds = args.n_rounds
    if args.n_seeds:
        cfg.search.n_seeds = args.n_seeds
    seeds = load_seeds(args.seeds, cfg.search.n_seeds)
    seed_tactics = [t for p in args.inject for t in _load_tactics(p)]
    results = search_all(
        site,
        fleet,
        curves,
        seeds,
        run_episode,
        args.log_dir,
        args.out,
        families=args.families,
        cfg=cfg,
        master_seed=args.seed,
        workers=args.workers,
        seed_tactics=seed_tactics,
    )
    for fam, r in results.items():
        t, s = r.best()
        print(
            f"{fam:16s} best={s.adversary_score:.3f} miss={s.miss_rate:.2f} margin={s.mean_margin_s:+.0f}s origin={t.origin} entry={t.entry_id} phase={t.phase:.2f} episodes={r.n_episodes}"
        )
    print(f"written to {args.out}")
    return 0


def _load_tactics(path: Path) -> list[Tactic]:
    """A file holding one Tactic, a JSON list of tactics, or a ProposalBatch."""
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return [Tactic.model_validate(t) for t in raw]
    if "tactics" in raw:
        return [Tactic.model_validate(t) for t in raw["tactics"]]
    return [Tactic.model_validate(raw)]


def _load_prior(directory: Path) -> dict[str, SearchResult]:
    prior: dict[str, SearchResult] = {}
    for fam in FAMILIES:
        f = directory / f"top_{fam}.json"
        if f.exists():
            prior[fam] = SearchResult.model_validate_json(f.read_text())
    return prior


def cmd_propose(args: argparse.Namespace) -> int:
    from airtight.redteam.llm import BudgetExceeded, LlmClient
    from airtight.redteam.proposer import propose

    site, fleet, curves, cfg = _load_scene(args)
    if args.model:
        cfg.llm.model = args.model
    if args.budget is not None:
        cfg.llm.budget_usd = args.budget
    mock = (
        Path(str(resources.files("airtight.redteam.fixtures").joinpath("proposals_mock.json")))
        if args.mock
        else None
    )
    client = LlmClient(cfg.llm, args.cache_dir, args.ledger, mock_path=mock)
    prior = _load_prior(args.prior) if args.prior else None
    try:
        batch = propose(site, fleet, curves, cfg, client, prior, n=args.n)
    except BudgetExceeded as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(batch.model_dump_json(indent=2))
    for t in batch.tactics:
        print(
            f"accepted {t.id}: {t.family} via {t.entry_id} phase={t.phase:.2f} speed={t.speed_mps:.1f} :: {batch.rationales[t.id][:90]}"
        )
    for r in batch.rejected:
        print(f"rejected #{r.index}: {'; '.join(r.errors)[:120]}")
    print(
        f"{len(batch.tactics)} accepted, {len(batch.rejected)} rejected; spent so far ${client.spent_usd():.4f}; written to {args.out}"
    )
    return 0


def _client_from(args: argparse.Namespace, cfg: RedTeamConfig):  # type: ignore[no-untyped-def]
    from airtight.redteam.llm import LlmClient

    if args.model:
        cfg.llm.model = args.model
    if args.budget is not None:
        cfg.llm.budget_usd = args.budget
    mock = (
        Path(str(resources.files("airtight.redteam.fixtures").joinpath("proposals_mock.json")))
        if args.mock
        else None
    )
    return LlmClient(cfg.llm, args.cache_dir, args.ledger, mock_path=mock)


def cmd_difficulty(args: argparse.Namespace) -> int:
    from airtight.redteam.difficulty import check_difficulty
    from airtight.sim.runner import run_episode

    site, fleet, curves, cfg = _load_scene(args)
    seeds = load_seeds(args.seeds, args.n_seeds or cfg.search.n_seeds)
    rep = check_difficulty(
        site,
        fleet,
        curves,
        seeds,
        run_episode,
        args.log_dir,
        cfg,
        args.n_per_family,
        workers=args.workers,
    )
    for r in rep.families:
        print(
            f"{r.family:16s} mean Pd={r.mean_pd:.2f} worst Pd={r.worst_pd:.2f} over {r.n_tactics} random tactics"
        )
    print(rep.verdict)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rep.model_dump_json(indent=2))
    return 0 if rep.in_band() else 3


def cmd_campaign(args: argparse.Namespace) -> int:
    from airtight.redteam.campaign import run_campaign
    from airtight.redteam.llm import BudgetExceeded
    from airtight.sim.runner import run_episode

    site, fleet, curves, cfg = _load_scene(args)
    if args.n_random:
        cfg.search.n_random = args.n_random
    if args.n_rounds is not None:
        cfg.search.n_rounds = args.n_rounds
    seeds = load_seeds(args.seeds, args.n_seeds or cfg.search.n_seeds)
    client = _client_from(args, cfg)
    try:
        result, _ = run_campaign(
            site,
            fleet,
            curves,
            seeds,
            run_episode,
            args.log_dir,
            args.out,
            client,
            cfg,
            args.families,
            args.seed,
            args.workers,
        )
    except BudgetExceeded as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    for f in result.families:
        print(
            f"{f.family:16s} search-only={f.search_only_best:.3f} with-llm={f.with_llm_best:.3f} llm in elites={f.llm_tactic_in_elites} llm is best={f.llm_tactic_is_best}"
        )
    print(
        f"{result.proposals_accepted} accepted, {result.proposals_rejected} rejected, ${result.llm_spent_usd:.4f} spent, {result.n_episodes} episodes; written to {args.out}"
    )
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    from airtight.redteam.accounting import compare_planners, ledger_from, summarize_ledger

    cfg = RedTeamConfig()
    calls = ledger_from(args.ledger, cfg.llm)
    summary = summarize_ledger(calls)
    print(summary.model_dump_json(indent=2))
    cmp = compare_planners(
        cfg.llm, calls, episodes_per_config=args.episodes_per_config, n_configs=args.n_configs
    )
    print(
        f"naive planner (LLM every episode): ${cmp.naive_usd:.2f}; propose-then-search: ${cmp.propose_then_search_usd:.4f}; ratio {cmp.ratio:.0f}x ({cmp.basis})"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="airtight-redteam", description="Lane C adversary tools.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="draw random valid tactics of one family")
    _add_scene_args(s)
    s.add_argument("--family", choices=FAMILIES, required=True)
    s.add_argument("--n", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_sample)

    v = sub.add_parser("validate", help="validate tactic files against a site")
    _add_scene_args(v)
    v.add_argument("tactic", type=Path, nargs="+")
    v.set_defaults(fn=cmd_validate)

    c = sub.add_parser("coverage", help="geometry-only coverage map")
    _add_scene_args(c)
    c.add_argument("--out", type=Path, default=None)
    c.set_defaults(fn=cmd_coverage)

    r = sub.add_parser("search", help="find the worst tactics per family against an episode runner")
    _add_scene_args(r)
    r.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    r.add_argument("--seeds", type=Path, default=REPO_ROOT / "data" / "seeds.json")
    r.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "tactics")
    r.add_argument("--log-dir", type=Path, default=REPO_ROOT / "data" / "search_logs")
    r.add_argument(
        "--inject",
        type=Path,
        nargs="*",
        default=[],
        help="tactic files added to the initial population",
    )
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--seed", type=int, default=0, help="master seed for the search RNG")
    r.add_argument("--n-random", type=int, default=None)
    r.add_argument("--n-rounds", type=int, default=None)
    r.add_argument("--n-seeds", type=int, default=None)
    r.set_defaults(fn=cmd_search)

    q = sub.add_parser(
        "propose", help="ask the LLM for tactics (cache first; --mock never calls the API)"
    )
    _add_scene_args(q)
    q.add_argument("--mock", action="store_true", help="use the bundled fixture instead of the API")
    q.add_argument("--n", type=int, default=None)
    q.add_argument("--model", default=None)
    q.add_argument(
        "--budget", type=float, default=None, help="hard cap in USD over the whole ledger"
    )
    q.add_argument(
        "--prior", type=Path, default=None, help="directory with top_<family>.json from a search"
    )
    q.add_argument(
        "--out", type=Path, default=REPO_ROOT / "data" / "tactics" / "llm_proposals.json"
    )
    q.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "llm_cache")
    q.add_argument("--ledger", type=Path, default=REPO_ROOT / "data" / "llm_calls.jsonl")
    q.set_defaults(fn=cmd_propose)

    d = sub.add_parser(
        "difficulty", help="random tactics against the baseline: is mean Pd in the 0.6 to 0.9 band?"
    )
    _add_scene_args(d)
    d.add_argument("--seeds", type=Path, default=REPO_ROOT / "data" / "seeds.json")
    d.add_argument("--n-seeds", type=int, default=None)
    d.add_argument("--n-per-family", type=int, default=30)
    d.add_argument("--log-dir", type=Path, default=REPO_ROOT / "data" / "search_logs")
    d.add_argument("--workers", type=int, default=1)
    d.add_argument("--out", type=Path, default=None)
    d.set_defaults(fn=cmd_difficulty)

    cp = sub.add_parser(
        "campaign",
        help="search, propose against the results, search again with proposals injected, compare",
    )
    _add_scene_args(cp)
    cp.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    cp.add_argument("--seeds", type=Path, default=REPO_ROOT / "data" / "seeds.json")
    cp.add_argument("--n-seeds", type=int, default=None)
    cp.add_argument("--n-random", type=int, default=None)
    cp.add_argument("--n-rounds", type=int, default=None)
    cp.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "campaign")
    cp.add_argument("--log-dir", type=Path, default=REPO_ROOT / "data" / "search_logs")
    cp.add_argument("--workers", type=int, default=1)
    cp.add_argument("--seed", type=int, default=0)
    cp.add_argument("--mock", action="store_true")
    cp.add_argument("--model", default=None)
    cp.add_argument("--budget", type=float, default=None)
    cp.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "llm_cache")
    cp.add_argument("--ledger", type=Path, default=REPO_ROOT / "data" / "llm_calls.jsonl")
    cp.set_defaults(fn=cmd_campaign)

    ld = sub.add_parser("ledger", help="spend so far and the naive-versus-propose cost comparison")
    ld.add_argument("--ledger", type=Path, default=REPO_ROOT / "data" / "llm_calls.jsonl")
    ld.add_argument("--episodes-per-config", type=int, default=200)
    ld.add_argument("--n-configs", type=int, default=12)
    ld.set_defaults(fn=cmd_ledger)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
