"""One episode, drawn: python -m airtight.sim.debug_plot --tactic jog --drones 2 --out data/debug.png

Left: the fence, asset, docks, the critical ring of radius r_c round the asset, each agent's
trail, the intruder's path with a marker where it is at t_cdp, and benign paths. Right: the
intruder's score and each benign object's score against time, with TAU_REF and t_cdp marked.

Uses matplotlib's Agg backend, so it runs headless.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from airtight.sim import adapt, scenarios  # noqa: E402
from airtight.sim.constants import TAU_REF  # noqa: E402
from airtight.sim.episode import EpisodeParams, simulate  # noqa: E402
from airtight.sim.geometry import WEIGHT_MODES, polyline_position  # noqa: E402
from airtight.sim.recorder import timely_at_ref  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from airtight.contracts import FleetConfig
    from airtight.sim.episode import EpisodeScores
    from airtight.sim.sensing import Look

    Array = npt.NDArray[np.float64]

INTRUDER_ID = "intruder"


class TrailRecorder:
    """Remembers every pose and every score so they can be drawn."""

    def __init__(self) -> None:
        self.trails: dict[str, list[tuple[float, float]]] = {}
        self.scores: dict[str, list[tuple[float, float]]] = {}
        self.result: EpisodeScores | None = None

    def on_poses(self, t: float, poses: Mapping[str, Array]) -> None:
        for object_id, xy in poses.items():
            self.trails.setdefault(object_id, []).append((float(xy[0]), float(xy[1])))

    def on_looks(self, t: float, looks: Sequence[Look]) -> None:
        for look in looks:
            self.scores.setdefault(look.object_id, []).append((t, look.score))

    def on_finish(self, scores: EpisodeScores) -> None:
        self.result = scores


def fleet_of_drones(n: int, scenario: str) -> FleetConfig:
    """n copies of the scenario's first drone, ids d0..d(n-1)."""
    template = scenarios.load_fleet(scenarios.names("fleet", scenario)[0], scenario)
    drone = template.agents[0]
    agents = [drone.model_copy(update={"id": f"d{i}"}) for i in range(n)]
    return template.model_copy(update={"name": f"{n}-drone-debug", "agents": agents})


def draw(
    tactic_name: str, drones: int, mode: str, seed: int, out: Path, scenario: str
) -> EpisodeScores:
    site = scenarios.load_site(scenario)
    tactic = scenarios.load_tactic(tactic_name, scenario)
    curves = scenarios.load_sensor_curves(scenario)
    fleet = fleet_of_drones(drones, scenario)
    params = EpisodeParams(weight_mode=mode)
    recorder = TrailRecorder()
    scores = simulate(site, fleet, tactic, curves, seed, params, recorder)
    agent_ids = set(adapt.agent_ids(fleet))

    fig, (yard, chart) = plt.subplots(1, 2, figsize=(15, 6.5))

    xmin, ymin, xmax, ymax = adapt.bounds(site)
    yard.add_patch(Polygon(adapt.perimeter(site), closed=True, fill=False, lw=2, ec="black"))
    r_c = adapt.critical_radius_m(site, params.v_ref_mps)
    for ax_, ay_ in adapt.assets(site):
        yard.add_patch(Circle((ax_, ay_), r_c, fill=False, ls="--", ec="crimson", lw=1.2))
        yard.plot(ax_, ay_, marker="*", ms=16, color="gold", mec="black", ls="none")
    yard.plot([], [], ls="--", color="crimson", label=f"critical ring r_c = {r_c:.0f} m")
    yard.plot([], [], marker="*", ms=12, color="gold", mec="black", ls="none", label="asset")
    docks = adapt.docks(site)
    yard.plot(docks[:, 0], docks[:, 1], "s", ms=8, color="grey", mec="black", label="dock")

    for object_id in sorted(recorder.trails):
        trail = np.array(recorder.trails[object_id])
        if object_id in agent_ids:
            yard.plot(trail[:, 0], trail[:, 1], lw=1.0, alpha=0.8, label=f"{object_id} trail")
        elif object_id != INTRUDER_ID:
            yard.plot(trail[:, 0], trail[:, 1], ":", lw=2, color="green")
            yard.annotate(object_id, trail[0], fontsize=8, color="green")

    path = adapt.intruder_path(site, tactic)
    yard.plot(path[:, 0], path[:, 1], "-", lw=2.5, color="red", label="intruder path")
    at_cdp, _ = polyline_position(path, adapt.intruder_speed_mps(tactic), scores.t_cdp)
    yard.plot(*at_cdp, marker="X", ms=13, color="red", mec="black", ls="none")
    yard.plot(
        [], [], marker="X", ms=10, color="red", mec="black", ls="none", label="intruder at t_cdp"
    )
    yard.set_xlim(xmin, xmax)
    yard.set_ylim(ymin, ymax)
    yard.set_aspect("equal")
    yard.set_xlabel("x (m)")
    yard.set_ylabel("y (m)")
    yard.set_title(f"{site.name}: {drones} drone(s), tactic {tactic.id}, weight mode {mode}")
    yard.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8)

    for object_id in sorted(recorder.scores):
        series = np.array(recorder.scores[object_id])
        if object_id == INTRUDER_ID:
            chart.step(
                series[:, 0], series[:, 1], where="post", color="red", lw=2, label="intruder"
            )
        else:
            chart.step(series[:, 0], series[:, 1], where="post", lw=1.2, label=object_id)
    chart.axhline(TAU_REF, color="black", ls="--", lw=1, label=f"TAU_REF = {TAU_REF}")
    chart.axvline(
        scores.t_cdp, color="crimson", ls="--", lw=1, label=f"t_cdp = {scores.t_cdp:.1f} s"
    )
    chart.axvline(
        scores.t_reach, color="grey", ls=":", lw=1, label=f"t_reach = {scores.t_reach:.1f} s"
    )
    chart.set_xlim(0.0, scores.t_end)
    chart.set_xlabel("t (s), 0 = intruder on its entry point")
    chart.set_ylabel("track score (log-likelihood ratio)")
    verdict = "TIMELY" if timely_at_ref(scores) else "not timely"
    alarm = scores.intruder_t_alarm_ref
    alarm_text = f"alarm at {alarm:.2f} s" if alarm is not None else "no alarm"
    chart.set_title(f"seed {seed}: {verdict}, {alarm_text}, {scores.n_looks} looks")
    chart.legend(loc="center right", fontsize=8)
    if not recorder.scores:
        chart.text(0.5, 0.5, "nothing was ever looked at", ha="center", transform=chart.transAxes)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", default=scenarios.DEFAULT_SCENARIO)
    parser.add_argument("--tactic", default="jog", help="a tactic name in the scenario")
    parser.add_argument("--drones", type=int, default=2)
    parser.add_argument("--mode", default="asset", choices=WEIGHT_MODES)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("data/debug.png"))
    args = parser.parse_args(argv)
    scores = draw(args.tactic, args.drones, args.mode, args.seed, args.out, args.scenario)
    print(f"wrote {args.out}: timely={timely_at_ref(scores)} t_alarm={scores.intruder_t_alarm_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
