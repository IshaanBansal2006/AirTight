from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

from airtight.sim import debug_plot

if TYPE_CHECKING:
    from pathlib import Path

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_backend_is_headless() -> None:
    assert matplotlib.get_backend().lower() == "agg"


def test_main_writes_a_png_and_creates_the_folder(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "debug.png"
    args = [
        "--tactic",
        "jog",
        "--drones",
        "2",
        "--mode",
        "band",
        "--seed",
        "1000",
        "--out",
        str(out),
    ]
    assert debug_plot.main(args) == 0
    assert out.read_bytes()[:8] == PNG_MAGIC and out.stat().st_size > 20_000


def test_draw_returns_the_same_scores_as_a_plain_run(
    tmp_path: Path, yard_site, yard_curve, fleet_of
):  # type: ignore[no-untyped-def]
    from airtight.sim import scenarios
    from airtight.sim.episode import simulate

    drawn = debug_plot.draw("walk", 1, "asset", 1001, tmp_path / "a.png", "yard_night")
    plain = simulate(yard_site, fleet_of(1), scenarios.load_tactic("walk"), yard_curve, 1001)
    assert drawn == plain
