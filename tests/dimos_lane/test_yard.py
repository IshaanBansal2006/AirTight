from __future__ import annotations

from pathlib import Path  # noqa: TC003

from airtight.dimos_lane.person import default_intruder_xy, go2_spawn_xy, mujoco_start_pos
from airtight.dimos_lane.site_io import load_example_site
from airtight.dimos_lane.yard import (
    apply_dimsim,
    default_benign_blobs,
    is_occupied,
    occupancy_counts,
    perimeter_edges,
    site_to_occupancy,
    write_occupancy_npy,
)


class _FakeDimSim:
    def __init__(self) -> None:
        self.walls: list[str] = []
        self.objects: list[str] = []

    def add_wall(self, *args: object, **kwargs: object) -> None:
        self.walls.append(str(kwargs.get("name", "")))

    def add_object(self, *args: object, **kwargs: object) -> None:
        self.objects.append(str(kwargs.get("name", "")))


def test_yard_has_walls_asset_docks_and_two_benign() -> None:
    site = load_example_site()
    grid = site_to_occupancy(site)
    occupied, free = occupancy_counts(grid)
    assert grid.shape == (1600, 2400)
    assert occupied > 0 and free > occupied
    assert is_occupied(grid, site, site.asset.x, site.asset.y)
    spawn = go2_spawn_xy(site)
    assert not is_occupied(grid, site, spawn.x, spawn.y)
    intruder = default_intruder_xy(site)
    assert not is_occupied(grid, site, intruder.x, intruder.y)
    for blob in default_benign_blobs():
        assert is_occupied(grid, site, blob.x, blob.y)
    # interior cell away from walls, asset, docks, and benign blobs
    assert not is_occupied(grid, site, 80.0, 30.0)
    assert is_occupied(grid, site, site.perimeter[0].x, site.perimeter[0].y)


def test_write_occupancy_npy_round_trip(tmp_path: Path) -> None:
    site = load_example_site()
    path = write_occupancy_npy(site, tmp_path / "yard_occupancy.npy")
    import numpy as np

    loaded = np.load(path)
    assert loaded.shape == site_to_occupancy(site).shape
    assert mujoco_start_pos(site) == "10.0,70.0"


def test_mujoco_crop_keeps_spawn_wall_and_intruder(tmp_path: Path) -> None:
    from airtight.dimos_lane.yard import crop_to_radius, write_mujoco_occupancy_npy

    site = load_example_site()
    spawn = go2_spawn_xy(site)
    full = site_to_occupancy(site)
    cropped = crop_to_radius(full, site, spawn.x, spawn.y, 18.0)
    assert is_occupied(cropped, site, site.perimeter[0].x, 70.0) or is_occupied(
        cropped, site, 5.0, spawn.y
    )
    assert not is_occupied(cropped, site, spawn.x, spawn.y)
    assert not is_occupied(cropped, site, default_intruder_xy(site).x, default_intruder_xy(site).y)
    # far asset is dropped so MuJoCo lidar stays small
    assert not is_occupied(cropped, site, site.asset.x, site.asset.y)
    path = write_mujoco_occupancy_npy(site, tmp_path / "yard_occupancy_mujoco.npy")
    assert path.is_file()


def test_apply_dimsim_places_walls_markers_and_benign() -> None:
    site = load_example_site()
    client = _FakeDimSim()
    result = apply_dimsim(client, site)  # type: ignore[arg-type]
    assert len(client.walls) == len(perimeter_edges(site))
    assert "asset" in client.objects
    for dock in site.docks:
        assert dock.id in client.objects
    for blob in default_benign_blobs():
        assert blob.name in client.objects
        assert blob.name in result["benign"]
