# Decision 012: Score-preserving hot-path optimizations

## Context
- Episode loops, quiet nights, red-team search, and clip rendering spend most of their time in a few inner kernels: `mark_seen` / `retarget` over the patrol grid, polyline queries, log replay, coverage maps, JPDA/UKF, and JSONL summaries.
- Official numbers are keyed by `ENGINE_VERSION`. A rewrite that changes which cells are marked, which looks fire, or which RNG draws happen would invalidate caches and the v3 results archive.
- `dt`, `warmup_s`, `BENIGN_HORIZON_S`, search `n_*`, and occupancy cell size stay as they are.

## Decision
Optimize **how** the existing formulas run, not **what** they compute.

| Area | Change | Why it is preserving |
|---|---|---|
| `PatrolController.mark_seen` | Disk bbox; cache the mask for fixed sensors | Cells outside the window cannot be inside the disk; fixed pose is constant |
| `PatrolController.retarget` | One `staleness(t) * weight` per wave | Same array, reused |
| `voronoi_mask` | Broadcast peers instead of a Python loop | Same squared-distance ties (lower id wins) |
| `in_wedge` | Wrap Δbearing with `mod`; scalar path for one point | Same `[-π, π]` comparison |
| `Polyline` | Cache segment lengths on each object | Same positions as a fresh `polyline_position` |
| `do_looks` | Position each alive object once; one `pd_per_look` for targets | Same look order, same RNG draws, same LLR |
| `spawn_benign` | One `Polyline` shared by every object on a route | Same arrivals, same positions |
| `iter_ticks` / clip HTML | `bisect` on sorted timestamps | Same last-pose-at-or-before rule, O(T log T) instead of O(T²) |
| `snapshot_staleness` | `np.nonzero` over the weighted stale mask | Same row-major cell order |
| `GeometryCoverage` | Meshgrid range/FOV | Same range + half-FOV tests |
| `summarize_log` | One JSONL pass | Same peaks |
| logreport ROC | Vectorized threshold grid | Same counts; bootstrap draws batched |
| `FleetMemoryStore.query` | Index by kind | Same items, no JSON change |
| JPDA / UKF / Hungarian inner loops | `solve` / broadcast sigma / vectorized Dijkstra step | Algebraically the same |

`ENGINE_VERSION` stays `"1"`.

## Reason
- Quiet nights are ~14k steps × observers × 2400 cells. A disk window cuts that to the footprint, which is the dominant win.
- Replay HTML of a 2 Hz log was a reverse scan per tick; bisect is the same lookup.
- Vectorizing coverage and Voronoi removes Python per-cell loops without changing the predicates tests already lock (wedge edges, Voronoi ties, look-order / score-floor non-commutativity).

## Consequences
- Do **not** swap `argsort(..., kind="stable")` for `argpartition` without bumping `ENGINE_VERSION` (tie sets can change).
- Do **not** thin `mark_seen` during warm-up, coarsen the patrol grid, or skip looks: those change patrol targets and detection times.
- Remaining cost that would change scores if touched: `BENIGN_HORIZON_S`, search population sizes, thinning `mark_seen` during warm-up, coarser patrol/occupancy cells, `argpartition` for top-k (tie sets), JPDA truncation (`max_events`).
- Remaining cost that is I/O or API, not the formulas: full-log JSONL (pydantic events at 2 Hz plus every look), matplotlib MP4 rendering, occupancy rasters at 0.05 m for MuJoCo.
- `tests/sim/test_hotpath_opts.py` locks window ≡ full-grid `mark_seen`, cached polylines, and bisect ≡ reverse scan.

## Follow-up: skip I/O unless a replay needs it

- v0 `run_config(..., prune_logs=True)` (CLI default) reads peaks from `EpisodeScores` and writes no JSONL. `--keep-logs` is the old full-log path. Replay export and miss/catch clips still request `full_log=True`, and clip search uses header+outcome until a pair is found.
- `--handoff` writes HTML + `clips.json`; pass `--mp4` for ffmpeg.
- Evidence conflicts compare a field tuple, not two `model_dump_json()` strings. `delta` caches the JSONL line after the first serialize.
