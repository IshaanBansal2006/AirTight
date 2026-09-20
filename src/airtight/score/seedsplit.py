"""The campaign's seeds, split once into roles that never cross.

Selection and reporting never share seeds. The campaign list holds N_SEEDS seeds from
numpy.random.default_rng([20260919, 77]). Intrusion seeds: search 0-199, validation 200-599,
final 600-1799. Quiet nights use 1800-1999, split the same three ways. Anything chosen on one
role's seeds is reported on another's. load raises on any overlap, and check_role raises when
a stage asks for a seed outside its own role.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

GENERATOR_KEY = (20260919, 77)
N_SEEDS = 2000
ROLES = ("search", "validation", "final")
INTRUSION_RANGES = {"search": (0, 200), "validation": (200, 600), "final": (600, 1800)}
QUIET_RANGES = {"search": (1800, 1820), "validation": (1820, 1860), "final": (1860, 2000)}


@dataclass(frozen=True)
class SeedSplit:
    intrusion: dict[str, tuple[int, ...]]  # role -> seeds
    quiet: dict[str, tuple[int, ...]]

    def check_role(self, role: str, seeds: Sequence[int], quiet_seeds: Sequence[int]) -> None:
        """Raise unless every seed belongs to this role."""
        if role not in ROLES:
            raise ValueError(f"unknown seed role {role!r}; choose one of {ROLES}")
        stray = set(seeds) - set(self.intrusion[role])
        stray_quiet = set(quiet_seeds) - set(self.quiet[role])
        if stray or stray_quiet:
            raise ValueError(
                f"role {role!r} was given seeds from outside its split: "
                f"{sorted(stray)[:5]} intrusion, {sorted(stray_quiet)[:5]} quiet"
            )

    def describe(self) -> dict[str, str]:
        out = {f"{r} intrusion": f"indices {a}-{b - 1}" for r, (a, b) in INTRUSION_RANGES.items()}
        out.update({f"{r} quiet": f"indices {a}-{b - 1}" for r, (a, b) in QUIET_RANGES.items()})
        return out


def generate_seeds() -> list[int]:
    """N_SEEDS distinct seeds from default_rng(GENERATOR_KEY). The same list on any machine."""
    rng = np.random.default_rng(list(GENERATOR_KEY))
    seeds: list[int] = []
    seen: set[int] = set()
    while len(seeds) < N_SEEDS:
        seed = int(rng.integers(0, 2**31 - 1))
        if seed not in seen:
            seen.add(seed)
            seeds.append(seed)
    return seeds


def write_seeds(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = f"numpy.random.default_rng({list(GENERATOR_KEY)})"
    path.write_text(json.dumps({"generator": generator, "seeds": generate_seeds()}))


def split(seeds: Sequence[int]) -> SeedSplit:
    """Split a campaign seed list. Raises on a wrong length, a duplicate, or any overlap."""
    seeds = [int(s) for s in seeds]
    if len(seeds) != N_SEEDS:
        raise ValueError(f"the campaign seed list must hold {N_SEEDS} seeds, got {len(seeds)}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("the campaign seed list holds duplicates, so the splits would overlap")
    out = SeedSplit(
        intrusion={r: tuple(seeds[a:b]) for r, (a, b) in INTRUSION_RANGES.items()},
        quiet={r: tuple(seeds[a:b]) for r, (a, b) in QUIET_RANGES.items()},
    )
    assert_no_overlap(out)
    return out


def assert_no_overlap(parts: SeedSplit) -> None:
    groups = {f"{r} intrusion": set(s) for r, s in parts.intrusion.items()}
    groups.update({f"{r} quiet": set(s) for r, s in parts.quiet.items()})
    names = sorted(groups)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            clash = groups[a] & groups[b]
            if clash:
                raise ValueError(f"seed splits {a!r} and {b!r} overlap: {sorted(clash)[:5]}")


def load(path: Path, create: bool = False) -> SeedSplit:
    """Read and split the campaign list. create=True writes it first when it is missing. A
    list that is not the generator's own output is refused: nobody picks campaign seeds."""
    if create and not path.is_file():
        write_seeds(path)
    seeds = [int(s) for s in json.loads(path.read_text())["seeds"]]
    if seeds != generate_seeds():
        raise ValueError(f"{path} is not the list default_rng({list(GENERATOR_KEY)}) generates")
    return split(seeds)
