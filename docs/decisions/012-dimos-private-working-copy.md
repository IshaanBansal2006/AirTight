# Decision 012: dimOS Private Working Copy

## Context
- Two coworkers are starting on dimOS internals: algorithm design and optimization, and the cheapest configuration that holds accuracy 95 percent of the time. dimOS is tuned for a ground robot; the demo fleet is drones.
- Decision 002 pinned upstream dimOS by commit and kept edits as patch files, which only works for a handful of changes.
- The integration must not change: AirTight's modules, blueprints and skills must keep working against the tuned dimOS without edits.

## Decision
dimOS lives in a private repository under the team's account, `IshaanBansal2006/dimos-airtight`, with upstream's full history and a working branch `drone-autonomy`. AirTight's sibling checkout points at it; upstream stays a remote for merges and for LFS assets. Inner algorithms, the memory layer and tuning may change; the public surface listed in the repository's `AIRTIGHT.md` may not.

## Reason
- A GitHub fork of a public repository is forced public; the disclosure rule keeps new repositories private until the user says otherwise.
- A real branch with pull requests is how two people tune the same code without stepping on each other; patch files are not.
- Freezing the public surface is what keeps the fork integrable later: AirTight pins a commit of the branch and nothing else changes.

## Consequences
- `pins.toml` records the repository and branch; the pinned commit moves when a tuned build is adopted.
- `patches/dimos/` is retired; `scripts/dimos_patch.sh` stays only to apply the empty set.
- LFS assets are still served by upstream through the repository's `.lfsconfig`, so clones of the working copy fetch models and scenes without a second LFS store.
