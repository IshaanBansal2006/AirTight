# Decision 002: Workspace Layout And Dependencies

## Context
- dimOS is a 20k-file platform checkout with its own venv, torch CUDA wheels and a `uv.lock` with override dependencies; drone-swarm-autonomy is a public sibling repo that must not be modified from here.
- dimOS discovers external blueprints through the `dimos.blueprints` entry-point group, so airtight must be installed in the same interpreter as dimos.
- Three people work in parallel for 24 hours and need one setup command.

## Decision
airtight owns its own uv venv; dimOS and drone-swarm-autonomy stay as sibling checkouts consumed as editable path dependencies, pinned by commit in `pins.toml` and checked by `scripts/check_pins.sh`.

## Reason
- A vendored copy or subtree would fork interview-critical code and pull the whole dimos tree into this repo; a submodule adds ergonomics cost for no pinning benefit over `pins.toml`.
- Installing airtight into the dimos venv would be wiped by any exact `uv sync` there; owning the venv makes `uv sync` in this repo the single setup step.
- Editable path deps mean upstream fixes flow immediately and the no-modification rule is enforced by construction: everything is wrapped or subclassed.

## Consequences
- `pyproject.toml` replicates dimos's `[tool.uv]` override dependencies and the PyTorch cu128 index, because uv does not inherit those from a path dependency; when dimos changes them, copy the change.
- Neither sibling has release tags, so the pin is a commit; drift fails `scripts/check_pins.sh`.
- Lane B builds on drone-swarm-autonomy's headless harness, not on vlm-swarm-coverage, so the CBBA allocator's drone-typed agent interface must be adapted for guards and the Go2 inside `src/airtight/sim/`.
