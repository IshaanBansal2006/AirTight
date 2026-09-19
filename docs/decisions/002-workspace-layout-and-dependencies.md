# Decision 002: Workspace Layout And Dependencies

## Context
- dimOS is a 20k-file platform checkout with its own venv, torch CUDA wheels and a `uv.lock` with override dependencies.
- dimOS discovers external blueprints through the `dimos.blueprints` entry-point group, so airtight must be installed in the same interpreter as dimos.
- Three people work in parallel for 24 hours and need one setup command.

## Decision
airtight owns its own uv venv; dimOS stays a sibling checkout consumed as an editable path dependency, pinned by commit in `pins.toml` and checked by `scripts/check_pins.sh`.

## Reason
- Installing airtight into the dimos venv would be wiped by any exact `uv sync` there; owning the venv makes `uv sync` in this repo the single setup step.
- Copying dimos into this repo would pull a 20k-file platform in; a submodule adds ergonomics cost for no pinning benefit over `pins.toml`.
- An editable path means dimos fixes flow immediately and the no-modification rule is enforced by construction: everything is wrapped or subclassed.

## Consequences
- `pyproject.toml` replicates dimos's `[tool.uv]` override dependencies and the PyTorch cu128 index, because uv does not inherit those from a path dependency; when dimos changes them, copy the change.
- dimos has no release tags, so the pin is a commit; drift fails `scripts/check_pins.sh`.
