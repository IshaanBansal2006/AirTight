# dimOS patches

dimOS is a sibling checkout pinned in `pins.toml`. We are allowed to change it, but the checkout is not
in this repo, so every change is kept here as a patch and applied on each machine:

```bash
./scripts/dimos_patch.sh save <name>     # turn the current uncommitted diff in ../dimos into patches/dimos/NN-<name>.patch
./scripts/dimos_patch.sh apply           # apply every patch in order to a clean pinned checkout
./scripts/dimos_patch.sh status          # which patches are applied, and whether ../dimos matches the pin
```

Rules: one concern per patch, a one-line reason at the top of the patch file, and the patch is committed
in the same PR as the code that needs it. If the patch set grows past a handful, fork dimOS and pin the
fork instead.
