# Decision 011: Operating Point From Quiet Nights

## Context
- The first report set the false-alarm rate from benign objects seen inside intrusion episodes: about a minute of exposure each, with an intruder always present, which inflated decisions per hour to tens per hour.
- Lane B's part 3 added quiet nights (fleet and benign traffic, no intruder, one charge cycle each) and a coverage profile over the battery duty cycles.

## Decision
The report's false-alarm rate and operating point come from quiet nights: for each configuration, 20 quiet seeds taken from the end of the committed seed list, disjoint from the intrusion seeds. Detection stays from the intrusion episodes at the operating threshold. Human decisions per hour is the quiet-night false-alarm rate at the engine's deployed alarm threshold of 4.0, not at the operating point. Coverage gap seconds per hour is the engine's coverage profile.

## Reason
- One false alarm per hour is only measurable with hours of benign exposure; intrusion windows give minutes.
- At the operating point the false-alarm rate is one per hour by construction, so decisions per hour there would be the same for every fleet; at the deployed threshold it differs, which is the attention cost a buyer would actually pay.
- The coverage profile is deterministic given the fleet's duty cycles, so the gap column no longer depends on which tactics were run.

## Consequences
- `airtight-sweep --quiet-seeds 20` is the default; `--quiet-seeds 0` or the stub engine fall back to the old estimate and the report's conditions say so.
- Paired deltas now include the coverage gap.
- The bootstrap interval on detection holds the threshold fixed; quiet nights are not resampled.
