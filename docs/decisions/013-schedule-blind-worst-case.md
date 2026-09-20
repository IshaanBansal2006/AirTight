# Decision 013: Schedule-Blind Worst Case

## Context
- The re-attack worst case sits near zero at every budget (results/minmax). The adversary in that number knows the charge schedule to the second: every tactic carries a phase of the charge cycle, and search tunes it.
- A buyer can protect the schedule. A number that assumes it is public overstates the exposure for a site that keeps it private, and understates what hiding it is worth.
- Adding a second adversary model to the search would double its cost and add a knob nobody has time to tune.

## Decision
The report carries one more number per configuration, `worst_tactic_pd_schedule_blind`: the same worst tactics, each re-run with its entry phase drawn at random per seed. The tactics are unchanged otherwise. The sweep runs this pass by default (`--no-schedule-blind` skips it); the deck, write-up and console show it beside the full-knowledge worst case. The field is optional, so old reports still validate.

## Reason
- One extra evaluation pass on tactics the search already found, no second search: the cost is one more sweep, and the semantics stay simple enough to state in a sentence on a slide.
- Randomising the phase per seed is the smallest change that models "knows the site, not the schedule": entry, route, speed and decoys stay as the adversary chose them.
- The phase comes from the seed, not from a new random stream, so the pass is reproducible on the shared seed list.

## Consequences
- `airtight-sweep` takes about twice as long on the intrusion episodes; quiet nights are unchanged.
- The minmax loop still optimises the full-knowledge worst case; each round records the blind number beside it, reported, not optimised.
- `results/v4/` is the first archive that carries the field.
