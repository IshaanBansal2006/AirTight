# Decision 006: LLM Proposer And Budget

## Context
- The plan wants an LLM planner as the first adversary and a token-cost comparison against a search-only adversary.
- The user has few API credits and asked for very sparing use.
- The listed cheapest capable models are the nano tier; reasoning-tier models spend hidden tokens that make cost unpredictable.

## Decision
The LLM only proposes: one JSON-schema call returns up to 6 programs in a primitive vocabulary (enter, wait, move, sprint, drop_decoy, cut_comms), compiled to tactics, validated, and injected into the search population. Default model `gpt-4.1-nano`, temperature 0.8, 1,200 output tokens, a hard `budget_usd` of 0.50 checked before every network call with a worst-case estimate, a content-addressed cache, an append-only ledger, and a mock fixture that tests and demos use.

## Reason
- Propose-then-search costs one or two calls per configuration instead of one per episode; the measured ratio at 200 episodes per cell is 100x, and that ratio is the slide.
- Strict JSON schema removes the parser as a failure mode; the validator, not the prompt, decides what is playable.
- Cache-first plus the cap means a re-run of any stage is free and an accidental loop cannot spend more than the cap; the mock keeps the API out of CI.

## Consequences
- One live call was made to verify the path: 865 input and 154 output tokens, $0.00015, one valid charging-window tactic. The nano model returned one proposal when asked for four; `--model gpt-4.1-mini` is the escalation if diversity matters at hour 10, at roughly four times the cost per call.
- The price table in `LlmConfig` is a best-effort default and must be checked against the provider's pricing page before a dollar figure goes on a slide.
- Waits before `enter` become phase shifts using the fleet's mean charge cycle; waits after entry are rejected, because the open-loop tactic has no pause primitive (decision 004).
