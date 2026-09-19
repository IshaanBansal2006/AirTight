"""Lane B: headless episode runner built on airtight.swarm.

Other lanes call exactly one thing: airtight.sim.runner.run_episode.

What airtight.score may import from here. Anything not on this list is private to sim/ and may
change without notice:

- airtight.sim.episode: simulate, simulate_quiet, EpisodeScores, QuietScores, EpisodeParams,
  official_params
- airtight.sim.coverage: coverage_profile, uncovered_intervals, uncovered_s_per_hour
- airtight.sim.constants: every constant
- airtight.sim.scenarios: the scenario loader

official_params() is the one source of truth for official numbers. run_episode uses it and the
sweep must too, so a sweep can never disagree with run_episode.
"""
