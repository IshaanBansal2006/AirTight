"""Offline scoring.

Lane B: roc, quiet, config_score, quick, sweep, report, replays, fix. These score the engine's
threshold-free EpisodeScores directly and write data/report.json.
Lane C: the logreport subpackage, a scorer and sweep runner that work from episode logs
(airtight-sweep).
"""
