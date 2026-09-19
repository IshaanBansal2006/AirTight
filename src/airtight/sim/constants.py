"""Track-score constants shared by the sim and the offline scorer.

Two false-alarm numbers exist and are never merged:

- The contract's ``SensorCurve.pfa_per_look_by_class`` is the TRUE rate at which the world
  produces false detections on benign objects. It is used only to draw hits.
- ``ASSUMED_PFA`` is what the track score assumes inside its likelihood ratio. The system does
  not know what an object really is, so the same assumed value applies to every object.
"""

from __future__ import annotations

TAU_REF = 4.0  # reference alarm threshold on the track score until the offline sweep re-fixes it
NEVER_SEEN = -1e9  # peak score of an object no sensor ever looked at
SCORE_FLOOR = -5.0  # a track score is clamped below at this value
ASSUMED_PFA = 0.02  # per-look false-alarm probability assumed by the likelihood ratio
DECOY_DURATION_S = 60.0  # the contract's Decoy has a lead time but no duration, so the sim owns it
DEFAULT_CELL_SIZE_M = 5.0  # patrol grid cell edge; Site has no cell size, so the sim owns it
