"""Turns what the episode loop reports into the contract's log events.

Event kinds written: position (2 Hz), detection (every hit), score (every look),
alarm_delivered (the intruder's first crossing of TAU_REF) and outcome. Kinds left empty in v0
because the sim does not model them yet: task, battery, comms_graph. Every event has t >= 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from airtight.contracts import (
    XY,
    AlarmEvent,
    DetectionEvent,
    OutcomeEvent,
    PositionEvent,
    ScoreEvent,
)
from airtight.sim.constants import TAU_REF, TIME_EPS
from airtight.sim.sensing import LookSchedule

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    import numpy.typing as npt
    from pydantic import BaseModel

    from airtight.sim.episode import EpisodeScores
    from airtight.sim.sensing import Look

    Array = npt.NDArray[np.float64]

POSE_RATE_HZ = 2.0
INTRUDER_ID = "intruder"
_POSES = "poses"


def timely_at_ref(scores: EpisodeScores) -> bool:
    """The contract verdict at the reference threshold: alarmed at or before t_cdp."""
    t_alarm = scores.intruder_t_alarm_ref
    return t_alarm is not None and t_alarm <= scores.t_cdp + TIME_EPS


def human_decisions_at_ref(scores: EpisodeScores) -> int:
    """Alarms a human would have had to judge at TAU_REF: intruder, decoy and benign alike."""
    count = int(scores.intruder_t_alarm_ref is not None)
    count += int(scores.decoy_peak is not None and scores.decoy_peak >= TAU_REF)
    return count + sum(peak >= TAU_REF for peak in scores.benign_peaks.values())


def outcome_event(scores: EpisodeScores) -> OutcomeEvent:
    return OutcomeEvent(
        t=scores.t_end,
        timely_detected=timely_at_ref(scores),
        t_alarm=scores.intruder_t_alarm_ref,
        t_cdp=scores.t_cdp,
        human_decisions=human_decisions_at_ref(scores),
    )


class LogRecorder:
    """Collects the full log in time order. events ends with the outcome after on_finish."""

    def __init__(self, dt: float) -> None:
        self.events: list[BaseModel] = []
        self._pose_schedule = LookSchedule({_POSES: POSE_RATE_HZ}, dt)
        self._alarmed = False

    def on_poses(self, t: float, poses: Mapping[str, Array]) -> None:
        if not self._pose_schedule.due(_POSES, t):
            return
        for object_id in sorted(poses):
            x, y = poses[object_id]
            self.events.append(
                PositionEvent(t=t, object_id=object_id, position=XY(x=float(x), y=float(y)))
            )

    def on_looks(self, t: float, looks: Sequence[Look]) -> None:
        for look in looks:
            if look.hit:
                self.events.append(
                    DetectionEvent(
                        t=t,
                        agent_id=look.agent_id,
                        object_id=look.object_id,
                        true_positive=look.object_id == INTRUDER_ID,
                    )
                )
            self.events.append(ScoreEvent(t=t, object_id=look.object_id, value=look.score))
            if look.object_id == INTRUDER_ID and look.score >= TAU_REF and not self._alarmed:
                self._alarmed = True
                self.events.append(AlarmEvent(t=t, object_id=INTRUDER_ID, via=look.agent_id))

    def on_finish(self, scores: EpisodeScores) -> None:
        self.events.append(outcome_event(scores))
