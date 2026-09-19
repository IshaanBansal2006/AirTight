"""Schemas shared by every lane. Changing anything here needs all three owners."""

from airtight.contracts.episode import (
    AlarmEvent,
    BatteryEvent,
    CommsGraphEvent,
    DetectionEvent,
    EpisodeHeader,
    EpisodeResult,
    Event,
    OutcomeEvent,
    PositionEvent,
    ScoreEvent,
    TaskEvent,
    read_episode_log,
    write_episode_log,
)
from airtight.contracts.fleet import AgentSpec, AgentType, ChargePolicy, CommsMode, FleetConfig
from airtight.contracts.report import Conditions, ConfigResult, PairedDelta, Report, RocPoint
from airtight.contracts.sensor_curve import SensorCurve, SensorCurves
from airtight.contracts.site import XY, BenignRoute, Dock, EntryPoint, FixedSensor, Site
from airtight.contracts.tactic import CommsEvent, Decoy, Tactic, TacticFamily, TacticOrigin

__all__ = [
    "XY",
    "Site",
    "EntryPoint",
    "Dock",
    "FixedSensor",
    "BenignRoute",
    "AgentType",
    "AgentSpec",
    "ChargePolicy",
    "CommsMode",
    "FleetConfig",
    "TacticFamily",
    "TacticOrigin",
    "Decoy",
    "CommsEvent",
    "Tactic",
    "SensorCurve",
    "SensorCurves",
    "EpisodeHeader",
    "Event",
    "PositionEvent",
    "DetectionEvent",
    "ScoreEvent",
    "TaskEvent",
    "BatteryEvent",
    "CommsGraphEvent",
    "AlarmEvent",
    "OutcomeEvent",
    "EpisodeResult",
    "write_episode_log",
    "read_episode_log",
    "RocPoint",
    "PairedDelta",
    "Conditions",
    "ConfigResult",
    "Report",
]
