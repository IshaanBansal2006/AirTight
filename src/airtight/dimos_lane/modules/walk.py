"""Drive the simulated Go2 via cmd_vel and host the site-brain skills."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out  # noqa: TC002  # runtime stream markers for autoconnect
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped  # noqa: TC002
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image  # noqa: TC002

from airtight.dimos_lane.modules.fleet_memory import format_recall, load_fleet_memory, remember_item
from airtight.dimos_lane.modules.orchestrator import Orchestrator, north_gate_xy

DEFAULT_SNAPSHOT = "data/go2_intruder.jpg"
GO2_ID = "go2_1"


def _xy_of(pose: Any) -> tuple[float, float] | None:
    position = getattr(pose, "position", pose)
    x = getattr(position, "x", None)
    y = getattr(position, "y", None)
    if x is None or y is None:
        return None
    return (float(x), float(y))


def displacement_m(start: tuple[float, float], end: tuple[float, float]) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return (dx * dx + dy * dy) ** 0.5


class WalkModule(Module):
    """Publishes Twist so a running GO2Connection actually walks.

    Also embeds the site Orchestrator and binds `DimosBackend.goto` to
    `walk_to` so A5/A6 dispatch moves the live Go2.
    """

    cmd_vel: Out[Twist]
    odom: In[PoseStamped]
    color_image: In[Image]
    human_input: Out[str]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.brain = Orchestrator()
        self.brain.backend.bind_move_to(self._drive_xy)
        self.brain.backend.bind_pose_reader(lambda: self._try_xy(timeout=0.4))
        self.memory = load_fleet_memory()

    def _unconnected(self, port: Any) -> bool:
        return (
            type(port).__name__ == "In"
            and getattr(port, "connection", None) is None
            and getattr(port, "_transport", None) is None
        )

    def _read_odom(self, timeout: float = 1.0) -> Any | None:
        odom = getattr(self, "odom", None)
        get_next = getattr(odom, "get_next", None)
        if get_next is None or self._unconnected(odom):
            return None
        try:
            return get_next(timeout=timeout)
        except Exception:
            return None

    def _try_xy(self, timeout: float = 1.0) -> tuple[float, float] | None:
        pose = self._read_odom(timeout=timeout)
        if pose is None:
            return None
        return _xy_of(pose)

    def _drive_xy(self, x: float, y: float) -> str:
        return self.walk_to(x, y)

    def _sync_brain_pose(self, xy: tuple[float, float] | None) -> None:
        if xy is None:
            return
        brain = getattr(self, "brain", None)
        if brain is None:
            return
        brain.backend.set_pose(GO2_ID, [xy[0], xy[1], 0.0])
        brain.fleet.set_pose(GO2_ID, xy[0], xy[1], 0.0)

    def _turn_toward(self, x: float, y: float, pose: Any) -> None:
        xy = _xy_of(pose)
        if xy is None:
            return
        yaw = float(getattr(pose, "yaw", 0.0) or 0.0)
        desired = math.atan2(y - xy[1], x - xy[0])
        err = (desired - yaw + math.pi) % (2.0 * math.pi) - math.pi
        if abs(err) <= 0.25:
            return
        wz = 0.7 if err > 0 else -0.7
        self.cmd_vel.publish(Twist(angular=Vector3(z=wz)))
        time.sleep(min(1.5, abs(err) / 0.7))
        self.cmd_vel.publish(Twist.zero())

    @rpc
    def start(self) -> None:
        super().start()
        self.brain.backend.bind_move_to(self._drive_xy)
        self.brain.backend.bind_pose_reader(lambda: self._try_xy(timeout=0.4))

    @skill
    def walk_forward(self, seconds: float = 1.0, vx: float = 0.25) -> str:
        """Drive forward at vx m/s for `seconds`, then stop. No planner, no LLM."""
        start = self._try_xy()
        self.cmd_vel.publish(Twist(linear=Vector3(x=float(vx), y=0.0, z=0.0)))
        time.sleep(max(0.0, float(seconds)))
        self.cmd_vel.publish(Twist.zero())
        end = self._try_xy()
        extra = ""
        if start is not None and end is not None:
            extra = (
                f" displacement={displacement_m(start, end):.3f}m "
                f"start={start[0]:.2f},{start[1]:.2f} end={end[0]:.2f},{end[1]:.2f}"
            )
        return f"walked {float(seconds):.1f}s at {float(vx):.2f} m/s{extra}"

    @skill
    def walk_to(self, x: float, y: float, seconds: float = 2.5, vx: float = 0.35) -> str:
        """Turn toward (x, y) then walk forward. Does not wait for arrival."""
        pose = self._read_odom()
        start = _xy_of(pose) if pose is not None else None
        if pose is not None:
            self._turn_toward(x, y, pose)
        self.cmd_vel.publish(Twist(linear=Vector3(x=float(vx), y=0.0, z=0.0)))
        time.sleep(max(0.0, float(seconds)))
        self.cmd_vel.publish(Twist.zero())
        end = self._try_xy()
        self._sync_brain_pose(end)
        extra = ""
        if start is not None and end is not None:
            extra = f" displacement={displacement_m(start, end):.3f}m"
        return f"walk_to ({float(x):.1f},{float(y):.1f}) {float(seconds):.1f}s{extra}"

    @skill
    def prompt_agent(self, message: str) -> str:
        """Publish `message` on the site agent bus (same topic as MCP agent_send)."""
        bus = getattr(self, "human_input", None)
        publish = getattr(bus, "publish", None)
        if publish is None:
            return "no agent bus"
        publish(message)
        return f"queued for agent: {message[:80]}"

    @skill
    def snapshot_camera(self, path: str = DEFAULT_SNAPSHOT) -> str:
        """Write the current Go2 color frame to `path` as JPEG."""
        frame = self.color_image.get_next(timeout=8.0)
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(frame.to_jpeg_bytes(quality=85))
        return f"wrote {dest} {int(frame.width)}x{int(frame.height)}"

    def _note_memory(self, claim: str | None = None) -> None:
        memory = getattr(self, "memory", None)
        brain = getattr(self, "brain", None)
        if memory is None or brain is None:
            return
        for did, pos in brain.fleet.snapshot().items():
            remember_item(memory, kind="coverage", x=float(pos[0]), y=float(pos[1]), key=did)
        if claim:
            remember_item(memory, kind="claim", key="last_dispatch", value=claim)

    @skill
    def dispatch_verify(self, x: float, y: float) -> str:
        """Create a verify task at (x, y), auction it, send the winner."""
        result = self.brain.dispatch_verify(x, y)
        self._note_memory(self.brain.last_dispatch)
        return result

    @skill
    def fleet_status(self) -> str:
        """Summarize agent poses, last dispatch and pending approvals."""
        self._note_memory()
        return self.brain.fleet_status()

    @skill
    def recall(self, query: str = "*", kind: str = "coverage") -> str:
        """Return fleet-memory entries for `kind`, each with age in seconds."""
        self._note_memory()
        return format_recall(self.memory, query, kind)

    @skill
    def site_status(self) -> str:
        """Describe the loaded site: name, entries and docks."""
        from airtight.dimos_lane.site_io import load_example_site

        site = load_example_site()
        entries = ", ".join(e.id for e in site.entry_points)
        return (
            f"site {site.name}: {len(site.entry_points)} entry points ({entries}), "
            f"{len(site.docks)} docks, response time {site.response_time_s:.0f}s"
        )

    @skill
    def check_north_gate(self) -> str:
        """Convenience for the demo prompt 'check the north gate'."""
        gx, gy = north_gate_xy()
        result = self.brain.dispatch_verify(gx, gy)
        self._note_memory(self.brain.last_dispatch)
        return result
