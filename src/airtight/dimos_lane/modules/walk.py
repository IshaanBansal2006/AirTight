"""Drive the simulated Go2 via cmd_vel. No LLM, no spatial stack."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from dimos.agents.annotation import skill
from dimos.core.module import Module
from dimos.core.stream import In, Out  # noqa: TC002  # runtime stream markers for autoconnect
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped  # noqa: TC002
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image  # noqa: TC002

DEFAULT_SNAPSHOT = "data/go2_intruder.jpg"


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
    """Publishes Twist so a running GO2Connection actually walks."""

    cmd_vel: Out[Twist]
    odom: In[PoseStamped]
    color_image: In[Image]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _try_xy(self, timeout: float = 1.0) -> tuple[float, float] | None:
        odom = getattr(self, "odom", None)
        get_next = getattr(odom, "get_next", None)
        if get_next is None:
            return None
        # Unconnected In ports have neither a transport nor a remote Out.
        if (
            type(odom).__name__ == "In"
            and getattr(odom, "connection", None) is None
            and getattr(odom, "_transport", None) is None
        ):
            return None
        try:
            return _xy_of(get_next(timeout=timeout))
        except Exception:
            return None

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
    def snapshot_camera(self, path: str = DEFAULT_SNAPSHOT) -> str:
        """Write the current Go2 color frame to `path` as JPEG."""
        frame = self.color_image.get_next(timeout=5.0)
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(frame.to_jpeg_bytes(quality=85))
        return f"wrote {dest} {int(frame.width)}x{int(frame.height)}"
