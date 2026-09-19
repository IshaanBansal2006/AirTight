from __future__ import annotations

from pathlib import Path  # noqa: TC003
from types import SimpleNamespace

from dimos.msgs.geometry_msgs.Twist import Twist  # noqa: TC002

from airtight.dimos_lane.modules.walk import WalkModule, _xy_of, displacement_m
from airtight.dimos_lane.simulator import A1_LLM_TURN, A1_SKILL_CALL


class _FakeOut:
    def __init__(self) -> None:
        self.msgs: list[Twist] = []

    def publish(self, msg: Twist) -> None:
        self.msgs.append(msg)


class _FakeOdom:
    def __init__(self, poses: list[tuple[float, float]]) -> None:
        self._poses = list(poses)

    def get_next(self, timeout: float = 1.0) -> SimpleNamespace:
        x, y = self._poses.pop(0)
        return SimpleNamespace(position=SimpleNamespace(x=x, y=y))


class _FakeImage:
    width = 8
    height = 6

    def to_jpeg_bytes(self, quality: int = 75) -> bytes:
        return b"\xff\xd8fakejpeg" + bytes([quality])


class _FakeCamera:
    def __init__(self, frame: _FakeImage) -> None:
        self.frame = frame
        self.calls = 0

    def get_next(self, timeout: float = 5.0) -> _FakeImage:
        self.calls += 1
        return self.frame


def test_walk_skills_are_marked() -> None:
    assert getattr(WalkModule.walk_forward, "__skill__", False)
    assert getattr(WalkModule.snapshot_camera, "__skill__", False)
    assert getattr(WalkModule.walk_to, "__skill__", False)
    assert getattr(WalkModule.dispatch_verify, "__skill__", False)
    assert getattr(WalkModule.fleet_status, "__skill__", False)
    assert getattr(WalkModule.check_north_gate, "__skill__", False)
    assert getattr(WalkModule.prompt_agent, "__skill__", False)
    assert A1_SKILL_CALL is True
    assert A1_LLM_TURN is False


def test_displacement_and_xy_helpers() -> None:
    pose = SimpleNamespace(position=SimpleNamespace(x=10.0, y=70.0))
    assert _xy_of(pose) == (10.0, 70.0)
    assert displacement_m((10.0, 70.0), (13.0, 74.0)) == 5.0


def test_walk_forward_publishes_twist_then_stop() -> None:
    walker = SimpleNamespace()
    walker.cmd_vel = _FakeOut()
    walker.odom = _FakeOdom([(10.0, 70.0), (10.4, 70.0)])
    walker._unconnected = lambda port: False
    walker._read_odom = lambda timeout=1.0: WalkModule._read_odom(walker, timeout)
    walker._try_xy = lambda timeout=1.0: WalkModule._try_xy(walker, timeout)
    result = WalkModule.walk_forward(walker, seconds=0.0, vx=0.3)
    assert "0.0s" in result
    assert "0.30 m/s" in result
    assert "displacement=0.400m" in result
    assert len(walker.cmd_vel.msgs) == 2
    assert walker.cmd_vel.msgs[0].linear.x == 0.3
    assert walker.cmd_vel.msgs[1].is_zero()


def test_snapshot_camera_writes_jpeg(tmp_path: Path) -> None:
    walker = SimpleNamespace()
    walker.color_image = _FakeCamera(_FakeImage())
    dest = tmp_path / "go2_intruder.jpg"
    result = WalkModule.snapshot_camera(walker, path=str(dest))
    assert dest.is_file()
    assert dest.read_bytes().startswith(b"\xff\xd8")
    assert "8x6" in result


class _FakeOdomYaw(_FakeOdom):
    def get_next(self, timeout: float = 1.0) -> SimpleNamespace:
        x, y = self._poses.pop(0)
        return SimpleNamespace(position=SimpleNamespace(x=x, y=y), yaw=0.0)


def test_prompt_agent_publishes() -> None:
    walker = SimpleNamespace()
    walker.human_input = _FakeOut()
    result = WalkModule.prompt_agent(walker, "check the north gate")
    assert "queued" in result
    assert walker.human_input.msgs == ["check the north gate"]


def test_walk_to_publishes_twist_then_stop() -> None:
    walker = SimpleNamespace()
    walker.cmd_vel = _FakeOut()
    walker.odom = _FakeOdomYaw([(10.0, 70.0), (10.4, 70.1)])
    walker.brain = None
    walker._unconnected = lambda port: False
    walker._read_odom = lambda timeout=1.0: WalkModule._read_odom(walker, timeout)
    walker._try_xy = lambda timeout=1.0: WalkModule._try_xy(walker, timeout)
    walker._turn_toward = lambda x, y, pose: WalkModule._turn_toward(walker, x, y, pose)
    walker._sync_brain_pose = lambda xy: WalkModule._sync_brain_pose(walker, xy)
    result = WalkModule.walk_to(walker, 60.0, 75.0, seconds=0.0, vx=0.35)
    assert "walk_to (60.0,75.0)" in result
    assert "displacement=0.412m" in result or "displacement=0.4" in result
    assert walker.cmd_vel.msgs[0].linear.x == 0.35
    assert walker.cmd_vel.msgs[-1].is_zero()
