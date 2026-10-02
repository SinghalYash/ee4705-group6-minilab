"""goto_object(): the Task 4 entry point the Task 3 executor calls (Student C).

    result = goto_object(robot, "chair", "green", cancel_event=ev)
    # -> MissionResult(status="SUCCESS" | "FAIL", reason, t, d, ...)

`robot` is the Task 4 adapter around Student A's platform (robot_adapter.PlatformRobot)
and must provide (see interfaces.py):
    get_latest_frame() -> (rgb uint8 HxWx3, sim_time) | None
    set_velocity(vx, vy, wz)      normalised [-1, 1], held until changed
    stop()
    get_base_pose() -> (x, y, yaw)

The call blocks until the mission ends, so the executor simply moves on to
the next queued action afterwards. It must run in a worker thread (the Task 3
executor thread), never in the simulation thread.

Ground truth object positions (objects.yaml in the project root) are used ONLY to print
d in the [FOUND] line and for evaluation, after the controller has already
decided on its own that the object is found.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from pathlib import Path

import yaml

from approach import ApproachController, CameraModel, ControllerConfig
from perception import Detector, normalize_class
from robot_adapter import get_robot

HERE = Path(__file__).resolve().parent.parent
_DETECTOR: Detector | None = None
_DETECTOR_LOCK = threading.Lock()


def get_detector() -> Detector:
    """Load YOLO once and reuse it (loading takes ~1 s)."""
    global _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            _DETECTOR = Detector()
        return _DETECTOR


@dataclass
class MissionResult:
    status: str
    reason: str = ""
    t: float = 0.0
    d: float | None = None
    cls: str = ""
    color: str | None = None
    found_bbox: tuple | None = None
    found_color: str | None = None
    frames: int = 0
    trace: list = field(default_factory=list)


class GroundTruth:
    """Object positions for logging/evaluation only. Never passed to the controller."""

    def __init__(self, path: Path | str = HERE / "objects.yaml"):
        self.objects = yaml.safe_load(Path(path).read_text())["objects"]

    def distance(self, cls, color, x, y) -> float | None:
        cls = normalize_class(cls)
        c = [o for o in self.objects
             if normalize_class(o["class"]) == cls and (color is None or o["color"] == color)]
        if not c:
            return None
        return min(math.hypot(x - o["x"], y - o["y"]) for o in c)


def finish(result: MissionResult, gt: GroundTruth | None, pose, log):
    """Print the [FOUND]/[MISSION] lines. Called after the controller decided."""
    if result.status == "SUCCESS":
        if gt is not None:
            result.d = gt.distance(result.cls, result.color, pose[0], pose[1])
        d_txt = f" d={result.d:.2f} m" if result.d is not None else ""
        col = result.color or result.found_color or "any"
        log(f"[FOUND] class={result.cls.replace(' ', '_')} color={col} "
            f"t={result.t:.1f} s{d_txt}")
        log("[MISSION] status=SUCCESS")
    else:
        log(f"[MISSION] status=FAIL reason={result.reason}")
    return result


def camera_for(robot, cam: CameraModel | None = None) -> CameraModel:
    """Camera model to use: explicit > robot.camera_model() > robot width/height."""
    if cam is not None:
        return cam
    fn = getattr(robot, "camera_model", None)
    if fn is not None:
        return fn()
    w, h = getattr(robot, "width", None), getattr(robot, "height", None)
    if w and h:
        return CameraModel(width=int(w), height=int(h))
    return CameraModel()


def _release(robot):
    """Give control back after a mission (adapter -> A's release_control())."""
    rel = getattr(robot, "release", None)
    if rel is not None:
        rel()


def goto_object(robot, target_class: str, target_color: str | None = None, *,
                cancel_event: threading.Event | None = None,
                cfg: ControllerConfig | None = None, cam: CameraModel | None = None,
                gt: GroundTruth | None = None, log=print,
                frame_poll_s: float = 0.005) -> MissionResult:
    """Blocking, real-time search-and-approach. Run from the executor thread."""
    ctrl = ApproachController(get_detector(), target_class, target_color, cfg,
                              camera_for(robot, cam), log)
    if gt is None:
        try:
            gt = GroundTruth()
        except FileNotFoundError:
            gt = None
    last_t = None
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                robot.stop()
                res = MissionResult("FAIL", "cancelled", cls=ctrl.cls, color=ctrl.color)
                return finish(res, gt, robot.get_base_pose(), log)
            item = robot.get_latest_frame()
            if item is None or item[1] == last_t:
                time.sleep(frame_poll_s)
                continue
            rgb, last_t = item
            x, y, yaw = robot.get_base_pose()
            r = ctrl.tick(rgb, last_t, yaw, (x, y))
            robot.set_velocity(*r.cmd)
            if r.status is not None:
                robot.stop()
                res = MissionResult(
                    r.status, r.reason, t=ctrl.trace[-1]["t"], cls=ctrl.cls,
                    color=ctrl.color, frames=ctrl.frames, trace=ctrl.trace,
                    found_bbox=r.found_det.bbox if r.found_det else None,
                    found_color=r.found_det.color if r.found_det else None,
                )
                return finish(res, gt, robot.get_base_pose(), log)
    except Exception:
        robot.stop()
        raise
    finally:
        _release(robot)


def run_goto_object_action(action: dict, cancel_event: threading.Event | None = None,
                           robot=None, log=print) -> MissionResult:
    """Execute one parsed Task 3 action {"action": "goto_object", "class": ..., "color": ...}.

    This is the single call Student B's executor makes for goto_object. The
    platform robot is the one registered by the integration script
    (robot_adapter.register_robot) unless one is passed in.
    """
    if action.get("action") != "goto_object":
        raise ValueError(f"not a goto_object action: {action!r}")
    robot = robot if robot is not None else get_robot()
    if robot is None:
        raise RuntimeError("no platform robot registered (start via task_4/run_integrated.py)")
    cls = action.get("class")
    if not cls:
        log("[MISSION] status=FAIL reason=no_class_given")
        return MissionResult("FAIL", "no_class_given")
    color = action.get("color") or None
    return goto_object(robot, cls, color, cancel_event=cancel_event, log=log)


def goto_object_lockstep(sim, target_class, target_color=None, *, cfg=None, cam=None,
                         gt=None, log=print, on_frame=None) -> MissionResult:
    """Same controller, but the sim waits for each YOLO call (repeatable eval).

    `sim` steps physics itself (PlatformSim or the dev SandboxSim) and exposes
    the RobotAPI methods plus step(), fallen() and camera_model().
    """
    ctrl = ApproachController(get_detector(), target_class, target_color, cfg,
                              camera_for(sim, cam), log)
    while True:
        while not sim.step():
            pass
        if sim.fallen():
            sim.stop()
            res = MissionResult("FAIL", "robot_fell", cls=ctrl.cls, color=ctrl.color)
            return finish(res, gt, sim.get_base_pose(), log)
        rgb, t = sim.get_latest_frame()
        x, y, yaw = sim.get_base_pose()
        r = ctrl.tick(rgb, t, yaw, (x, y))
        sim.set_velocity(*r.cmd)
        if on_frame is not None:
            on_frame(rgb, ctrl, r)
        if r.status is not None:
            sim.stop()
            res = MissionResult(
                r.status, r.reason, t=ctrl.trace[-1]["t"], cls=ctrl.cls,
                color=ctrl.color, frames=ctrl.frames, trace=ctrl.trace,
                found_bbox=r.found_det.bbox if r.found_det else None,
                found_color=r.found_det.color if r.found_det else None,
            )
            # let the robot come to rest before logging d
            for _ in range(40):
                sim.step()
            _release(sim)
            return finish(res, gt, sim.get_base_pose(), log)
