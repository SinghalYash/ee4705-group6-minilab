"""goto_object(): the Task 4 entry point the Task 3 executor calls (Student C).

    result = goto_object(robot, "chair", "green", cancel_event=ev)
    # -> MissionResult(status="SUCCESS" | "FAIL", reason, t, d, ...)

`robot` is Student A's platform object and must provide (see interfaces.py):
    get_latest_frame() -> (rgb uint8 HxWx3, sim_time) | None
    set_velocity(vx, vy, wz)      normalised [-1, 1], held until changed
    stop()
    get_base_pose() -> (x, y, yaw)

The call blocks until the mission ends, so the executor simply moves on to
the next queued action afterwards. It must run in a worker thread (the Task 3
executor thread), never in the simulation thread.

Ground truth object positions (config/objects.yaml) are used ONLY to print
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


def goto_object(robot, target_class: str, target_color: str | None = None, *,
                cancel_event: threading.Event | None = None,
                cfg: ControllerConfig | None = None, cam: CameraModel | None = None,
                gt: GroundTruth | None = None, log=print,
                frame_poll_s: float = 0.005) -> MissionResult:
    """Blocking, real-time search-and-approach. Run from the executor thread."""
    ctrl = ApproachController(get_detector(), target_class, target_color, cfg, cam, log)
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


def goto_object_lockstep(sim, target_class, target_color=None, *, cfg=None, cam=None,
                         gt=None, log=print, on_frame=None) -> MissionResult:
    """Same controller, but the sandbox sim waits for each YOLO call (fast eval)."""
    ctrl = ApproachController(get_detector(), target_class, target_color, cfg, cam, log)
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
            return finish(res, gt, sim.get_base_pose(), log)
