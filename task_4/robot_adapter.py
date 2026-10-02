"""Task 4 robot adapter (Student C).

Wraps Student A's Task 2 platform objects into the one robot object that
goto_object() drives (see interfaces.RobotAPI):

    camera_pipeline : task_2.camera_pipeline.FrontCameraPipeline
    motion_skills   : task_2.motion_skills.MotionSkills

Only A's public methods are called. Where Task 4 wants something A's API may
not offer yet, the method is looked up with getattr() at call time, so when A
adds it the adapter uses A's version without any edit here. Each fallback is
marked "FALLBACK".

The adapter is also registered globally (register_robot / get_robot) so that
Student B's executor can reach the platform without being passed it.
"""
from __future__ import annotations

import threading
import time

from approach import CameraModel

_ROBOT = None
_ROBOT_LOCK = threading.Lock()


def register_robot(robot) -> None:
    """Make `robot` the platform used by run_goto_object_action()."""
    global _ROBOT
    with _ROBOT_LOCK:
        _ROBOT = robot


def get_robot():
    """Return the registered platform robot, or None before the platform is up."""
    with _ROBOT_LOCK:
        return _ROBOT


class PlatformRobot:
    """RobotAPI on top of A's FrontCameraPipeline + MotionSkills."""

    def __init__(self, camera_pipeline, motion_skills, model=None):
        self.camera = camera_pipeline
        self.skills = motion_skills
        self.model = model if model is not None else getattr(camera_pipeline, "model", None)

    # ------------------------------------------------------------ camera
    @property
    def width(self) -> int:
        return int(self.camera.width)

    @property
    def height(self) -> int:
        return int(self.camera.height)

    def get_latest_frame(self):
        """(rgb HxWx3 uint8, sim_time) or None before the first frame (A's method)."""
        return self.camera.get_latest_frame()

    def camera_model(self) -> CameraModel:
        """Camera geometry for ranging/steering. Image size comes from A's pipeline."""
        name = getattr(self.camera, "camera_name", "dog_front_camera")
        if self.model is not None:
            return CameraModel.from_mujoco(self.model, name, self.width, self.height)
        return CameraModel(width=self.width, height=self.height)

    # ------------------------------------------------------------ motion
    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        self.skills.set_velocity(vx, vy, wz)

    def stop(self) -> None:
        self.skills.stop()

    def turn(self, angle_deg: float) -> bool:
        """A's closed-loop turn (blocking; call from a worker thread)."""
        return self.skills.turn(angle_deg)

    def get_base_pose(self):
        return self.skills.get_base_pose()

    def release(self) -> None:
        """Hand control back after a mission so keyboard/queued moves work again.

        A's stop() keeps autonomous ownership at zero velocity by design;
        release_control() is A's method that gives it back.
        """
        release = getattr(self.skills, "release_control", None)
        if release is not None:
            release()

    def is_idle(self):
        """True when no timed move is running. Uses A's is_idle() if it exists.

        FALLBACK: A's MotionSkills has no public idle query yet, so this returns
        None ("unknown") instead of guessing from private fields. Task 4 does
        not depend on it.
        """
        fn = getattr(self.skills, "is_idle", None)
        return fn() if fn is not None else None

    # ------------------------------------------------------------ startup
    def wait_until_ready(self, timeout_s: float = 30.0) -> None:
        """Block until A's loop has produced a pose and a camera frame."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                self.skills.get_base_pose()
                if self.camera.get_latest_frame() is not None:
                    return
            except RuntimeError:
                pass  # A raises until the first update() from the sim loop
            time.sleep(0.05)
        raise TimeoutError("platform produced no pose/frame; is play.py's loop running?")
