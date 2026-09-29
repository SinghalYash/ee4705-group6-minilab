"""Interfaces Task 4 needs from Task 2 (Student A) and Task 3 (Student B).

This file is the contract. Task 4 codes against it; the sandbox simulator
(sim_backend.SandboxSim) implements it today, and Student A's platform should
implement the same four methods so goto_object() runs unchanged on it.
"""
from __future__ import annotations

from typing import Optional, Protocol, Tuple

import numpy as np


class RobotAPI(Protocol):
    """What goto_object() calls. All methods must be safe to call from a
    worker thread while the simulation loop runs in the main thread."""

    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Latest onboard front-camera image and the sim time it was rendered.

        Returns (rgb, t): rgb is HxWx3 uint8 in RGB order (as mujoco.Renderer
        gives it), t is simulation time in seconds. Returns None before the
        first frame. Task 4 detects a new frame by t changing, so t must
        increase with every newly rendered frame.
        """

    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Continuous velocity command, normalised to [-1, 1], held until the
        next call. Used for closed-loop servoing at the camera rate
        (10-20 Hz), so it must take effect on the next control step and must
        not go through the timed-move queue."""

    def stop(self) -> None:
        """Equivalent to set_velocity(0, 0, 0)."""

    def get_base_pose(self) -> Tuple[float, float, float]:
        """(x, y, yaw) of the trunk from mj_data.qpos. x, y in metres (world
        frame), yaw in radians from the quaternion. Task 4 uses yaw to detect a
        full search turn and x, y for odometry (distance walked) and d logging;
        it never uses object positions to steer."""


# What Task 4 expects from Task 3 (Student B) ---------------------------------
#
# JSON action produced by the parser:
#     {"action": "goto_object", "class": "<COCO class name>", "color": "<colour or null>"}
#   - class: one of the COCO names in the scene, e.g. "chair", "stop sign",
#     "sports ball". normalize_class() also accepts "ball", "stop_sign", ...
#   - color: one of perception.COLOR_NAMES or null when the user gave none.
#
# Executor call (blocking, in the executor thread, not the sim thread):
#     from goto_object import goto_object
#     result = goto_object(robot, cmd["class"], cmd.get("color"),
#                          cancel_event=executor_cancel_event)
#     # result.status is "SUCCESS" or "FAIL"; move on to the next action.
#
# A "stop" command sets executor_cancel_event; goto_object then stops the
# robot and returns status FAIL reason=cancelled within one frame.
#
# Logging split: Task 3 prints [CMD] / [EXEC] / [DONE];
# Task 4 prints [SEARCH] / [DETECT] / [FOUND] / [MISSION].
