"""
Task 2(iv): Reusable motion skills.

Provides:
- move(vx, vy, wz, duration): queued timed motion for Task 3
- set_velocity(vx, vy, wz): continuous command for Task 4
- stop(): persistent zero-velocity command
- get_base_pose(): thread-safe (x, y, yaw)
- turn(angle_deg): closed-loop relative yaw turn

The simulation loop calls update() every control cycle.

Student A contribution.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import mujoco
import numpy as np


@dataclass
class TimedCommand:
    """One queued timed velocity command."""

    vx: float
    vy: float
    wz: float
    duration: float


class MotionSkills:
    """Thread-safe motion interface shared by Tasks 3 and 4."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # --------------------------------------------------------------
        # Task 3 timed-motion queue
        # --------------------------------------------------------------

        self._queue: Deque[TimedCommand] = deque()
        self._active: Optional[TimedCommand] = None
        self._active_start_time: Optional[float] = None

        # --------------------------------------------------------------
        # Task 4 / closed-loop continuous command
        # --------------------------------------------------------------

        self._continuous_cmd = np.zeros(
            3,
            dtype=np.float32,
        )

        self._continuous_active = False

        # --------------------------------------------------------------
        # State copied from MuJoCo by the simulation thread
        # --------------------------------------------------------------

        self._base_pose: Optional[
            Tuple[float, float, float]
        ] = None

        self._sim_time: Optional[float] = None

    # ==================================================================
    # Utility functions
    # ==================================================================

    @staticmethod
    def _clip_velocity(value: float) -> float:
        """Clamp normalized velocity to [-1, 1]."""

        return float(
            np.clip(value, -1.0, 1.0)
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """
        Wrap an angle to [-pi, pi].

        This is used for measuring the small change in yaw between two
        consecutive simulation states.
        """

        return math.atan2(
            math.sin(angle),
            math.cos(angle),
        )

    # ==================================================================
    # Task 3: timed move queue
    # ==================================================================

    def move(
        self,
        vx: float,
        vy: float,
        wz: float,
        duration: float,
    ) -> None:
        """
        Append a timed velocity command to the queue.

        Args:
            vx:
                Forward velocity command in [-1, 1].

            vy:
                Left/right velocity command in [-1, 1].

            wz:
                Yaw velocity command in [-1, 1].
                Positive = left / counter-clockwise.

            duration:
                Command duration in simulation seconds.
        """

        if duration <= 0:
            raise ValueError(
                "duration must be > 0"
            )

        command = TimedCommand(
            vx=self._clip_velocity(vx),
            vy=self._clip_velocity(vy),
            wz=self._clip_velocity(wz),
            duration=float(duration),
        )

        with self._lock:
            # Timed commands take ownership away from continuous
            # Task 4 servoing.
            self._continuous_active = False
            self._continuous_cmd[:] = 0.0

            self._queue.append(command)

    # ==================================================================
    # Task 4: continuous velocity API
    # ==================================================================

    def set_velocity(
        self,
        vx: float,
        vy: float,
        wz: float,
    ) -> None:
        """
        Set a continuous velocity command.

        This bypasses the timed-move queue, as required by RobotAPI
        for Task 4 visual servoing.
        """

        command = np.array(
            [
                self._clip_velocity(vx),
                self._clip_velocity(vy),
                self._clip_velocity(wz),
            ],
            dtype=np.float32,
        )

        with self._lock:
            # Continuous control replaces queued motion.
            self._queue.clear()
            self._active = None
            self._active_start_time = None

            self._continuous_cmd = command
            self._continuous_active = True

    def stop(self) -> None:
        """
        Immediately stop and retain autonomous ownership.

        This is intended for Task 4. After stop(), the robot remains
        commanded at [0, 0, 0] until another autonomous command is given.
        """

        with self._lock:
            self._queue.clear()
            self._active = None
            self._active_start_time = None

            self._continuous_cmd[:] = 0.0
            self._continuous_active = True

    def release_control(self) -> None:
        """
        Stop and release autonomous command ownership.

        Used when a Task 3 skill such as turn() has completed.
        """

        with self._lock:
            self._queue.clear()
            self._active = None
            self._active_start_time = None

            self._continuous_cmd[:] = 0.0
            self._continuous_active = False

    # ==================================================================
    # Simulation-thread update
    # ==================================================================

    def update(
        self,
        data: mujoco.MjData,
    ) -> Tuple[np.ndarray, bool]:
        """
        Update cached robot state and timed-motion state.

        This must be called by the main simulation loop.

        Returns:
            (cmd, owns_control)

            cmd:
                Current normalized [vx, vy, wz].

            owns_control:
                True when MotionSkills should override manual
                keyboard/browser commands.

                This can be True even when cmd == [0, 0, 0].
        """

        sim_time = float(data.time)

        # --------------------------------------------------------------
        # Copy robot base pose from MuJoCo
        # --------------------------------------------------------------

        x = float(data.qpos[0])
        y = float(data.qpos[1])

        # MuJoCo free-joint quaternion is:
        # (w, x, y, z)
        w, qx, qy, qz = [
            float(value)
            for value in data.qpos[3:7]
        ]

        yaw = math.atan2(
            2.0 * (w * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        with self._lock:
            self._base_pose = (
                x,
                y,
                yaw,
            )

            self._sim_time = sim_time

            # ----------------------------------------------------------
            # Continuous command has priority
            # ----------------------------------------------------------

            if self._continuous_active:
                return (
                    self._continuous_cmd.copy(),
                    True,
                )

            # ----------------------------------------------------------
            # Start next queued timed command
            # ----------------------------------------------------------

            if (
                self._active is None
                and self._queue
            ):
                self._active = (
                    self._queue.popleft()
                )

                self._active_start_time = (
                    sim_time
                )

            # Nothing autonomous is active.
            if self._active is None:
                return (
                    np.zeros(
                        3,
                        dtype=np.float32,
                    ),
                    False,
                )

            elapsed = (
                sim_time
                - float(self._active_start_time)
            )

            # ----------------------------------------------------------
            # Current timed command finished
            # ----------------------------------------------------------

            if elapsed >= self._active.duration:
                self._active = None
                self._active_start_time = None

                # Start next queued command immediately.
                if self._queue:
                    self._active = (
                        self._queue.popleft()
                    )

                    self._active_start_time = (
                        sim_time
                    )

                else:
                    return (
                        np.zeros(
                            3,
                            dtype=np.float32,
                        ),
                        False,
                    )

            command = np.array(
                [
                    self._active.vx,
                    self._active.vy,
                    self._active.wz,
                ],
                dtype=np.float32,
            )

            return command, True

    # ==================================================================
    # State API
    # ==================================================================

    def get_base_pose(
        self,
    ) -> Tuple[float, float, float]:
        """
        Return latest cached trunk pose.

        Returns:
            (x, y, yaw)

        x and y are world-frame metres.
        yaw is radians.
        """

        with self._lock:
            if self._base_pose is None:
                raise RuntimeError(
                    "Simulation state is not available yet."
                )

            return tuple(self._base_pose)

    def get_sim_time(self) -> float:
        """Return latest MuJoCo simulation time."""

        with self._lock:
            if self._sim_time is None:
                raise RuntimeError(
                    "Simulation time is not available yet."
                )

            return float(self._sim_time)

    # ==================================================================
    # Task 2 closed-loop turn
    # ==================================================================

    def turn(
        self,
        angle_deg: float,
        tolerance_deg: float = 3.0,
        max_wz: float = 0.65,
        min_wz: float = 0.30,
        kp: float = 1.5,
        timeout: float = 25.0,
    ) -> bool:
        """
        Perform a closed-loop relative yaw turn.

        Positive angle:
            left / counter-clockwise.

        Negative angle:
            right / clockwise.

        Unlike a target-heading-only controller, this implementation
        accumulates the robot's measured yaw change. This preserves the
        requested direction at exactly +/-180 degrees and also supports
        rotations greater than 180 degrees.

        This function must run in a worker thread while the main
        simulation thread continues stepping MuJoCo.

        Args:
            angle_deg:
                Requested relative rotation in degrees.

            tolerance_deg:
                Allowed remaining angular error.

            max_wz:
                Maximum normalized yaw command magnitude.

            min_wz:
                Minimum command magnitude while outside tolerance.
                This prevents the learned walking policy from stalling
                near the target.

            kp:
                Proportional controller gain.

            timeout:
                Maximum MuJoCo simulation time for the turn.

        Returns:
            True if the requested rotation is reached.
            False if the turn times out.
        """

        requested_deg = float(angle_deg)

        # A zero-degree turn is already complete.
        if abs(requested_deg) <= tolerance_deg:
            print(
                f"[TURN] target={requested_deg:.1f} deg "
                f"final_error={requested_deg:.1f} deg"
            )

            return True

        # --------------------------------------------------------------
        # Initial measured state
        # --------------------------------------------------------------

        _, _, previous_yaw = (
            self.get_base_pose()
        )

        start_time = self.get_sim_time()

        requested_rad = math.radians(
            requested_deg
        )

        tolerance_rad = math.radians(
            float(tolerance_deg)
        )

        # Accumulated physical rotation.
        accumulated_yaw = 0.0

        last_debug_time = start_time

        # --------------------------------------------------------------
        # Closed-loop controller
        # --------------------------------------------------------------

        while True:
            _, _, current_yaw = (
                self.get_base_pose()
            )

            current_time = (
                self.get_sim_time()
            )

            # ----------------------------------------------------------
            # Measure incremental yaw motion.
            #
            # Example crossing +180 -> -179:
            #
            # raw difference ~= -359 deg
            # wrapped difference = +1 deg
            #
            # Therefore accumulated_yaw remains continuous.
            # ----------------------------------------------------------

            yaw_step = self._wrap_angle(
                current_yaw - previous_yaw
            )

            accumulated_yaw += yaw_step
            previous_yaw = current_yaw

            # Remaining requested rotation.
            error = (
                requested_rad
                - accumulated_yaw
            )

            error_deg = math.degrees(
                error
            )

            accumulated_deg = math.degrees(
                accumulated_yaw
            )

            # ----------------------------------------------------------
            # Success
            # ----------------------------------------------------------

            if abs(error) <= tolerance_rad:
                self.release_control()

                print(
                    f"[TURN] "
                    f"target={requested_deg:.1f} deg "
                    f"actual={accumulated_deg:.1f} deg "
                    f"final_error={error_deg:.1f} deg"
                )

                return True

            # ----------------------------------------------------------
            # Timeout
            # ----------------------------------------------------------

            if (
                current_time - start_time
                >= timeout
            ):
                self.release_control()

                print(
                    f"[TURN] "
                    f"target={requested_deg:.1f} deg "
                    f"actual={accumulated_deg:.1f} deg "
                    f"final_error={error_deg:.1f} deg "
                    f"status=TIMEOUT"
                )

                return False

            # ----------------------------------------------------------
            # Proportional yaw controller
            # ----------------------------------------------------------

            wz = float(
                np.clip(
                    kp * error,
                    -max_wz,
                    max_wz,
                )
            )

            # The learned locomotion policy tends to stop physically
            # rotating when the normalized yaw command becomes too
            # small. Keep a minimum magnitude until inside tolerance.
            if abs(wz) < min_wz:
                wz = math.copysign(
                    min_wz,
                    error,
                )

            self.set_velocity(
                0.0,
                0.0,
                wz,
            )

            # ----------------------------------------------------------
            # Development diagnostic
            # ----------------------------------------------------------

            if (
                current_time
                - last_debug_time
                >= 1.0
            ):
                print(
                    f"[TURN DEBUG] "
                    f"actual={accumulated_deg:+.1f} deg "
                    f"error={error_deg:+.1f} deg "
                    f"cmd_wz={wz:+.3f} "
                    f"yaw={math.degrees(current_yaw):+.1f} deg"
                )

                last_debug_time = (
                    current_time
                )

            # Yield to the main simulation thread.
            time.sleep(0.02)