"""
Task 2(iv): Reusable motion skills.

Provides:
    - timed move(vx, vy, wz, duration)
    - continuous set_velocity(vx, vy, wz)
    - stop()
    - robot base pose / yaw
    - closed-loop turn(angle_deg)

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
    """Thread-safe motion interface for Tasks 3 and 4."""

    def __init__(self):
        self._lock = threading.Lock()

        # Timed commands used by move().
        self._queue: Deque[TimedCommand] = deque()
        self._active: Optional[TimedCommand] = None
        self._active_start_time: Optional[float] = None

        # Continuous command used by Task 4 visual servoing.
        self._continuous_cmd = np.zeros(3, dtype=np.float32)
        self._continuous_active = False

        # Latest MuJoCo data reference, updated by the simulation thread.
        self._base_pose: Optional[Tuple[float, float, float]] = None
        self._sim_time: Optional[float] = None

    @staticmethod
    def _clip_velocity(value: float) -> float:
        """Clamp a normalized velocity command to [-1, 1]."""
        return float(np.clip(value, -1.0, 1.0))

    def move(
        self,
        vx: float,
        vy: float,
        wz: float,
        duration: float,
    ) -> None:
        """Queue a timed velocity command."""

        if duration <= 0:
            raise ValueError("duration must be > 0")

        command = TimedCommand(
            vx=self._clip_velocity(vx),
            vy=self._clip_velocity(vy),
            wz=self._clip_velocity(wz),
            duration=float(duration),
        )

        with self._lock:
            # Timed motion takes ownership of the command source.
            self._continuous_active = False
            self._queue.append(command)

    def set_velocity(
        self,
        vx: float,
        vy: float,
        wz: float,
    ) -> None:
        """
        Set a continuous velocity command.

        Used by Task 4 for closed-loop visual servoing. This deliberately
        bypasses the timed move queue.
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
            self._queue.clear()
            self._active = None
            self._active_start_time = None

            self._continuous_cmd = command
            self._continuous_active = True

    def stop(self) -> None:
        """Immediately stop and clear pending motion."""

        with self._lock:
            self._queue.clear()
            self._active = None
            self._active_start_time = None

            self._continuous_cmd[:] = 0.0
            self._continuous_active = True

    def release_control(self) -> None:
        """Stop motion and return command ownership to the normal executor/manual source."""

        with self._lock:
            self._queue.clear()
            self._active = None
            self._active_start_time = None

            self._continuous_cmd[:] = 0.0
            self._continuous_active = False
    
    def update(
        self,
        data: mujoco.MjData,
    ) -> tuple[np.ndarray, bool]:
        """
        Update motion state.

        Returns:
            (cmd, owns_control)

            cmd:
                Current normalized [vx, vy, wz].

            owns_control:
                True when MotionSkills should override manual keyboard/browser
                commands, including an intentional zero-velocity stop.
        """

        sim_time = float(data.time)

        # Read pose before acquiring the lock.
        x = float(data.qpos[0])
        y = float(data.qpos[1])

        w, qx, qy, qz = data.qpos[3:7]

        yaw = math.atan2(
            2.0 * (w * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        with self._lock:
            self._base_pose = (x, y, yaw)
            self._sim_time = sim_time

            # Continuous command: Task 4 / closed-loop turn.
            if self._continuous_active:
                return self._continuous_cmd.copy(), True

            # Start next timed command.
            if self._active is None and self._queue:
                self._active = self._queue.popleft()
                self._active_start_time = sim_time

            if self._active is None:
                return np.zeros(3, dtype=np.float32), False

            elapsed = sim_time - float(self._active_start_time)

            if elapsed >= self._active.duration:
                self._active = None
                self._active_start_time = None

                if self._queue:
                    self._active = self._queue.popleft()
                    self._active_start_time = sim_time
                else:
                    return np.zeros(3, dtype=np.float32), False

            cmd = np.array(
                [
                    self._active.vx,
                    self._active.vy,
                    self._active.wz,
                ],
                dtype=np.float32,
            )

            return cmd, True
    
    def get_base_pose(self) -> Tuple[float, float, float]:
        """Return the latest cached trunk (x, y, yaw)."""

        with self._lock:
            if self._base_pose is None:
                raise RuntimeError("Simulation state is not available yet.")

            return self._base_pose
    

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap an angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def turn(
        self,
        angle_deg: float,
        tolerance_deg: float = 3.0,
        max_wz: float = 0.6,
        min_wz: float = 0.18,
        kp: float = 1.5,
        timeout: float = 20.0,
    ) -> bool:
        """
        Closed-loop relative turn using measured MuJoCo yaw.

        Runs from a worker thread while the simulation continues in the
        main thread.

        Args:
            angle_deg:
                Relative requested rotation in degrees.
                Positive = left / counter-clockwise.

            tolerance_deg:
                Acceptable final yaw error.

            max_wz:
                Maximum normalized yaw command.

            min_wz:
                Minimum yaw command while outside the tolerance.
                Prevents the learned locomotion policy from stalling near
                the target.

            kp:
                Proportional yaw-controller gain.

            timeout:
                Maximum SIMULATION time allowed for the turn.

        Returns:
            True if the target was reached, False on timeout.
        """

        _, _, start_yaw = self.get_base_pose()

        target_delta = math.radians(float(angle_deg))
        target_yaw = self._wrap_angle(start_yaw + target_delta)

        tolerance = math.radians(tolerance_deg)

        # Use simulation time rather than wall-clock time.
        with self._lock:
            if self._sim_time is None:
                raise RuntimeError("Simulation time is not available yet.")

            start_time = self._sim_time

        last_debug_time = -1.0
        while True:
            _, _, current_yaw = self.get_base_pose()

            with self._lock:
                current_time = self._sim_time

            error = self._wrap_angle(target_yaw - current_yaw)

            # Target reached.
            if abs(error) <= tolerance:
                self.release_control()

                final_error_deg = math.degrees(error)

                print(
                    f"[TURN] target={angle_deg:.1f} deg "
                    f"final_error={final_error_deg:.1f} deg"
                )

                return True

            # Timeout based on MuJoCo simulation time.
            if current_time - last_debug_time >= 0.5:
                print(
                    f"[TURN DEBUG] "
                    f"error={math.degrees(error):+.1f}deg "
                    f"cmd_wz={wz:+.3f} "
                    f"yaw={math.degrees(current_yaw):+.1f}deg"
                )
                last_debug_time = current_time

                return False

            # Proportional controller.
            wz = kp * error

            # Clamp maximum command.
            wz = float(np.clip(wz, -max_wz, max_wz))

            # Maintain enough command for the locomotion policy to
            # physically rotate the robot.
            if abs(wz) < min_wz:
                wz = math.copysign(min_wz, error)

            
            print(
                f"[TURN DEBUG] "
                f"error={math.degrees(error):+.1f}deg "
                f"cmd_wz={wz:+.3f} "
                f"yaw={math.degrees(current_yaw):+.1f}deg"
            )
            
            self.set_velocity(
                0.0,
                0.0,
                wz,
            )

            time.sleep(0.02)
    