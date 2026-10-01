"""
Task 2(ii): Onboard camera pipeline.

Provides RGB frames from the MuJoCo robot's front camera for use by
Task 4 perception.

Student A contribution.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import mujoco
import numpy as np
from PIL import Image


class FrontCameraPipeline:
    """Offscreen RGB renderer for the robot's onboard front camera."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_name: str = "dog_front_camera",
        width: int = 640,
        height: int = 480,
        perception_hz: float = 15.0,
    ):
        self.model = model
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.perception_hz = perception_hz

        # Minimum simulation-time interval between perception frames.
        self.frame_period = 1.0 / perception_hz
        self._last_render_time = -np.inf

        # Latest frame shared with Task 4.
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_time: Optional[float] = None

        # Protect the latest frame because Task 4 may read it from
        # another thread.
        self._lock = threading.Lock()

        # Check that the required onboard camera actually exists.
        camera_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )

        if camera_id == -1:
            raise ValueError(
                f"Camera '{camera_name}' does not exist in the MuJoCo model."
            )

        # Ensure MuJoCo's offscreen framebuffer is large enough.
        model.vis.global_.offwidth = max(
            int(model.vis.global_.offwidth), width
        )
        model.vis.global_.offheight = max(
            int(model.vis.global_.offheight), height
        )

        self._renderer = mujoco.Renderer(
            model,
            height=height,
            width=width,
        )

        print(
            f"[CAMERA] initialized camera={camera_name} "
            f"resolution={width}x{height} rate={perception_hz:.1f}Hz"
        )

    def update(self, data: mujoco.MjData) -> bool:
        """
        Render a new RGB frame when the perception period has elapsed.

        This method may be called every physics step. Rendering itself
        occurs only at perception_hz.

        Returns True when a new frame was rendered.
        """

        sim_time = float(data.time)

        if sim_time - self._last_render_time < self.frame_period:
            return False

        self._renderer.update_scene(
            data,
            camera=self.camera_name,
        )

        rgb = self._renderer.render().copy()

        with self._lock:
            self._latest_frame = rgb
            self._latest_time = sim_time

        self._last_render_time = sim_time

        return True

    def get_latest_frame(
        self,
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Return the latest onboard front-camera frame and simulation time.

        Returns:
            (rgb, t), where:
                rgb: HxWx3 uint8 RGB image
                t: simulation timestamp in seconds

            Returns None before the first frame has been rendered.
        """

        with self._lock:
            if self._latest_frame is None:
                return None

            return (
                self._latest_frame.copy(),
                float(self._latest_time),
            )

    def close(self) -> None:
        """Release MuJoCo rendering resources."""

        self._renderer.close()

    def save_latest_frame(self, path: str) -> bool:
        """Save the latest RGB frame to disk."""

        result = self.get_latest_frame()

        if result is None:
            return False

        rgb, timestamp = result

        Image.fromarray(rgb).save(path)

        print(
            f"[CAMERA] saved frame={path} "
            f"shape={rgb.shape} "
            f"t={timestamp:.3f}s"
        )

        return True