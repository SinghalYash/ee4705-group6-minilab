"""Standalone headless simulator for developing Task 4 before Task 2 lands.

It reuses the example repo's pieces unchanged (scene composition, the ONNX
walking policy, the observation builder, the PD loop) and exposes the
interface Task 4 expects from Task 2 (see interfaces.py):

    get_latest_frame() -> (rgb uint8 HxWx3, sim_time) | None
    set_velocity(vx, vy, wz) / stop()
    get_base_pose() -> (x, y, yaw)
    sim_time() -> float

When Student A's real platform is ready, the controller is pointed at A's
object instead and this file is no longer used.
"""
from __future__ import annotations

import math
import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

REPO = Path(os.environ.get("QUADRUPED_REPO", Path(__file__).resolve().parent.parent/"quadruped_mujoco")).resolve()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "eg"))

from runtime_control import MapSpec, compose_scene, compute_pd_torques, make_standard_robot_cameras
import play

HERE = Path(__file__).resolve().parent.parent
ROBOT_XML = REPO / "eg" / "dog" / "xml" / "dog_terrain.xml"
FRONT_CAM = "dog_front_camera"


def yaw_from_quat_wxyz(q) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def build_model(scene_xml: Path, out_dir: Path) -> mujoco.MjModel:
    """Compose robot + one terrain-only scene map, the same way RuntimeScene does."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "task4_composed.xml"
    compose_scene(
        robot_xml=ROBOT_XML,
        map_specs={"task4": MapSpec(scene_xml)},
        output_path=out,
        robot_body_name="trunk",
        robot_cameras=make_standard_robot_cameras(prefix="dog"),
    )
    return mujoco.MjModel.from_xml_path(str(out))


class SandboxSim:
    """Headless quadruped + onboard camera. Thread-safe command/frame access."""

    def __init__(
        self,
        scene_xml: Path,
        width: int = 640,
        height: int = 480,
        perception_hz: float = 15.0,
        spawn=(0.0, 0.0, 0.0),
    ):
        cfg = yaml.safe_load((REPO / "eg" / "dog.yaml").read_text())
        self.cfg = cfg
        self.dt = cfg["simulation_dt"]
        self.decimation = cfg["control_decimation"]
        self.cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)

        self.model = build_model(scene_xml, HERE / ".build")
        self.model.opt.timestep = self.dt
        self.data = mujoco.MjData(self.model)
        self.policy = ort.InferenceSession(
            str(REPO / "eg" / "model_3400.onnx"), providers=["CPUExecutionProvider"]
        )
        self.in_name = self.policy.get_inputs()[0].name
        self.out_name = self.policy.get_outputs()[0].name

        self.width, self.height = width, height
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, FRONT_CAM)
        # render every N physics steps: 200 Hz / 15 Hz -> 13 steps (15.4 Hz)
        self.render_every = max(1, round(1.0 / (perception_hz * self.dt)))

        self.tau_limits = np.array(
            [play.TAU_LIMIT_HIP_THIGH, play.TAU_LIMIT_HIP_THIGH, play.TAU_LIMIT_CALF] * 4
        )
        self.kps = np.array(cfg["kps"], dtype=np.float64)
        self.kds = np.array(cfg["kds"], dtype=np.float64)

        self._lock = threading.Lock()
        self._cmd = np.zeros(3, dtype=np.float32)
        self._frame = None
        self._frame_t = -1.0
        self.reset(*spawn)

    # ------------------------------------------------------------------ setup
    def reset(self, x=0.0, y=0.0, yaw_deg=0.0):
        a = math.radians(yaw_deg) / 2
        play.reset_robot(
            self.model, self.data, play.DEFAULT_ANGLES_MUJOCO,
            position=[x, y, 0.42], quaternion=[math.cos(a), 0, 0, math.sin(a)],
        )
        self.obs_hist = play.ObsHistoryBuffer(6, 46)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.target_q = play.DEFAULT_ANGLES_MUJOCO.copy()
        self.step_count = 0
        with self._lock:
            self._cmd[:] = 0
            self._frame, self._frame_t = None, -1.0
        for _ in range(6):
            self.obs_hist.push(self._obs(np.zeros(3, dtype=np.float32)))
        # let the robot settle onto its feet before anything moves
        for _ in range(int(1.0 / self.dt)):
            self.step()

    def _obs(self, cmd):
        d = self.data
        q = d.qpos[3:7]
        quat_xyzw = np.array([q[1], q[2], q[3], q[0]])
        return play.build_single_obs(
            quat_xyzw, d.qvel[3:6].astype(np.float64),
            d.qpos[7:19].astype(np.float64)[play.MUJOCO_TO_ISAAC],
            d.qvel[6:18].astype(np.float64)[play.MUJOCO_TO_ISAAC],
            self.last_action, play.DEFAULT_ANGLES_ISAAC, cmd, self.cmd_scale,
            self.cfg["ang_vel_scale"], self.cfg["dof_pos_scale"],
            self.cfg["dof_vel_scale"], self.cfg["clip_obs"], height_cmd=0.25,
        )

    # --------------------------------------------------------------- stepping
    def step(self) -> bool:
        """One 200 Hz physics step. Returns True when a new camera frame was rendered."""
        with self._lock:
            cmd = self._cmd.copy()
        if self.step_count % self.decimation == 0:
            self.obs_hist.push(self._obs(cmd))
            act = self.policy.run([self.out_name], {self.in_name: self.obs_hist.get()})[0][0]
            act = np.clip(act, -10.0, 10.0)
            self.last_action = act.astype(np.float32)
            tq_isaac = act * self.cfg["action_scale"] + play.DEFAULT_ANGLES_ISAAC
            self.target_q = tq_isaac[play.ISAAC_TO_MUJOCO]
        d = self.data
        tau = compute_pd_torques(
            self.target_q, d.qpos[7:19], d.qvel[6:18], self.kps, self.kds,
            torque_limit=self.tau_limits,
        )
        d.ctrl[:12] = tau
        mujoco.mj_step(self.model, d)
        self.step_count += 1
        if self.step_count % self.render_every == 0:
            self.renderer.update_scene(d, camera=self.cam_id)
            img = self.renderer.render().copy()
            with self._lock:
                self._frame, self._frame_t = img, float(d.time)
            return True
        return False

    def fallen(self) -> bool:
        return self.data.qpos[2] < 0.15

    # ------------------------------------------------- Task 2-style interface
    def get_latest_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame, self._frame_t

    def set_velocity(self, vx: float, vy: float, wz: float):
        with self._lock:
            self._cmd[:] = np.clip([vx, vy, wz], -1.0, 1.0)

    def stop(self):
        self.set_velocity(0.0, 0.0, 0.0)

    def get_base_pose(self):
        q = self.data.qpos
        return float(q[0]), float(q[1]), yaw_from_quat_wxyz(q[3:7])

    def sim_time(self) -> float:
        return float(self.data.time)

    # ------------------------------------------------------------- utilities
    def render_static(self, x, y, yaw_deg, z=None):
        """Place the robot (no physics) and render one front-camera frame."""
        a = math.radians(yaw_deg) / 2
        self.data.qpos[:3] = [x, y, self.data.qpos[2] if z is None else z]
        self.data.qpos[3:7] = [math.cos(a), 0, 0, math.sin(a)]
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return self.renderer.render().copy()

    def camera_info(self):
        """Intrinsics/extrinsics Task 4 needs for monocular ranging."""
        fovy = float(self.model.cam_fovy[self.cam_id])
        f = (self.height / 2) / math.tan(math.radians(fovy) / 2)
        cam_x = self.data.cam_xpos[self.cam_id]
        return {"fovy_deg": fovy, "f_px": f, "cam_z": float(cam_x[2])}
