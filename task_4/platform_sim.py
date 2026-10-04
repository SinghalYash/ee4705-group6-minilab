"""Task 2 platform without the viewer, for Task 4 evaluation (Student C).

Uses Student A's actual Task 2 pieces:
  * scene      : task_2/scene/task2_scene.xml (composed with the robot exactly
                 like RuntimeScene does, via runtime_control.compose_scene)
  * camera     : task_2.camera_pipeline.FrontCameraPipeline (320x240 @ 15 Hz)
  * motion     : task_2.motion_skills.MotionSkills (set_velocity/stop/pose)
  * policy/PD  : play.py's observation builder, joint remapping, ONNX policy
                 and compute_pd_torques, called the same way as in play.py

Only the outer loop is Task 4's (run_eval.run_realtime): it steps physics
paced to the wall clock like play.py, while goto_object() runs in a worker
thread, so evaluation behaves like the live system. Unlike play.py it can
start the robot at any pose for each trial. The interactive path (play.py's
own loop, used for the video) is task_4/run_integrated.py.

Not reproduced from play.py: the browser panel's live tuning/randomisation
and motor-delay sliders (they default to the nominal values used here).
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "quadruped_mujoco"
for p in (ROOT, REPO / "src", REPO / "eg"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mujoco  # noqa: E402
import onnxruntime as ort  # noqa: E402

from runtime_control import MapSpec, compose_scene, compute_pd_torques  # noqa: E402
from runtime_control import make_standard_robot_cameras  # noqa: E402
from quadruped_mujoco.eg import play  # noqa: E402  (A's play.py: only helpers/constants are used)
from task_2.camera_pipeline import FrontCameraPipeline  # noqa: E402
from task_2.motion_skills import MotionSkills  # noqa: E402

from robot_adapter import PlatformRobot  # noqa: E402

TASK2_SCENE = ROOT / "task_2" / "scene" / "task2_scene.xml"
BUILD_DIR = Path(__file__).resolve().parent / ".build"


def build_model(scene_xml: Path = TASK2_SCENE, map_name: str = "task2_scene") -> mujoco.MjModel:
    """Robot + one terrain-only map, composed the same way RuntimeScene does."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out = BUILD_DIR / f"{map_name}_composed.xml"
    compose_scene(
        robot_xml=play.DEFAULT_ROBOT_XML,
        map_specs={map_name: MapSpec(scene_xml)},
        output_path=out,
        robot_body_name="trunk",
        robot_cameras=make_standard_robot_cameras(prefix="dog"),
    )
    return mujoco.MjModel.from_xml_path(str(out))


class PlatformSim:
    """Task 2 platform, stepped by the caller. Exposes RobotAPI via self.robot."""

    def __init__(self, scene_xml: Path = TASK2_SCENE, spawn=(0.0, 0.0, 0.0),
                 width: int = 320, height: int = 240, perception_hz: float = 15.0):
        cfg = yaml.safe_load(play.DEFAULT_CONFIG.read_text())
        self.cfg = cfg
        self.dt = cfg["simulation_dt"]
        self.decimation = cfg["control_decimation"]
        self.cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
        self.kps = np.array(cfg["kps"], dtype=np.float64)
        self.kds = np.array(cfg["kds"], dtype=np.float64)
        self.tau_limits = np.array(
            [play.TAU_LIMIT_HIP_THIGH, play.TAU_LIMIT_HIP_THIGH, play.TAU_LIMIT_CALF] * 4)

        self.model = build_model(scene_xml)
        self.model.opt.timestep = self.dt
        self.data = mujoco.MjData(self.model)
        self.policy = ort.InferenceSession(str(play.DEFAULT_ONNX),
                                           providers=["CPUExecutionProvider"])
        self.in_name = self.policy.get_inputs()[0].name
        self.out_name = self.policy.get_outputs()[0].name
        self._cam_args = dict(camera_name="dog_front_camera", width=width,
                              height=height, perception_hz=perception_hz)
        self.camera = None
        self.reset(*spawn)

    # ---------------------------------------------------------------- setup
    def reset(self, x=0.0, y=0.0, yaw_deg=0.0):
        a = math.radians(yaw_deg) / 2
        play.reset_robot(self.model, self.data, play.DEFAULT_ANGLES_MUJOCO,
                         position=[x, y, 0.42], quaternion=[math.cos(a), 0, 0, math.sin(a)])
        # Fresh A objects per trial: FrontCameraPipeline keeps its last render
        # time and has no reset, and sim time restarts at 0 here.
        if self.camera is not None:
            self.camera.close()
        self.camera = FrontCameraPipeline(model=self.model, **self._cam_args)
        self.skills = MotionSkills()
        self.robot = PlatformRobot(self.camera, self.skills, self.model)

        self.obs_hist = play.ObsHistoryBuffer(6, 46)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.target_q = play.DEFAULT_ANGLES_MUJOCO.copy()
        self.step_count = 0
        for _ in range(6):
            self.obs_hist.push(self._obs(np.zeros(3, dtype=np.float32)))
        for _ in range(int(1.0 / self.dt)):  # settle on its feet
            self.step()

    def _obs(self, cmd):
        d = self.data
        q = d.qpos[3:7]
        return play.build_single_obs(
            np.array([q[1], q[2], q[3], q[0]]), d.qvel[3:6].astype(np.float64),
            d.qpos[7:19].astype(np.float64)[play.MUJOCO_TO_ISAAC],
            d.qvel[6:18].astype(np.float64)[play.MUJOCO_TO_ISAAC],
            self.last_action, play.DEFAULT_ANGLES_ISAAC, cmd, self.cmd_scale,
            self.cfg["ang_vel_scale"], self.cfg["dof_pos_scale"],
            self.cfg["dof_vel_scale"], self.cfg["clip_obs"], height_cmd=0.25)

    # ------------------------------------------------------------- stepping
    def step(self) -> bool:
        """One 200 Hz physics step, in play.py's order. True if a new frame was rendered."""
        cmd, _owns = self.skills.update(self.data)   # A's command source
        if self.step_count % self.decimation == 0:
            self.obs_hist.push(self._obs(cmd))
            act = self.policy.run([self.out_name], {self.in_name: self.obs_hist.get()})[0][0]
            act = np.clip(act, -10.0, 10.0)
            self.last_action = act.astype(np.float32)
            self.target_q = (act * self.cfg["action_scale"]
                             + play.DEFAULT_ANGLES_ISAAC)[play.ISAAC_TO_MUJOCO]
        d = self.data
        d.ctrl[:12] = compute_pd_torques(self.target_q, d.qpos[7:19], d.qvel[6:18],
                                         self.kps, self.kds, torque_limit=self.tau_limits)
        mujoco.mj_step(self.model, d)
        self.step_count += 1
        return self.camera.update(d)                 # A's camera pipeline

    def fallen(self) -> bool:
        return self.data.qpos[2] < 0.15

    def sim_time(self) -> float:
        return float(self.data.time)

    # ------------------------------------------- RobotAPI (delegates to A)
    def get_latest_frame(self):
        return self.robot.get_latest_frame()

    def set_velocity(self, vx, vy, wz):
        self.robot.set_velocity(vx, vy, wz)

    def stop(self):
        self.robot.stop()

    def release(self):
        self.robot.release()

    def get_base_pose(self):
        return self.robot.get_base_pose()   # A's pose from mj_data.qpos

    def camera_model(self):
        return self.robot.camera_model()

    def close(self):
        if self.camera is not None:
            self.camera.close()