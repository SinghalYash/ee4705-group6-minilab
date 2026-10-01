"""
Dog Sim2Sim — IsaacGym -> MuJoCo

EE4705 Task 2 integration:
- custom object scene
- 15 Hz onboard front-camera pipeline
- timed move queue
- continuous velocity RobotAPI
- closed-loop yaw turn

Joint order:
  MuJoCo:   FL, FR, RR, RL
  IsaacGym: FL, FR, RL, RR

The rear legs therefore require RL/RR remapping.
"""

import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml


# ---------------------------------------------------------------------------
# Paths / Task 2 imports
# ---------------------------------------------------------------------------

DEMO_DIR = Path(__file__).resolve().parent
RUNTIME_CONTROL_DIR = DEMO_DIR.parent
PACKAGE_SRC = RUNTIME_CONTROL_DIR / "src"
PROJECT_ROOT = RUNTIME_CONTROL_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_2.camera_pipeline import FrontCameraPipeline
from task_2.motion_skills import MotionSkills


try:
    from runtime_control import (
        MapSpec,
        MotorCommandDelay,
        RuntimeScene,
        bundled_map_specs,
        compute_pd_torques,
        make_runtime_config,
        make_standard_robot_cameras,
        scale_torque_limits,
        setup_tracking_camera,
        standard_camera_options,
    )
except ModuleNotFoundError as exc:
    if exc.name != "runtime_control":
        raise

    sys.path.insert(0, str(PACKAGE_SRC))

    from runtime_control import (
        MapSpec,
        MotorCommandDelay,
        RuntimeScene,
        bundled_map_specs,
        compute_pd_torques,
        make_runtime_config,
        make_standard_robot_cameras,
        scale_torque_limits,
        setup_tracking_camera,
        standard_camera_options,
    )


DEFAULT_CONFIG = DEMO_DIR / "dog.yaml"
DEFAULT_ONNX = DEMO_DIR / "model_3400.onnx"
DEFAULT_ROBOT_XML = DEMO_DIR / "dog" / "xml" / "dog_terrain.xml"

MAP_SPECS = bundled_map_specs()

TASK2_SCENE = PROJECT_ROOT / "task_2" / "scene" / "task2_scene.xml"

MAP_SPECS["task2_scene"] = MapSpec(
    path=TASK2_SCENE,
    strict=True,
)

ROBOT_CAMERAS = make_standard_robot_cameras(prefix="dog")
CAMERA_OPTIONS = standard_camera_options(prefix="dog")


# ---------------------------------------------------------------------------
# IsaacGym <-> MuJoCo joint ordering
# ---------------------------------------------------------------------------

MUJOCO_TO_ISAAC = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]
ISAAC_TO_MUJOCO = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]

DEFAULT_ANGLES_ISAAC = np.array(
    [
        -0.1, -0.8, -1.5,   # FL
         0.1,  0.8,  1.5,   # FR
         0.1, -1.0, -1.5,   # RL
        -0.1,  1.0,  1.5,   # RR
    ],
    dtype=np.float64,
)

DEFAULT_ANGLES_MUJOCO = DEFAULT_ANGLES_ISAAC[ISAAC_TO_MUJOCO]

TAU_LIMIT_HIP_THIGH = 23.7
TAU_LIMIT_CALF = 35.55
OUTPUT_PRINT_SCALE = 0.25


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def quat_rotate_inverse(q, v):
    q_w = q[3]
    q_vec = q[:3]

    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0

    return a - b + c


class ObsHistoryBuffer:
    def __init__(self, history_len, single_obs_dim):
        self.history_len = history_len
        self.single_obs_dim = single_obs_dim
        self.total_dim = history_len * single_obs_dim

        self.buffer = np.zeros(
            self.total_dim,
            dtype=np.float32,
        )

    def push(self, new_obs):
        self.buffer[self.single_obs_dim:] = (
            self.buffer[:-self.single_obs_dim].copy()
        )

        self.buffer[:self.single_obs_dim] = new_obs

    def get(self):
        return self.buffer.reshape(1, -1).copy()

    def reset(self):
        self.buffer[:] = 0.0


def build_single_obs(
    quat_xyzw,
    omega,
    joint_q_isaac,
    joint_dq_isaac,
    last_action_isaac,
    default_angles_isaac,
    cmd,
    cmd_scale,
    ang_vel_scale,
    dof_pos_scale,
    dof_vel_scale,
    clip_obs,
    height_cmd=0.25,
):
    """Build one 46-dimensional policy observation."""

    obs = np.zeros(46, dtype=np.float32)

    # 3 command values
    obs[0:3] = cmd * cmd_scale[:3]

    # 3 body angular velocity
    obs[3:6] = omega.astype(np.float32) * ang_vel_scale

    # 3 projected gravity
    gravity_world = np.array(
        [0.0, 0.0, -1.0],
        dtype=np.float64,
    )

    projected_gravity = quat_rotate_inverse(
        quat_xyzw,
        gravity_world,
    )

    obs[6:9] = projected_gravity.astype(np.float32)

    # 12 joint-position deviations
    obs[9:21] = (
        (joint_q_isaac - default_angles_isaac)
        * dof_pos_scale
    ).astype(np.float32)

    # 12 joint velocities
    obs[21:33] = (
        joint_dq_isaac * dof_vel_scale
    ).astype(np.float32)

    # 12 previous actions
    obs[33:45] = last_action_isaac

    # 1 commanded body height
    obs[45] = np.float32(
        (height_cmd - 0.25) / 0.1
    )

    return np.clip(
        obs,
        -clip_obs,
        clip_obs,
    )


# ---------------------------------------------------------------------------
# Keyboard control
# ---------------------------------------------------------------------------

try:
    import evdev
except ImportError:
    evdev = None


_pressed_keys = set()


def _find_keyboards():
    keyboards = []

    if evdev is None:
        return keyboards

    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
            caps = dev.capabilities()

            if evdev.ecodes.EV_KEY in caps:
                keys = caps[evdev.ecodes.EV_KEY]

                if evdev.ecodes.KEY_W in keys:
                    keyboards.append(dev)

        except Exception:
            pass

    return keyboards


_DIR_SCANCODES = (
    {}
    if evdev is None
    else {
        evdev.ecodes.KEY_W: "w",
        evdev.ecodes.KEY_S: "s",
        evdev.ecodes.KEY_A: "a",
        evdev.ecodes.KEY_D: "d",
        evdev.ecodes.KEY_Q: "q",
        evdev.ecodes.KEY_E: "e",
        evdev.ecodes.KEY_UP: "w",
        evdev.ecodes.KEY_DOWN: "s",
        evdev.ecodes.KEY_LEFT: "a",
        evdev.ecodes.KEY_RIGHT: "d",
    }
)


def _evdev_keyboard_thread(dev):
    try:
        for event in dev.read_loop():
            if event.type != evdev.ecodes.EV_KEY:
                continue

            if event.code not in _DIR_SCANCODES:
                continue

            direction = _DIR_SCANCODES[event.code]

            if event.value == 1:
                _pressed_keys.add(direction)

            elif event.value == 0:
                _pressed_keys.discard(direction)

    except Exception as exc:
        print(f"[KEYBOARD] evdev error: {exc}")


_kb_devs = _find_keyboards()

for _dev in _kb_devs:
    threading.Thread(
        target=_evdev_keyboard_thread,
        args=(_dev,),
        daemon=True,
    ).start()


if _kb_devs:
    print(
        f"[KEYBOARD] evdev: listening on "
        f"{len(_kb_devs)} keyboard device(s)"
    )
elif evdev is None:
    print(
        "[KEYBOARD] evdev not installed; "
        "browser keyboard control is available"
    )


GLFW_KEY_R = 82
GLFW_KEY_F = 70
GLFW_KEY_Z = 90
GLFW_KEY_T = 84
GLFW_KEY_Y = 89
GLFW_KEY_X = 88
GLFW_KEY_C = 67
GLFW_KEY_SPACE = 32


height_cmd = 0.25
reset_flag = False
print_action_flag = False
capture_camera_flag = False


def key_callback(keycode):
    global height_cmd
    global reset_flag
    global print_action_flag
    global capture_camera_flag

    if keycode == GLFW_KEY_R:
        height_cmd = max(
            0.20,
            height_cmd - 0.02,
        )

    elif keycode == GLFW_KEY_F:
        height_cmd = min(
            0.35,
            height_cmd + 0.02,
        )

    elif keycode == GLFW_KEY_Z:
        height_cmd = 0.25

    elif keycode == GLFW_KEY_T:
        reset_flag = True

    elif keycode == GLFW_KEY_Y:
        print_action_flag = not print_action_flag

    elif keycode == GLFW_KEY_X:
        _pressed_keys.clear()

        if "motion_skills" in globals():
            motion_skills.stop()

    elif keycode == GLFW_KEY_C:
        capture_camera_flag = True

    elif (
        keycode == GLFW_KEY_SPACE
        and "runtime" in globals()
    ):
        runtime.request_push()

    if "runtime" in globals():
        if keycode in (
            GLFW_KEY_R,
            GLFW_KEY_F,
            GLFW_KEY_Z,
        ):
            runtime_config["command"]["height"] = height_cmd
            runtime.sync_height_to_panel()

        elif keycode == GLFW_KEY_X:
            runtime.sync_stop_to_panel()


def get_commands():
    vx = (
        1.0 if "w" in _pressed_keys
        else -1.0 if "s" in _pressed_keys
        else 0.0
    )

    vy = (
        1.0 if "a" in _pressed_keys
        else -1.0 if "d" in _pressed_keys
        else 0.0
    )

    wz = (
        1.0 if "q" in _pressed_keys
        else -1.0 if "e" in _pressed_keys
        else 0.0
    )

    return np.array(
        [vx, vy, wz],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Reset / runtime configuration
# ---------------------------------------------------------------------------

def reset_robot(
    model,
    data,
    default_angles_mujoco,
    position=None,
    quaternion=None,
):
    mujoco.mj_resetData(
        model,
        data,
    )

    data.qpos[:3] = (
        position
        if position is not None
        else [0.0, 0.0, 0.42]
    )

    data.qpos[3:7] = (
        quaternion
        if quaternion is not None
        else [1.0, 0.0, 0.0, 0.0]
    )

    data.qpos[7:19] = default_angles_mujoco
    data.qvel[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )


def build_runtime_config(args, kps, kds):
    map_spawns = {
        name: {
            "position": [0.0, 0.0, 0.42],
            "quaternion": [1.0, 0.0, 0.0, 0.0],
        }
        for name in MAP_SPECS
    }

    map_spawns["rc26_track"] = {
        "position": [3.7, -9.0, 0.45],
        "quaternion": [
            -0.7071068,
            0.0,
            0.0,
            0.7071068,
        ],
    }

    map_spawns["google_barkour"] = {
        "position": [1.0, -1.5, 0.42],
        "quaternion": [1.0, 0.0, 0.0, 0.0],
    }

    map_labels = {
        "rc26_track": "26RC Track",
        "race_track": "Race Track",
        "stairs": "Stairs",
        "cross_stairs": "Cross Stairs",
        "cross_slope": "Cross Slope",
        "google_barkour": "Google Barkour",
        "gap_jump": "Gap Jump",
        "hurdles": "Hurdles",
        "suspended_steps": "Suspended Steps",
        "perlin_rough": "Perlin Rough Terrain",
        "dynamic_obstacles": "Dynamic Obstacles",
        "task2_scene": "Task 2 Object Scene",
    }

    return make_runtime_config(
        gui=args.gui,
        title="Dog MuJoCo Live Tuning",
        maps=map_labels,
        map_spawns=map_spawns,
        kp=kps[0],
        kd=kds[0],
        torque_limit=TAU_LIMIT_CALF,

        # Start in Task 2's object scene.
        initial_position=map_spawns["task2_scene"]["position"],
        initial_quaternion=map_spawns["task2_scene"]["quaternion"],

        command=(1.0, 1.0, 1.0, 0.25),
        height_range=(0.2, 0.35),
        cameras=CAMERA_OPTIONS,
        port=args.gui_port,

        tracking_camera={
            "camera_distance": 2.0,
            "camera_azimuth": 135.0,
            "camera_elevation": -25.0,
        },

        randomization={
            "kp": [32.0, 48.0],
            "kd": [0.8, 1.2],
            "torque_limit": [28.0, 42.0],
            "motor_strength": [0.9, 1.1],
            "motor_delay_ms": [0.0, 15.0],
            "mass_scale": [0.9, 1.1],
            "payload_mass": [0.0, 2.0],
            "friction_scale": [0.5, 1.5],
            "gravity_z": [-10.3, -9.3],
        },

        push={
            "force_range": [40.0, 100.0],
            "duration_range": [0.10, 0.25],
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--onnx",
        type=Path,
        default=DEFAULT_ONNX,
    )

    parser.add_argument(
        "--no-policy",
        action="store_true",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
    )

    parser.add_argument(
        "--duration",
        type=float,
    )

    parser.add_argument(
        "--verbose-inference",
        action="store_true",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
    )

    parser.add_argument(
        "--gui-port",
        type=int,
        default=8765,
    )

    # Temporary Task 2(iv) controller test.
    parser.add_argument(
        "--test-turn",
        type=float,
        default=None,
        metavar="DEG",
        help="run one closed-loop relative turn test",
    )

    args = parser.parse_args()

    if args.gui and args.headless:
        parser.error(
            "--gui and --headless cannot be used together"
        )

    if (
        args.duration is not None
        and args.duration <= 0
    ):
        parser.error(
            "--duration must be greater than 0"
        )

    config_path = DEFAULT_CONFIG

    policy_path = (
        None
        if args.no_policy
        else str(
            args.onnx.expanduser().resolve()
        )
    )

    if (
        policy_path is not None
        and not Path(policy_path).is_file()
    ):
        raise FileNotFoundError(
            f"ONNX model not found: {policy_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    simulation_duration = (
        args.duration
        if args.duration is not None
        else config["simulation_duration"]
    )

    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(
        config["kps"],
        dtype=np.float64,
    )

    kds = np.array(
        config["kds"],
        dtype=np.float64,
    )

    action_scale = config["action_scale"]
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]

    cmd_scale = np.array(
        config["cmd_scale"],
        dtype=np.float32,
    )

    clip_obs = config.get(
        "clip_obs",
        100.0,
    )

    runtime_config = build_runtime_config(
        args,
        kps,
        kds,
    )

    tau_limits_mujoco = np.array(
        [
            TAU_LIMIT_HIP_THIGH,
            TAU_LIMIT_HIP_THIGH,
            TAU_LIMIT_CALF,
        ]
        * 4,
        dtype=np.float64,
    )

    NUM_ONE_STEP_OBS = 46
    HISTORY_LEN = 6
    NUM_ACTIONS = 12

    print("\n" + "=" * 60)
    print("[CONFIG] Dog — Task 2 integrated platform")
    print("=" * 60)
    print("  IsaacGym dof: FL, FR, RL, RR")
    print("  MuJoCo dof:   FL, FR, RR, RL")
    print(
        f"  mapping: MUJOCO_TO_ISAAC = "
        f"{MUJOCO_TO_ISAAC}"
    )
    print(
        f"  observation: "
        f"{NUM_ONE_STEP_OBS} x {HISTORY_LEN} = "
        f"{NUM_ONE_STEP_OBS * HISTORY_LEN}"
    )
    print(
        f"  camera: dog_front_camera "
        f"320x240 @ 15 Hz"
    )
    print("=" * 60)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    scene = RuntimeScene(
        robot_xml=DEFAULT_ROBOT_XML,
        map_specs=MAP_SPECS,
        runtime_config=runtime_config,
        robot_body_name="trunk",
        robot_cameras=ROBOT_CAMERAS,
        dynamic_obstacle_map="dynamic_obstacles",
        output_name="dog_all_maps.xml",
    )

    scene.open()

    mj_model = scene.model
    mj_model.opt.timestep = simulation_dt

    mj_data = scene.data
    runtime = scene.runtime

    motor_delay = MotorCommandDelay(
        simulation_dt
    )

    camera_pipeline = FrontCameraPipeline(
        model=mj_model,
        camera_name="dog_front_camera",
        width=320,
        height=240,
        perception_hz=15.0,
    )

    motion_skills = MotionSkills()

    reset_robot(
        mj_model,
        mj_data,
        DEFAULT_ANGLES_MUJOCO,
        runtime_config["simulation"]["initial_position"],
        runtime_config["simulation"]["initial_quaternion"],
    )

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    if not args.no_policy:
        policy = ort.InferenceSession(
            policy_path,
            providers=["CPUExecutionProvider"],
        )

        input_name = policy.get_inputs()[0].name
        output_name = policy.get_outputs()[0].name

        print(
            f"[ONNX] "
            f"{policy.get_inputs()[0].shape} -> "
            f"{policy.get_outputs()[0].shape}"
        )

    obs_history = ObsHistoryBuffer(
        HISTORY_LEN,
        NUM_ONE_STEP_OBS,
    )

    target_q_mujoco = (
        DEFAULT_ANGLES_MUJOCO.copy()
    )

    action_isaac = np.zeros(
        NUM_ACTIONS,
        dtype=np.float64,
    )

    last_action_isaac = np.zeros(
        NUM_ACTIONS,
        dtype=np.float32,
    )

    # Warm observation history.
    for _ in range(HISTORY_LEN):
        quat_wxyz = mj_data.qpos[3:7]

        quat_xyzw = np.array(
            [
                quat_wxyz[1],
                quat_wxyz[2],
                quat_wxyz[3],
                quat_wxyz[0],
            ]
        )

        omega = mj_data.qvel[
            3:6
        ].astype(np.float64)

        joint_q_isaac = (
            mj_data.qpos[7:19]
            .astype(np.float64)[MUJOCO_TO_ISAAC]
        )

        joint_dq_isaac = (
            mj_data.qvel[6:18]
            .astype(np.float64)[MUJOCO_TO_ISAAC]
        )

        obs = build_single_obs(
            quat_xyzw,
            omega,
            joint_q_isaac,
            joint_dq_isaac,
            last_action_isaac,
            DEFAULT_ANGLES_ISAAC,
            np.zeros(3, dtype=np.float32),
            cmd_scale,
            ang_vel_scale,
            dof_pos_scale,
            dof_vel_scale,
            clip_obs,
            height_cmd=0.25,
        )

        obs_history.push(obs)

    # ------------------------------------------------------------------
    # Viewer / loop
    # ------------------------------------------------------------------

    print(
        "\n"
        "  W/S: forward/back   A/D: strafe   Q/E: turn\n"
        "  R/F: height         T: reset      X: emergency stop\n"
        "  C: capture onboard front-camera frame\n"
    )

    display = scene.viewer(
        args.gui or args.headless,
        key_callback=key_callback,
    )

    count = 0
    inference_count = 0
    turn_test_started = False

    with display as viewer:
        if not args.gui and not args.headless:
            setup_tracking_camera(
                viewer,
                mj_model,
                "trunk",
                distance=2.0,
                azimuth=135.0,
                elevation=-25.0,
            )

        wall_start = time.time()

        while (
            viewer.is_running()
            and time.time() - wall_start
            < simulation_duration
        ):
            step_start = time.time()

            # ----------------------------------------------------------
            # Manual command source
            # ----------------------------------------------------------

            browser_command_active = (
                runtime.update_command(
                    _pressed_keys
                )
            )

            if browser_command_active:
                manual_cmd = np.array(
                    [
                        runtime_config["command"]["linear_x"],
                        runtime_config["command"]["linear_y"],
                        runtime_config["command"]["yaw"],
                    ],
                    dtype=np.float32,
                )

                height_cmd = (
                    runtime_config["command"]["height"]
                )

            else:
                manual_cmd = get_commands()

            # ----------------------------------------------------------
            # Task 2 autonomous command source
            # ----------------------------------------------------------

            skill_cmd, skill_has_control = (
                motion_skills.update(mj_data)
            )

            cmd = (
                skill_cmd
                if skill_has_control
                else manual_cmd
            )

            # Optional development test.
            if (
                args.test_turn is not None
                and not turn_test_started
                and mj_data.time >= 1.0
            ):
                requested_angle = float(
                    args.test_turn
                )

                def run_turn_test():
                    print(
                        f"[TEST] starting closed-loop "
                        f"{requested_angle:.1f}-degree turn"
                    )

                    success = motion_skills.turn(
                        requested_angle
                    )

                    print(
                        f"[TEST] turn success={success}"
                    )

                threading.Thread(
                    target=run_turn_test,
                    daemon=True,
                ).start()

                turn_test_started = True

            # ----------------------------------------------------------
            # Runtime / reset
            # ----------------------------------------------------------

            runtime_state = runtime.runtime_control(
                mj_model,
                mj_data,
            )

            if runtime.consume_reset():
                reset_flag = True

            if reset_flag:
                motion_skills.release_control()

                reset_robot(
                    mj_model,
                    mj_data,
                    DEFAULT_ANGLES_MUJOCO,
                    runtime_config["simulation"]["initial_position"],
                    runtime_config["simulation"]["initial_quaternion"],
                )

                obs_history.reset()
                last_action_isaac[:] = 0.0
                action_isaac[:] = 0.0

                count = 0
                inference_count = 0

                height_cmd = (
                    runtime_config["command"]["height"]
                )

                motor_delay.reset()
                runtime.reset_simulation_state()

                reset_flag = False

            # ----------------------------------------------------------
            # Read state
            # ----------------------------------------------------------

            joint_q_mujoco = (
                mj_data.qpos[7:19]
                .astype(np.float64)
            )

            joint_dq_mujoco = (
                mj_data.qvel[6:18]
                .astype(np.float64)
            )

            joint_q_isaac = (
                joint_q_mujoco[
                    MUJOCO_TO_ISAAC
                ]
            )

            joint_dq_isaac = (
                joint_dq_mujoco[
                    MUJOCO_TO_ISAAC
                ]
            )

            quat_wxyz = mj_data.qpos[3:7]

            quat_xyzw = np.array(
                [
                    quat_wxyz[1],
                    quat_wxyz[2],
                    quat_wxyz[3],
                    quat_wxyz[0],
                ]
            )

            omega = (
                mj_data.qvel[3:6]
                .astype(np.float64)
            )

            # ----------------------------------------------------------
            # 50 Hz ONNX policy
            # ----------------------------------------------------------

            if count % control_decimation == 0:
                if not args.no_policy:
                    single_obs = build_single_obs(
                        quat_xyzw,
                        omega,
                        joint_q_isaac,
                        joint_dq_isaac,
                        last_action_isaac,
                        DEFAULT_ANGLES_ISAAC,
                        cmd,
                        cmd_scale,
                        ang_vel_scale,
                        dof_pos_scale,
                        dof_vel_scale,
                        clip_obs,
                        height_cmd=height_cmd,
                    )

                    obs_history.push(
                        single_obs
                    )

                    obs_input = (
                        obs_history.get()
                    )

                    action_raw = policy.run(
                        [output_name],
                        {input_name: obs_input},
                    )[0][0]

                    action_isaac[:] = np.clip(
                        action_raw,
                        -10.0,
                        10.0,
                    )

                    last_action_isaac = (
                        action_isaac.astype(
                            np.float32
                        )
                    )

                    target_q_isaac = (
                        action_isaac
                        * action_scale
                        + DEFAULT_ANGLES_ISAAC
                    )

                    target_q_mujoco = (
                        target_q_isaac[
                            ISAAC_TO_MUJOCO
                        ]
                    )

                    inference_count += 1

                else:
                    target_q_mujoco = (
                        DEFAULT_ANGLES_MUJOCO.copy()
                    )

            # ----------------------------------------------------------
            # 200 Hz PD / physics
            # ----------------------------------------------------------

            applied_target_q = motor_delay.apply(
                target_q_mujoco,
                runtime_state["motor_delay_ms"],
            )

            effective_tau_limits = (
                scale_torque_limits(
                    tau_limits_mujoco,
                    runtime_state["torque_limit"],
                    reference_limit=TAU_LIMIT_CALF,
                )
            )

            tau = compute_pd_torques(
                applied_target_q,
                joint_q_mujoco,
                joint_dq_mujoco,
                runtime_state["kp"],
                runtime_state["kd"],
                motor_strength=runtime_state[
                    "motor_strength"
                ],
                torque_limit=effective_tau_limits,
            )

            mj_data.ctrl[:NUM_ACTIONS] = tau

            runtime.apply_external_forces(
                mj_model,
                mj_data,
            )

            mujoco.mj_step(
                mj_model,
                mj_data,
            )

            count += 1

            # ----------------------------------------------------------
            # 15 Hz onboard perception
            # ----------------------------------------------------------

            camera_pipeline.update(
                mj_data
            )

            if capture_camera_flag:
                output_path = (
                    PROJECT_ROOT
                    / "task_2"
                    / "front_camera_test.png"
                )

                if camera_pipeline.save_latest_frame(
                    str(output_path)
                ):
                    print(
                        "[CAMERA] capture complete"
                    )

                capture_camera_flag = False

            # ----------------------------------------------------------
            # Diagnostics
            # ----------------------------------------------------------

            if (
                count
                % (control_decimation * 50)
                == 0
            ):
                _, _, current_yaw = (
                    motion_skills.get_base_pose()
                )

                print(
                    f"[SIM] "
                    f"t={mj_data.time:.1f}s "
                    f"H={mj_data.qpos[2]:.3f} "
                    f"yaw={np.degrees(current_yaw):+.1f}deg "
                    f"cmd="
                    f"[{cmd[0]:+.2f},"
                    f"{cmd[1]:+.2f},"
                    f"{cmd[2]:+.2f}]"
                )

            if (
                not args.gui
                and not args.headless
            ):
                viewer.sync()

            elapsed = (
                time.time() - step_start
            )

            if simulation_dt - elapsed > 0:
                time.sleep(
                    simulation_dt - elapsed
                )

    camera_pipeline.close()
    scene.close()

    print("\n[INFO] simulation finished")