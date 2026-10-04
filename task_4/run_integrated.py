"""Integrated Task 2 + 3 + 4 run (Student C).

Starts Student A's play.py loop (Task 2 scene, 320x240 front camera, motion
skills) in the main thread, wraps A's camera pipeline + motion skills in the
Task 4 adapter, registers it for goto_object, and runs a chat/executor thread
next to the simulation.

    python task_4/run_integrated.py                       # native MuJoCo viewer
    python task_4/run_integrated.py --gui                 # browser panel
    python task_4/run_integrated.py --executor task3      # Student B's chat + executor
    python task_4/run_integrated.py --headless --duration 120 \
        --actions-json '[{"action": "goto_object", "class": "chair", "color": "green"}]'
    python task_4/run_integrated.py --dev-sandbox         # DEV ONLY: old sandbox + regex

Unrecognised options (--gui, --headless, --duration, --quiet, ...) go to play.py.
"""
from __future__ import annotations

# torch/YOLO must be imported before MuJoCo creates its GL context (segfault
# seen on Linux otherwise), so perception is the very first import.
import pathlib
import sys

_TASK4 = pathlib.Path(__file__).resolve().parent
if str(_TASK4) not in sys.path:
    sys.path.insert(0, str(_TASK4))
import perception  # noqa: E402,F401

import argparse  # noqa: E402
import os  # noqa: E402
import runpy  # noqa: E402
import threading  # noqa: E402

ROOT = _TASK4.parent
PLAY_PY = ROOT / "quadruped_mujoco" / "eg" / "play.py"

# =============================================================================
# EXECUTOR SELECTION — change this one line (or pass --executor):
#   "task4_test" : Task 4 TEST executor (task_4/executor_task4_test.py), uses
#                  B's LLM parser but Task 4's own executor. For testing now.
#   "task3"      : Student B's chat_interface + executor (task_4/executor_task3.py).
#                  Final path, once B's executor drives the platform and
#                  dispatches goto_object.
EXECUTOR = "task4_test"
# =============================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--executor", choices=["task4_test", "task3", "task5"], default=EXECUTOR)
    ap.add_argument("--actions-json", default=None,
                    help="task4_test only: run this fixed action list (B's schema) "
                         "once instead of typing; no LLM needed")
    ap.add_argument("--dev-sandbox", action="store_true",
                    help="DEV ONLY: old standalone sandbox + regex parser")
    args, play_args = ap.parse_known_args()

    if args.dev_sandbox:
        import dev_sandbox
        dev_sandbox.main(play_args)
        return

    from robot_adapter import PlatformRobot, register_robot

    def on_exit():
        print("[INFO] chat ended, closing simulation")
        sys.stdout.flush()
        os._exit(0)

    def on_platform_ready(camera_pipeline, motion_skills, model, data):
        robot = PlatformRobot(camera_pipeline, motion_skills, model)
        register_robot(robot)
        print(f"[TASK4] platform ready: camera {robot.width}x{robot.height}, "
              f"executor={args.executor}")

        def start_chat():
            robot.wait_until_ready()

            if args.executor == "task3":
                import executor_task3
                executor_task3.start(on_exit=on_exit)

            elif args.executor == "task5":
                from task_5.chat_vlm import run_vlm_chat

                run_vlm_chat()

                if on_exit:
                    on_exit()

            else:
                import executor_task4_test
                executor_task4_test.run_chat(
                    robot,
                    args.actions_json,
                    on_exit=on_exit,
                )

        threading.Thread(target=start_chat, name="chat", daemon=True).start()

    # Student A's play.py, run as __main__ (its own loop, unchanged); the hook in play.py
    # calls on_platform_ready() once the camera pipeline and motion skills exist.
    sys.argv = [str(PLAY_PY)] + play_args
    runpy.run_path(str(PLAY_PY), init_globals={"on_platform_ready": on_platform_ready},
                   run_name="__main__")


if __name__ == "__main__":
    main()
