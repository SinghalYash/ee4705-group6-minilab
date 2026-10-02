"""Task 4 package (Student C).

Task 4 modules import each other by plain name (from approach import ...), so
this package adds its own folder to sys.path. Import Task 4 from other tasks
through this package:

    from task_4 import run_goto_object_action
    result = run_goto_object_action(action, cancel_event=cancel_event)

The import inside the function is deliberate: it loads torch/YOLO only when
the first goto_object action runs, not when the caller's module is imported.
"""
import pathlib
import sys

_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def run_goto_object_action(action, cancel_event=None, robot=None, log=print):
    """Run one parsed {"action": "goto_object", "class": ..., "color": ...}.

    Blocks until the mission ends; returns a MissionResult whose .status is
    "SUCCESS" or "FAIL". Prints [SEARCH]/[DETECT]/[FOUND]/[MISSION].
    """
    from goto_object import run_goto_object_action as _run

    return _run(action, cancel_event=cancel_event, robot=robot, log=log)


def get_robot():
    """The platform robot registered by task_4/run_integrated.py (None before start-up).

    robot.skills is Student A's MotionSkills, robot.camera A's FrontCameraPipeline.
    """
    from robot_adapter import get_robot as _get

    return _get()
