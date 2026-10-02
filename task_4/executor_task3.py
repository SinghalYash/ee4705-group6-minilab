"""Task 3 executor path (Student B's code only) for the integrated run.

Nothing here parses or executes commands: it starts Student B's own chat loop
(task_3/chat_interface.py: run_chat -> parse_command -> print_command ->
execute_actions) in a background thread, next to play.py's simulation loop.

For goto_object to work through this path, B's execute_actions() must route
goto_object to Task 4 with one line (see task_4/__init__.py):

    elif action_type == "goto_object":
        from task_4 import run_goto_object_action
        run_goto_object_action(action)

and use the platform registered by the integration script for move/turn:

    from task_4 import get_robot
    robot = get_robot()          # robot.skills is A's MotionSkills

Until B does that, this path runs B's code exactly as it is (currently its
move/turn are print+sleep stubs and goto_object is ignored); this file does
not patch or replace any of it.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

TASK3_DIR = Path(__file__).resolve().parent.parent / "task_3"


def start(on_exit=None) -> threading.Thread:
    """Start B's chat loop in a daemon thread. Returns the thread."""
    if str(TASK3_DIR) not in sys.path:
        sys.path.insert(0, str(TASK3_DIR))
    try:
        import task_3.chat_interface  # Student B (imports llm_parser -> OpenAI client)
    except Exception as exc:  # noqa: BLE001
        print(f"[TASK3] cannot start Student B's chat interface: "
              f"{type(exc).__name__}: {exc}")
        if on_exit:
            on_exit()
        return None

    def _run():
        task_3.chat_interface.run_chat()
        if on_exit:
            on_exit()

    thread = threading.Thread(target=_run, name="task3-chat", daemon=True)
    thread.start()
    return thread
