"""Task 4 TEST executor (Student C) — for testing Task 4 end to end only.

This is NOT the Task 3 deliverable. Student B's executor (task_3/executor.py)
is the real one; select it in task_4/run_integrated.py once it drives the
platform and dispatches goto_object. This file exists so Task 4 can be tested
on the live platform before that.

    parser   : Student B's parse_command()/print_command() from task_3/llm_parser.py
               (LLM, needs OPENAI_API_KEY), or a fixed JSON action list in
               B's schema (no LLM) via --actions-json.
    executor : this file. goto_object -> task_4.run_goto_object_action();
               move/turn/stop -> Student A's MotionSkills via the Task 4 adapter;
               chat -> prints the reply.

Each command runs in this chat thread; play.py's loop keeps the simulation
running in the main thread.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK3_DIR = ROOT / "task_3"


# ------------------------------------------------------------------ parser
def load_task3_parser():
    """Return (parse_command, print_command) from B's llm_parser, or (None, reason).

    B's llm_parser creates its OpenAI client at import time, so without
    OPENAI_API_KEY the import itself fails; that is reported, not hidden.
    """
    if str(TASK3_DIR) not in sys.path:
        sys.path.insert(0, str(TASK3_DIR))
    try:
        import task_3.llm_parser  # Student B
    except Exception as exc:  # noqa: BLE001  (OpenAIError, ImportError, ...)
        return None, f"{type(exc).__name__}: {exc}"
    return (task_3.llm_parser.parse_command, task_3.llm_parser.print_command), None


def print_command_fallback(result):
    """FALLBACK used only when B's llm_parser cannot be imported (no API key).
    Same output format as B's print_command()."""
    if not result.get("accepted"):
        print(f"[CMD] rejected reason={result.get('reason') or 'unknown'}")
        return
    parts = []
    for a in result["actions"]:
        k = a["action"]
        if k == "move":
            parts.append(f"move(vx={a['vx']}, vy={a['vy']}, wz={a['wz']}, {a['duration']} s)")
        elif k == "turn":
            parts.append(f"turn({a['angle_deg']} deg)")
        elif k == "goto_object":
            parts.append(f"goto_object(class={a['class']}, color={a['color']})")
        else:
            parts.append(k)
    print(f"[CMD] actions={', '.join(parts)} n={len(result['actions'])}")


def actions_to_result(actions):
    """Wrap a list of B-schema actions into B's top-level result object."""
    full = []
    for a in actions:
        item = {k: None for k in ("vx", "vy", "wz", "duration", "angle_deg",
                                  "class", "color", "reply")}
        item.update(a)
        full.append(item)
    return {"accepted": True, "reason": None, "actions": full}


# ---------------------------------------------------------------- executor
class Task4TestExecutor:
    """Runs one parsed result (B's schema) on the platform, in order."""

    def __init__(self, robot):
        self.robot = robot
        self.cancel = threading.Event()

    def _wait_sim(self, seconds: float):
        """Wait `seconds` of simulation time (A's move() is non-blocking)."""
        skills = self.robot.skills
        t0 = skills.get_sim_time()
        while skills.get_sim_time() - t0 < seconds:
            if self.cancel.is_set():
                return
            time.sleep(0.01)

    def execute(self, result):
        if not result.get("accepted"):
            return
        from task_4 import run_goto_object_action

        actions = result["actions"]
        n = len(actions)
        t_start = self.robot.skills.get_sim_time()
        self.cancel.clear()
        for i, a in enumerate(actions, start=1):
            kind = a["action"]
            if kind == "move":
                print(f"[EXEC] action={i}/{n} move vx={a['vx']} vy={a['vy']} "
                      f"wz={a['wz']} t={a['duration']} s")
                self.robot.skills.move(a["vx"] or 0.0, a["vy"] or 0.0,
                                       a["wz"] or 0.0, a["duration"])
                # A has no is_idle() yet: wait the commanded sim time plus one
                # control step so the next action cannot clear the queue early.
                self._wait_sim(float(a["duration"]) + 0.03)
                self.robot.release()
            elif kind == "turn":
                print(f"[EXEC] action={i}/{n} turn angle={a['angle_deg']} deg")
                self.robot.turn(a["angle_deg"])
            elif kind == "goto_object":
                print(f"[EXEC] action={i}/{n} goto_object class={a['class']} color={a['color']}")
                res = run_goto_object_action(a, cancel_event=self.cancel)
                if res.status != "SUCCESS":
                    print(f"[DONE] actions={i}/{n} aborted after goto_object failure")
                    return
            elif kind == "stop":
                print(f"[EXEC] action={i}/{n} stop")
                self.robot.stop()
                self.robot.release()
            elif kind == "chat":
                print(f"Robot> {a.get('reply') or ''}")
        dt = self.robot.skills.get_sim_time() - t_start
        print(f"[DONE] actions={n} t={dt:.1f} s")


# -------------------------------------------------------------- chat loop
def run_chat(robot, actions_json: str | None = None, on_exit=None):
    """Terminal loop: type a command (ENTER), it is parsed and executed.

    actions_json: run this fixed B-schema action list once instead of typing
    (routing test without an LLM).
    """
    executor = Task4TestExecutor(robot)
    parser, err = load_task3_parser()
    print_cmd = parser[1] if parser else print_command_fallback

    if actions_json is not None:
        actions = json.loads(actions_json)
        if isinstance(actions, dict):
            actions = actions.get("actions", [actions])
        result = actions_to_result(actions)
        print(f"[TEST] fixed JSON actions (no LLM): {json.dumps(actions)}")
        print_cmd(result)
        executor.execute(result)
        if on_exit:
            on_exit()
        return

    if parser is None:
        print(f"[TEST] Task 3 LLM parser unavailable ({err}). "
              "Set OPENAI_API_KEY, or use --actions-json.")
        if on_exit:
            on_exit()
        return

    parse_command = parser[0]
    history = []
    print("EE4705 Robot Command Interface (Task 4 TEST executor). Type 'exit' to quit.")
    while True:
        try:
            text = input("You> ").strip()
        except EOFError:
            break
        if text.lower() == "exit":
            break
        result = parse_command(text, history)
        history += [{"role": "user", "content": text},
                    {"role": "assistant", "content": json.dumps(result)}]
        print_cmd(result)
        executor.execute(result)
    if on_exit:
        on_exit()
