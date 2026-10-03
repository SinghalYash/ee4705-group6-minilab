"""Task 3 command executor.

Executes structured commands produced by the LLM parser using
Task 2 MotionSkills and Task 4 object navigation.
"""

from __future__ import annotations

import threading
import time


# Shared cancellation signal.
# Task 4 goto_object() can observe this event.
executor_cancel_event = threading.Event()


def _get_robot():
    """Get the robot registered by the integrated simulation."""

    from task_4 import get_robot

    robot = get_robot()

    if robot is None:
        raise RuntimeError(
            "Robot platform has not been registered yet. "
            "Start the integrated simulation first."
        )

    return robot

def stop_execution():
    """Cancel the current mission and immediately stop the robot."""

    executor_cancel_event.set()

    try:
        robot = _get_robot()
        robot.skills.stop()
    except RuntimeError:
        pass

    print("[STOP] current mission cancelled")


def execute_actions(result):
    """Execute all accepted actions sequentially."""

    if not result.get("accepted", False):
        return

    actions = result.get("actions", [])

    if not actions:
        return

    robot = _get_robot()
    skills = robot.skills

    executor_cancel_event.clear()

    start_time = time.perf_counter()

    for index, action in enumerate(actions, start=1):

        if executor_cancel_event.is_set():
            print("[DONE] status=CANCELLED")
            return

        action_type = action["action"]

        # --------------------------------------------------
        # MOVE
        # --------------------------------------------------

        if action_type == "move":

            vx = action["vx"]
            vy = action["vy"]
            wz = action["wz"]
            duration = action["duration"]

            print(
                f"[EXEC] action={index}/{len(actions)} "
                f"move "
                f"vx={vx} "
                f"vy={vy} "
                f"wz={wz} "
                f"t={duration} s"
            )

            start_sim_time = skills.get_sim_time()

            skills.move(
                vx,
                vy,
                wz,
                duration,
            )

            while True:

                if executor_cancel_event.is_set():
                    skills.stop()
                    print("[EXEC] cancelled")
                    print("[DONE] status=CANCELLED")
                    return

                current_sim_time = skills.get_sim_time()

                if current_sim_time - start_sim_time >= duration:
                    break

                time.sleep(0.02)

        # --------------------------------------------------
        # TURN
        # --------------------------------------------------

        elif action_type == "turn":

            angle = action["angle_deg"]

            print(
                f"[EXEC] action={index}/{len(actions)} "
                f"turn angle={angle} deg"
            )

            success = skills.turn(angle)

            if not success:
                print(
                    f"[DONE] status=FAIL "
                    f"reason=turn_timeout"
                )
                return

        # --------------------------------------------------
        # GOTO OBJECT
        # --------------------------------------------------

        elif action_type == "goto_object":

            print(
                f"[EXEC] action={index}/{len(actions)} "
                f"goto_object "
                f"class={action['class']} "
                f"color={action.get('color')}"
            )

            from task_4 import run_goto_object_action

            result_task4 = run_goto_object_action(
                action,
                cancel_event=executor_cancel_event,
            )

            if (
                result_task4 is not None
                and getattr(result_task4, "status", None) == "FAIL"
            ):
                return

        # --------------------------------------------------
        # STOP
        # --------------------------------------------------

        elif action_type == "stop":

            print(
                f"[EXEC] action={index}/{len(actions)} stop"
            )

            stop_execution()
            return

        # --------------------------------------------------
        # CHAT / CLARIFICATION
        # --------------------------------------------------

        elif action_type == "chat":

            reply = (
                action.get("reply")
                or result.get("reason")
                or "Please clarify your command."
            )

            print(f"Robot: {reply}")

        # --------------------------------------------------
        # UNKNOWN
        # --------------------------------------------------

        else:
            print(
                f"[EXEC] unsupported action={action_type}"
            )
            skills.stop()
            return

    elapsed = time.perf_counter() - start_time

    print(
        f"[DONE] actions={len(actions)} "
        f"t={elapsed:.1f} s"
    )