"""Unified bonus interface for EE4705 MiniLab.

Supports:
- typed robot commands through the Task 3 LLM parser
- speech-to-text commands
- VLM scene description / visual question answering
- YOLO vs VLM bounding-box comparison
- multi-goal autonomous missions
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from task_4 import get_robot


# --------------------------------------------------
# Reuse Task 3 without modifying its existing imports
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
TASK3_DIR = ROOT / "task_3"

if str(TASK3_DIR) not in sys.path:
    sys.path.insert(0, str(TASK3_DIR))

from llm_parser import parse_command, print_command
from executor import execute_actions, stop_execution
from chat_interface import listen_for_command


# --------------------------------------------------
# Task 5 functionality
# --------------------------------------------------

from task_5.vlm_interface import ask_robot
from task_5.bbox_compare import compare_bbox

# Task 5 functionality
from task_5.vlm_interface import ask_robot
from task_5.bbox_compare import compare_bbox


def run_vlm_chat():
    """Run the unified bonus interface."""

    robot = get_robot()

    if robot is None:
        raise RuntimeError(
            "Robot is not registered. "
            "Run this through the integrated simulator."
        )

    history = []
    execution_thread = None

    print("=" * 60)
    print("EE4705 Bonus Interface")
    print("=" * 60)
    print("Typed English commands : normal robot control")
    print("/voice                : speech input")
    print("/see <question>       : visual question answering")
    print("/bbox [color] <object>: compare YOLO and VLM boxes")
    print("stop                  : cancel current mission")
    print("exit                  : quit")
    print("=" * 60)

    while True:

        user_text = input("\nYou> ").strip()

        # ==================================================
        # EXIT
        # ==================================================

        if user_text.lower() == "exit":
            print("Exiting Bonus Interface.")
            break

        if not user_text:
            continue

        # ==================================================
        # SPEECH INPUT
        # ==================================================

        if user_text.lower() == "/voice":

            spoken_text = listen_for_command()

            if not spoken_text:
                continue

            user_text = spoken_text

        # ==================================================
        # EMERGENCY STOP
        # ==================================================

        if user_text.lower() in {
            "stop",
            "stop now",
            "emergency stop",
        }:

            print("[CMD] actions=stop n=1")
            stop_execution()
            continue

        # ==================================================
        # BOUNDING BOX COMPARISON
        # ==================================================

        if user_text.lower().startswith("/bbox"):

            parts = user_text.split()

            if len(parts) < 2:
                print("Usage: /bbox [color] <object>")
                continue

            if len(parts) >= 3:
                color = parts[1].lower()
                object_class = " ".join(parts[2:]).lower()
            else:
                color = None
                object_class = parts[1].lower()

            print(
                f"[BBOX] comparing target="
                f"{color + ' ' if color else ''}"
                f"{object_class}"
            )

            try:
                compare_bbox(
                    robot,
                    object_class,
                    color,
                )
            except Exception as exc:
                print(
                    f"[BBOX] comparison failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            continue

        # ==================================================
        # VLM VISUAL QUESTION ANSWERING
        # ==================================================

        if user_text.lower().startswith("/see"):

            question = user_text[4:].strip()

            if not question:
                question = "What can you see?"

            try:
                answer = ask_robot(
                    robot,
                    question,
                )

                print(f"Robot: {answer}")

            except Exception as exc:
                print(
                    f"[VLM] request failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            continue

        # ==================================================
        # NORMAL ENGLISH ROBOT COMMAND
        # ==================================================

        try:
            result = parse_command(
                user_text,
                history,
            )

        except Exception as exc:
            print(
                f"[LLM] request failed: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        print_command(result)

        # Maintain dialogue history.
        history.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": str(result),
            }
        )

        if not result.get("accepted", False):
            continue

        # Don't start another mission while one is active.
        if (
            execution_thread is not None
            and execution_thread.is_alive()
        ):
            print(
                "Robot: I am still executing the current mission. "
                "Type 'stop' to cancel it."
            )
            continue

        execution_thread = threading.Thread(
            target=execute_actions,
            args=(result,),
            name="task5-executor",
            daemon=True,
        )

        execution_thread.start()