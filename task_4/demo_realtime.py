"""Real-time, threaded demo of Task 4 in the sandbox (Student C).

Mimics the final architecture: the simulation runs in the main thread at
real time, a chat thread reads typed commands, and an executor thread runs
goto_object() against the RobotAPI while the sim keeps stepping.

The command parsing here is a PLACEHOLDER regex so Task 4 can be exercised
before Student B's LLM parser exists. It is not the Task 3 deliverable and is
replaced by B's parser during integration.

    python demo_realtime.py                        # type: go to the green chair
    python demo_realtime.py --commands "go to the red chair" "find the stop sign"
    python demo_realtime.py --view                 # show the camera with boxes (needs a display)
"""
from __future__ import annotations

import perception  # noqa: F401  (import torch before mujoco)

import argparse
import queue
import re
import threading
import time
from pathlib import Path

import cv2

from goto_object import get_detector, goto_object
from perception import COLOR_NAMES
from sim_backend import SandboxSim

HERE = Path(__file__).resolve().parent.parent
CLASSES = ["chair", "stop sign", "sports ball", "ball"]


def placeholder_parse(text: str):
    """Stand-in for Student B's LLM parser. Returns a goto_object dict or None."""
    t = text.lower()
    cls = next((c for c in CLASSES if c in t), None)
    if cls is None:
        return None
    color = next((c for c in COLOR_NAMES if re.search(rf"\b{c}\b", t)), None)
    return {"action": "goto_object", "class": cls, "color": color}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commands", nargs="*", help="scripted commands instead of typing")
    ap.add_argument("--spawn", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    ap.add_argument("--view", action="store_true")
    args = ap.parse_args()

    sim = SandboxSim(HERE / "test_scene_placeholder_for_task4.xml", spawn=tuple(args.spawn))
    get_detector()  # load YOLO before the clock starts
    jobs: queue.Queue = queue.Queue()
    cancel = threading.Event()
    busy = threading.Event()
    done = threading.Event()

    def chat():
        if args.commands:
            for c in args.commands:
                while busy.is_set() or not jobs.empty():
                    time.sleep(0.1)
                print(f"\nUser: {c}", flush=True)
                jobs.put(c)
                time.sleep(0.5)
            while busy.is_set() or not jobs.empty():
                time.sleep(0.1)
            done.set()
            return
        while True:
            try:
                line = input("\nUser: ").strip()
            except EOFError:
                done.set()
                return
            if line.lower() in ("quit", "exit"):
                done.set()
                return
            if line.lower() == "stop":
                cancel.set()
                continue
            jobs.put(line)

    def executor():
        while not done.is_set():
            try:
                text = jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            busy.set()
            cmd = placeholder_parse(text)
            if cmd is None:
                print("[CMD] rejected reason=placeholder parser only understands object goals",
                      flush=True)
                busy.clear()
                continue
            print(f"[CMD] goto_object class={cmd['class'].replace(' ', '_')} "
                  f"color={cmd['color'] or 'any'}", flush=True)
            cancel.clear()
            goto_object(sim, cmd["class"], cmd["color"], cancel_event=cancel,
                        log=lambda s: print(s, flush=True))
            busy.clear()

    threading.Thread(target=chat, daemon=True).start()
    threading.Thread(target=executor, daemon=True).start()

    # main thread: real-time simulation loop (never blocks on YOLO or input)
    t_wall0, t_sim0 = time.time(), sim.sim_time()
    while not done.is_set():
        new = sim.step()
        if args.view and new:
            frame = sim.get_latest_frame()[0]
            cv2.imshow("dog_front_camera", frame[..., ::-1])
            cv2.waitKey(1)
        lag = (sim.sim_time() - t_sim0) - (time.time() - t_wall0)
        if lag > 0:
            time.sleep(lag)
    print("\n[INFO] demo finished")


if __name__ == "__main__":
    main()
