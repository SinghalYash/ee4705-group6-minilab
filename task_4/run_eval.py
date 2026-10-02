"""Task 4 evaluation harness (Student C).

Runs goto_object on many (target, start pose) trials on the Task 2 platform
(platform_sim.PlatformSim: A's scene, A's 320x240 FrontCameraPipeline, A's
MotionSkills, play.py's policy/PD), lock-step so results do not depend on
laptop speed, and writes:
    results/eval_results.csv       one row per trial
    results/eval_summary.md        table + aggregate metrics for the report
    results/frames/<trial>.png     annotated camera frame at the end of each trial
    results/videos/<trial>.mp4     onboard-camera clip (with --video)

Usage:
    python run_eval.py                 # all trials
    python run_eval.py --only 3 4      # a subset
    python run_eval.py --video         # also save camera clips
    python run_eval.py --dev-sandbox   # DEV ONLY: old standalone sandbox sim

Metrics (definitions used in the report):
    detection     : the target was detected (class + colour) at least once
    grounding     : at the end, the nearest scene object to the robot is the
                    requested one (right object chosen, e.g. green vs red chair)
    success       : [MISSION] status=SUCCESS with ground-truth d <= 0.80 m
    d             : planar trunk-to-object-centre distance from ground truth
"""
from __future__ import annotations

import perception  # noqa: F401  (import torch before mujoco, see perception.py)

import argparse
import csv
import math
import time
from pathlib import Path

import cv2

from goto_object import GroundTruth, goto_object_lockstep
from perception import draw_detections, normalize_class

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

# id, class, colour, start x, y, yaw_deg, note, expected
TRIALS = [
    (1, "chair", "green", 0.0, 0.0, 0.0, "visible; red chair also in view", "SUCCESS"),
    (2, "chair", "red", 0.0, 0.0, 0.0, "visible; green chair also in view", "SUCCESS"),
    (3, "chair", "blue", 0.0, 0.0, 0.0, "not initially visible (behind)", "SUCCESS"),
    (4, "stop sign", None, 0.0, 0.0, 0.0, "not initially visible; no colour given", "SUCCESS"),
    (5, "chair", "green", 0.0, 0.0, 180.0, "facing away", "SUCCESS"),
    (6, "chair", "red", 1.0, -2.5, 90.0, "side start", "SUCCESS"),
    (7, "chair", "blue", 0.5, 2.5, -90.0, "not initially visible", "SUCCESS"),
    (8, "stop sign", "red", 1.0, 0.0, 90.0, "side start", "SUCCESS"),
    (9, "chair", "green", -1.0, -1.0, 45.0, "diagonal start", "SUCCESS"),
    (10, "chair", "red", 1.5, 0.3, -20.0, "close start, green chair nearby", "SUCCESS"),
    (11, "chair", "blue", -4.5, -0.5, 0.0, "far start", "SUCCESS"),
    (12, "chair", "yellow", 0.0, 0.0, 0.0, "target absent (negative test)", "FAIL"),
]


def nearest_object(gt: GroundTruth, x, y):
    return min(gt.objects, key=lambda o: math.hypot(x - o["x"], y - o["y"]))


def make_sim(spawn, dev_sandbox: bool):
    if dev_sandbox:
        from sim_backend import SandboxSim  # DEV ONLY
        return SandboxSim(spawn=spawn)
    from platform_sim import PlatformSim
    return PlatformSim(spawn=spawn)


def run_trial(trial, gt: GroundTruth, video: bool, dev_sandbox: bool = False):
    tid, cls, color, sx, sy, syaw, note, expected = trial
    sim = make_sim((sx, sy, syaw), dev_sandbox)
    t0 = sim.sim_time()
    lines = []

    def log(s):
        line = f"[t={sim.sim_time() - t0:5.1f}s] {s}"
        lines.append(s)
        print("   ", line, flush=True)

    target = f"{color or ''} {cls}".strip()
    print(f"\n=== Trial {tid}: go to the {target}  start=({sx},{sy},{syaw:.0f}deg)  [{note}]")
    print(f"    [CMD] goto_object class={cls.replace(' ', '_')} color={color or 'any'}")

    writer = None
    last = {}

    def on_frame(rgb, ctrl, r):
        nonlocal writer
        last["rgb"], last["ctrl"] = rgb, ctrl
        if video:
            if writer is None:
                (OUT / "videos").mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(OUT / "videos" / f"trial_{tid:02d}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                         (rgb.shape[1], rgb.shape[0]))
            img = draw_detections(rgb, ctrl.last_dets, ctrl.last_target)
            cv2.putText(img, f"{ctrl.state}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)
            writer.write(img)

    wall = time.time()
    res = goto_object_lockstep(sim, cls, color, gt=gt, log=log, on_frame=on_frame)
    wall = time.time() - wall
    if writer is not None:
        writer.release()

    # evidence frame
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    if "rgb" in last:
        ctrl = last["ctrl"]
        cv2.imwrite(str(OUT / "frames" / f"trial_{tid:02d}.png"),
                    draw_detections(last["rgb"], ctrl.last_dets, ctrl.last_target))

    x, y, _ = sim.get_base_pose()
    if hasattr(sim, "close"):
        sim.close()
    exists = any(normalize_class(o["class"]) == normalize_class(cls)
                 and (color is None or o["color"] == color) for o in gt.objects)
    d_true = gt.distance(cls, color, x, y) if exists else None
    near = nearest_object(gt, x, y)
    grounded = exists and normalize_class(near["class"]) == normalize_class(cls) and \
        (color is None or near["color"] == color)
    detected = any(s["seen"] for s in res.trace) if exists else None
    success = res.status == "SUCCESS" and d_true is not None and d_true <= 0.80
    passed = (success if expected == "SUCCESS" else res.status == "FAIL")
    return {
        "trial": tid, "target": target, "start": f"({sx}, {sy}, {syaw:.0f})", "note": note,
        "expected": expected, "status": res.status, "reason": res.reason,
        "t_s": round(res.t, 1), "d_m": None if d_true is None else round(d_true, 2),
        "detected": detected, "grounded": grounded if exists else None,
        "success": success, "passed": passed, "frames": res.frames,
        "wall_s": round(wall, 1), "log": " | ".join(lines),
    }


def summarize(rows):
    pos = [r for r in rows if r["expected"] == "SUCCESS"]
    neg = [r for r in rows if r["expected"] == "FAIL"]
    ds = [r["d_m"] for r in pos if r["success"]]
    ts = [r["t_s"] for r in pos if r["success"]]

    def pct(a, b):
        return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"

    md = ["| # | Target | Start (x, y, yaw) | Condition | Result | t (s) | d (m) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        res = r["status"] if r["status"] == "SUCCESS" else f"FAIL ({r['reason']})"
        md.append(f"| {r['trial']} | {r['target']} | {r['start']} | {r['note']} | {res} | "
                  f"{r['t_s']} | {'-' if r['d_m'] is None else r['d_m']} |")
    md += ["", "| Metric | Value |", "|---|---|",
           f"| Detection (target seen) | {pct(sum(bool(r['detected']) for r in pos), len(pos))} |",
           f"| Grounding (right object chosen) | {pct(sum(bool(r['grounded']) for r in pos), len(pos))} |",
           f"| Approach success (SUCCESS and d <= 0.80 m) | {pct(sum(r['success'] for r in pos), len(pos))} |",
           f"| Correct rejection of absent target | {pct(sum(r['passed'] for r in neg), len(neg))} |"]
    if ds:
        md.append(f"| Final d, mean (min-max) | {sum(ds) / len(ds):.2f} m "
                  f"({min(ds):.2f}-{max(ds):.2f}) |")
        md.append(f"| Time to found, mean (min-max) | {sum(ts) / len(ts):.1f} s "
                  f"({min(ts):.1f}-{max(ts):.1f}) |")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--dev-sandbox", action="store_true",
                    help="DEV ONLY: use the old standalone sandbox instead of the Task 2 platform")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    gt = GroundTruth()
    trials = [t for t in TRIALS if not args.only or t[0] in args.only]
    rows = [run_trial(t, gt, args.video, args.dev_sandbox) for t in trials]
    with open(OUT / "eval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    (OUT / "eval_summary.md").write_text(summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main()
