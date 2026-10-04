"""Task 4 evaluation harness (Student C).

Runs goto_object on many (target, start pose) trials on the Task 2 platform
(platform_sim.PlatformSim: A's scene, A's 320x240 FrontCameraPipeline, A's
MotionSkills, play.py's policy/PD) in REAL TIME, the same way the live system
runs: physics steps in the main thread paced to the wall clock (like play.py),
while goto_object() runs in a worker thread and always takes the newest camera
frame. Frames that arrive while YOLO is busy are skipped, so results depend on
the computer's speed and can vary between runs (use --repeats). Writes:
    results/eval_results.csv       one row per trial
    results/eval_summary.md        table + aggregate metrics for the report
    results/frames/<trial>.png     annotated camera frame at the end of each trial
    results/videos/<trial>.mp4     onboard-camera clip (with --video)

Usage:
    python run_eval.py                 # all trials
    python run_eval.py --only 3 4      # a subset
    python run_eval.py --video         # also save camera clips
    python run_eval.py --repeats 3     # run every trial 3 times
    python run_eval.py --dev-sandbox   # DEV ONLY: old standalone sandbox sim

Metrics (definitions used in the report):
    groups        : near starts (trials 1-12, target 2.1-4.6 m away) and far starts
                    (13-16, 5.1-5.6 m, i.e. consecutive commands without a reset)
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
import threading
import time
from pathlib import Path

import cv2

from goto_object import GroundTruth, get_detector, goto_object
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
    (11, "chair", "blue", -4.5, -0.5, 0.0, "long side start", "SUCCESS"),
    (12, "chair", "yellow", 0.0, 0.0, 0.0, "target absent (negative test)", "FAIL"),
    # Far starts (5.1-5.6 m): where a previous mission ends, i.e. consecutive
    # commands without resetting the robot. Beyond the range covered by 1-12.
    (13, "chair", "red", -1.7, 1.0, 148.0, "far: after blue-chair mission", "SUCCESS"),
    (14, "chair", "blue", 2.4, -0.8, -34.0, "far: after red-chair mission", "SUCCESS"),
    (15, "chair", "blue", 2.6, 0.9, 37.0, "far: after green-chair mission", "SUCCESS"),
    (16, "chair", "green", -1.2, -2.5, -121.0, "far: after stop-sign mission", "SUCCESS"),
]
FAR_IDS = {13, 14, 15, 16}


def nearest_object(gt: GroundTruth, x, y):
    return min(gt.objects, key=lambda o: math.hypot(x - o["x"], y - o["y"]))


def make_sim(spawn, dev_sandbox: bool):
    if dev_sandbox:
        from sim_backend import SandboxSim  # DEV ONLY
        return SandboxSim(spawn=spawn)
    from platform_sim import PlatformSim
    return PlatformSim(spawn=spawn)


def run_realtime(sim, cls, color, gt, log, on_new_frame=None):
    """Run goto_object() exactly as the live system does.

    Worker thread: goto_object(robot, ...) polls the newest camera frame,
    runs YOLO and commands A's MotionSkills. Main thread: steps physics and
    sleeps whenever simulation time gets ahead of the wall clock (play.py's
    pacing), so the simulation never waits for YOLO.
    """
    robot = getattr(sim, "robot", sim)
    out, done, cancel = {}, threading.Event(), threading.Event()

    def worker():
        try:
            out["res"] = goto_object(robot, cls, color, cancel_event=cancel, gt=gt, log=log)
        except Exception as exc:  # noqa: BLE001  (reported, not hidden)
            out["err"] = exc
        finally:
            done.set()

    th = threading.Thread(target=worker, name="goto_object", daemon=True)
    t_wall0, t_sim0 = time.time(), sim.sim_time()
    th.start()
    fell = False
    while not done.is_set():
        new = sim.step()
        if new and on_new_frame is not None:
            on_new_frame()
        if not fell and sim.fallen():
            fell = True
            cancel.set()
        lag = (sim.sim_time() - t_sim0) - (time.time() - t_wall0)
        if lag > 0:
            time.sleep(lag)
    th.join()
    if "err" in out:
        raise out["err"]
    res = out["res"]
    if fell:
        res.status, res.reason = "FAIL", "robot_fell"
    return res


def run_trial(trial, gt: GroundTruth, video: bool, dev_sandbox: bool = False, run: int = 1):
    tid, cls, color, sx, sy, syaw, note, expected = trial
    sim = make_sim((sx, sy, syaw), dev_sandbox)
    t0 = sim.sim_time()
    lines = []
    tag = f"{tid:02d}" if run == 1 else f"{tid:02d}_run{run}"

    def log(s):
        line = f"[t={sim.sim_time() - t0:5.1f}s] {s}"
        lines.append(s)
        print("   ", line, flush=True)

    target = f"{color or ''} {cls}".strip()
    print(f"\n=== Trial {tid}{'' if run == 1 else f' (run {run})'}: go to the {target}  "
          f"start=({sx},{sy},{syaw:.0f}deg)  [{note}]")
    print(f"    [CMD] goto_object class={cls.replace(' ', '_')} color={color or 'any'}")

    writer = None

    def on_new_frame():  # raw onboard-camera clip (with --video)
        nonlocal writer
        if not video:
            return
        rgb = sim.get_latest_frame()[0]
        if writer is None:
            (OUT / "videos").mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(OUT / "videos" / f"trial_{tag}.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                     (rgb.shape[1], rgb.shape[0]))
        writer.write(rgb[..., ::-1].copy())

    wall = time.time()
    res = run_realtime(sim, cls, color, gt, log, on_new_frame)
    wall = time.time() - wall
    if writer is not None:
        writer.release()

    # evidence frame: last camera image with the detections drawn on it
    (OUT / "frames").mkdir(parents=True, exist_ok=True)
    item = sim.get_latest_frame()
    if item is not None:
        rgb = item[0]
        dets = get_detector().detect(rgb, classes=[normalize_class(cls)])
        want = [d for d in dets if color is None or d.color == color]
        cv2.imwrite(str(OUT / "frames" / f"trial_{tag}.png"),
                    draw_detections(rgb, dets, max(want, key=lambda d: d.conf) if want else None))

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
    if exists and not res.trace:
        detected = None  # mission cancelled before any frame was processed
    d_start = gt.distance(cls, color, sx, sy) if exists else None
    success = res.status == "SUCCESS" and d_true is not None and d_true <= 0.80
    passed = (success if expected == "SUCCESS" else res.status == "FAIL")
    return {
        "trial": tid if run == 1 else f"{tid}.{run}", "run": run,
        "group": "far" if tid in FAR_IDS else "near",
        "target": target, "start": f"({sx}, {sy}, {syaw:.0f})", "note": note,
        "d_start_m": None if d_start is None else round(d_start, 2),
        "expected": expected, "status": res.status, "reason": res.reason,
        "t_s": round(res.t, 1), "d_m": None if d_true is None else round(d_true, 2),
        "detected": detected, "grounded": grounded if exists else None,
        "success": success, "passed": passed,
        "found_too_far": res.status == "SUCCESS" and d_true is not None and d_true > 0.80,
        "frames": res.frames,
        "wall_s": round(wall, 1), "log": " | ".join(lines),
    }


def summarize(rows):
    pos = [r for r in rows if r["expected"] == "SUCCESS"]
    neg = [r for r in rows if r["expected"] == "FAIL"]
    def label(name, g):
        ds = [r["d_start_m"] for r in g if r["d_start_m"] is not None]
        return f"{name} ({min(ds):.1f}-{max(ds):.1f} m)" if ds else name

    near = [r for r in pos if r["group"] == "near"]
    far = [r for r in pos if r["group"] == "far"]
    groups = [(label("Near starts", near), near), (label("Far starts", far), far),
              ("All positive trials", pos)]

    def pct(a, b):
        return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "n/a"

    def span(vals, unit, fmt):
        if not vals:
            return "-"
        return f"{sum(vals) / len(vals):{fmt}} {unit} ({min(vals):{fmt}}-{max(vals):{fmt}})"

    md = ["| # | Target | Start (x, y, yaw) | Start dist (m) | Condition | Result | t (s) | d (m) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        res = r["status"] if r["status"] == "SUCCESS" else f"FAIL ({r['reason']})"
        md.append(f"| {r['trial']} | {r['target']} | {r['start']} | "
                  f"{'-' if r['d_start_m'] is None else r['d_start_m']} | {r['note']} | {res} | "
                  f"{r['t_s']} | {'-' if r['d_m'] is None else r['d_m']} |")
    md += ["", "| Metric | " + " | ".join(g for g, _ in groups) + " |",
           "|---|" + "---|" * len(groups)]

    def row(name, fn):
        md.append(f"| {name} | " + " | ".join(fn(g) for _, g in groups) + " |")

    row("Detection (target seen)", lambda g: pct(sum(bool(r["detected"]) for r in g), len(g)))
    row("Grounding (right object chosen)", lambda g: pct(sum(bool(r["grounded"]) for r in g), len(g)))
    row("Approach success (SUCCESS and d <= 0.80 m)", lambda g: pct(sum(r["success"] for r in g), len(g)))
    row("SUCCESS reported but d > 0.80 m", lambda g: pct(sum(r["found_too_far"] for r in g), len(g)))
    row("Final d of successes, mean (min-max)",
        lambda g: span([r["d_m"] for r in g if r["success"]], "m", ".2f"))
    row("Time to found, mean (min-max)",
        lambda g: span([r["t_s"] for r in g if r["success"]], "s", ".1f"))
    md += ["", f"Correct FAIL for the absent target: {pct(sum(r['passed'] for r in neg), len(neg))}"]
    fails = [r for r in pos if not r["success"]]
    if fails:
        md += ["", "Failed positive trials:"]
        md += [f"- Trial {r['trial']} ({r['target']}, start {r['d_start_m']} m away): "
               f"{r['status']} {r['reason']}, d={r['d_m']} m" for r in fails]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--repeats", type=int, default=1, help="run every trial N times")
    ap.add_argument("--dev-sandbox", action="store_true",
                    help="DEV ONLY: use the old standalone sandbox instead of the Task 2 platform")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    gt = GroundTruth()
    trials = [t for t in TRIALS if not args.only or t[0] in args.only]
    get_detector()  # load YOLO before the first trial's clock starts
    print("[EVAL] real-time mode: physics paced to the wall clock, goto_object in a worker thread")
    rows = [run_trial(t, gt, args.video, args.dev_sandbox, run=k)
            for k in range(1, args.repeats + 1) for t in trials]
    with open(OUT / "eval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    (OUT / "eval_summary.md").write_text(summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main()