"""Check that YOLO detects every scene object, with the right colour, over range.

Places the robot (no physics) at several distances and viewing angles around
each object in objects.yaml, renders the front camera at the Task 2 pipeline
resolution (320x240) in A's Task 2 scene, runs the Task 4
detector and reports detection + colour-grounding rates. Saves an annotated
contact sheet to results/detection_check.png.

    python check_detection.py
    python check_detection.py --scene path/to/A_scene.xml --objects path/to/objects.yaml
"""
from __future__ import annotations

import perception  # noqa: F401  (import torch before mujoco)

import argparse
import collections
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from perception import Detector, draw_detections, normalize_class
from sim_backend import TASK2_SCENE, SandboxSim  # static renders only

HERE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=str(TASK2_SCENE))
    ap.add_argument("--objects", default=str(HERE / "objects.yaml"))
    ap.add_argument("--dists", nargs="*", type=float,
                    default=[4.0, 3.0, 2.0, 1.5, 1.0, 0.8, 0.7, 0.6])
    ap.add_argument("--angles", nargs="*", type=float, default=[-40, -20, 0, 20, 40])
    args = ap.parse_args()

    sim = SandboxSim(Path(args.scene), width=320, height=240)
    det = Detector()
    objs = yaml.safe_load(Path(args.objects).read_text())["objects"]
    sheet = []
    print(f"{'object':14s} {'range':>6s}  detected  colour_ok")
    for o in objs:
        cls = normalize_class(o["class"])
        per_d = collections.defaultdict(lambda: [0, 0, 0])
        for d in args.dists:
            for off in args.angles:
                a = math.atan2(-o["y"], -o["x"]) + math.radians(off)
                x, y = o["x"] + d * math.cos(a), o["y"] + d * math.sin(a)
                img = sim.render_static(x, y, math.degrees(a) + 180)
                ds = sorted([q for q in det.detect(img) if q.cls == cls],
                            key=lambda q: abs(q.cx - img.shape[1] / 2))
                per_d[d][0] += 1
                if ds:
                    per_d[d][1] += 1
                    per_d[d][2] += ds[0].color == o["color"]
                if off == 0 and d in (args.dists[2], args.dists[-2]):
                    tile = draw_detections(img, ds, ds[0] if ds else None)
                    cv2.putText(tile, f"{o['name']} @ {d} m", (6, 232),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    sheet.append(tile)
        for d, (n, hit, ok) in per_d.items():
            print(f"{o['name']:14s} {d:5.1f}m  {hit}/{n}       {ok}/{max(hit, 1)}")
    rows = [np.hstack(sheet[i:i + 4] + [np.zeros_like(sheet[0])] * (4 - len(sheet[i:i + 4])))
            for i in range(0, len(sheet), 4)]
    out = HERE / "results" / "detection_check.png"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print("saved", out)


if __name__ == "__main__":
    main()
