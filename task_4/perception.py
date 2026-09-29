"""Task 4 perception: YOLO detection + HSV colour grounding (Student C).

Detector.detect(rgb) -> list[Detection]
    Runs a COCO-trained Ultralytics YOLO model on one onboard-camera frame and
    attaches a colour label to each box, computed from the pixels inside it.

Colour grounding
----------------
YOLO knows classes, not colours, so colour is decided from the bbox pixels:
  1. crop the box, shrunk by a margin to drop edge background;
  2. convert to HSV and keep only "chromatic" pixels (S and V above floors),
     which removes grey floor, white highlights, dark legs and shadows
     (note: the default dark-blue skybox has S~147, V~113 and still passes,
     so gaps in a chair backrest add a few "blue" votes; the object's own
     colour dominates in practice, and temporal voting absorbs the rest);
  3. vote those pixels into named hue bins and take the winning bin.
Bin voting is used instead of a plain median hue because red sits on both
ends of the hue circle (0 and 180 in OpenCV): the median of a red object
split across the wrap lands in cyan/green. If too few pixels are chromatic,
the box is labelled white / grey / black from its median brightness.
"""
from __future__ import annotations

# Import torch/torchvision before mujoco/onnxruntime: on some Linux setups a
# lazy torchvision import after MuJoCo's GL context is created segfaults.
import torch  # noqa: F401
import torchvision  # noqa: F401
import torchvision.ops  # noqa: F401

from dataclasses import dataclass, field
import os

import cv2
import numpy as np
from ultralytics import YOLO

# OpenCV hue is 0..179. Red wraps around, so it has two ranges.
HUE_BINS = {
    "red": [(0, 8), (165, 180)],
    "orange": [(8, 20)],
    "yellow": [(20, 35)],
    "green": [(35, 85)],
    "cyan": [(85, 100)],
    "blue": [(100, 130)],
    "purple": [(130, 150)],
    "pink": [(150, 165)],
}
COLOR_NAMES = list(HUE_BINS) + ["white", "grey", "black"]


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: tuple  # (x1, y1, x2, y2) pixels, ints
    color: str
    color_score: float  # fraction of chromatic votes for the winning colour
    votes: dict = field(default_factory=dict)

    @property
    def cx(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    def log_line(self) -> str:
        b = ",".join(str(int(v)) for v in self.bbox)
        return (f"[DETECT] class={self.cls.replace(' ', '_')} color={self.color} "
                f"conf={self.conf:.2f} bbox=[{b}]")


def classify_color(rgb_crop: np.ndarray, s_min=110, v_min=60, min_chroma_frac=0.12):
    """Return (colour_name, score, votes) for an RGB crop (uint8, HxWx3)."""
    if rgb_crop.size == 0:
        return "unknown", 0.0, {}
    hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    chroma = (s >= s_min) & (v >= v_min)
    n_total = h.size
    n_chroma = int(chroma.sum())
    if n_chroma < max(20, min_chroma_frac * n_total):
        med_v = float(np.median(v))
        name = "white" if med_v > 170 else "black" if med_v < 60 else "grey"
        return name, 1.0 - n_chroma / n_total, {}
    hues = h[chroma]
    votes = {}
    for name, ranges in HUE_BINS.items():
        votes[name] = int(sum(((hues >= lo) & (hues < hi)).sum() for lo, hi in ranges))
    best = max(votes, key=votes.get)
    return best, votes[best] / n_chroma, votes


class Detector:
    """Thin wrapper around Ultralytics YOLO with colour grounding."""

    def __init__(self, weights: str | None = None, conf: float = 0.25,
                 imgsz: int = 640, margin: float = 0.12):
        weights = weights or os.environ.get("TASK4_YOLO_WEIGHTS", "yolo11n.pt")
        self.model = YOLO(weights)
        self.names = self.model.names
        self.conf = conf
        self.imgsz = imgsz
        self.margin = margin

    def detect(self, rgb: np.ndarray, classes: list[str] | None = None) -> list[Detection]:
        """rgb: HxWx3 uint8 in RGB order (as MuJoCo renders it)."""
        # Ultralytics treats raw numpy arrays as BGR (OpenCV convention).
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        cls_ids = None
        if classes:
            lookup = {v: k for k, v in self.names.items()}
            cls_ids = [lookup[c] for c in classes if c in lookup]
        res = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz,
                                 classes=cls_ids, verbose=False)[0]
        out = []
        H, W = rgb.shape[:2]
        for xyxy, c, s in zip(res.boxes.xyxy.cpu().numpy(),
                              res.boxes.cls.cpu().numpy(),
                              res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            mx, my = self.margin * (x2 - x1), self.margin * (y2 - y1)
            crop = rgb[int(max(0, y1 + my)):int(min(H, y2 - my)),
                       int(max(0, x1 + mx)):int(min(W, x2 - mx))]
            color, score, votes = classify_color(crop)
            out.append(Detection(
                cls=self.names[int(c)], conf=float(s),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                color=color, color_score=score, votes=votes,
            ))
        return out


def normalize_class(name: str) -> str:
    """Map parser output to COCO names ('sports_ball' / 'ball' -> 'sports ball')."""
    n = name.strip().lower().replace("_", " ")
    aliases = {"ball": "sports ball", "stopsign": "stop sign", "sofa": "couch",
               "plant": "potted plant", "table": "dining table", "tv monitor": "tv"}
    return aliases.get(n, n)


def draw_detections(rgb: np.ndarray, dets: list[Detection], target=None) -> np.ndarray:
    """Return a BGR image with boxes drawn (for saving evidence frames)."""
    img = np.ascontiguousarray(rgb[..., ::-1]).copy()
    for d in dets:
        is_t = target is not None and d is target
        col = (0, 255, 255) if is_t else (255, 255, 255)
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if is_t else 1)
        cv2.putText(img, f"{d.cls} {d.color} {d.conf:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return img
