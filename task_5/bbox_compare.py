"""Compare YOLO and VLM object grounding on the same camera frame."""

import json

from task_4.perception import Detector
from task_5.vlm_interface import image_to_data_url

from PIL import Image
from openai import OpenAI

from pathlib import Path
import cv2


client = OpenAI()
_detector = None


def get_detector():
    global _detector

    if _detector is None:
        _detector = Detector()

    return _detector


def iou(box_a, box_b):
    """Intersection-over-Union for two [x1, y1, x2, y2] boxes."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def compare_bbox(robot, target_class, target_color=None):
    """Compare YOLO and VLM grounding on one live camera frame."""

    frame_data = robot.camera.get_latest_frame()

    if frame_data is None:
        print("[BBOX] No camera frame available.")
        return

    rgb, timestamp = frame_data
    height, width = rgb.shape[:2]

    print(
        f"[BBOX] frame t={timestamp:.2f}s "
        f"target={target_color or ''} {target_class}".strip()
    )

    # --------------------------------------------------
    # YOLO
    # --------------------------------------------------

    detector = get_detector()

    detections = detector.detect(
        rgb,
        classes=[target_class],
    )

    candidates = [
        d for d in detections
        if (
            target_color is None
            or d.color == target_color
        )
    ]

    if not candidates:
        print("[YOLO] target not detected")
        return

    # Use highest-confidence matching detection.
    yolo_det = max(
        candidates,
        key=lambda d: d.conf,
    )

    yolo_box = list(yolo_det.bbox)

    print(
        f"[YOLO] bbox={yolo_box} "
        f"conf={yolo_det.conf:.2f} "
        f"color={yolo_det.color}"
    )

    # --------------------------------------------------
    # VLM
    # --------------------------------------------------

    image = Image.fromarray(rgb)
    image_url = image_to_data_url(image)

    target_text = (
        f"{target_color} {target_class}"
        if target_color
        else target_class
    )

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Locate the {target_text} in this image. "
                            "Return one bounding box around the visible "
                            "object using coordinates normalized from "
                            "0 to 1000, where (0,0) is the top-left and "
                            "(1000,1000) is the bottom-right. "
                            "Return JSON only with fields: "
                            "found, x1, y1, x2, y2."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "high",
                    },
                ],
            }
        ],
    )

    raw = response.output_text.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[VLM] invalid JSON: {raw}")
        return

    if not result.get("found", False):
        print("[VLM] target not detected")
        return

    # Convert normalized 0-1000 coordinates to pixels.
    vlm_box = [
        int(result["x1"] / 1000 * width),
        int(result["y1"] / 1000 * height),
        int(result["x2"] / 1000 * width),
        int(result["y2"] / 1000 * height),
    ]

    print(f"[VLM]  bbox={vlm_box}")

    score = iou(
        yolo_box,
        vlm_box,
    )

    print(f"[BBOX] IoU={score:.3f}")

    # --------------------------------------------------
    # SAVE VISUAL COMPARISON
    # --------------------------------------------------

    # OpenCV uses BGR.
    annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # YOLO box
    yx1, yy1, yx2, yy2 = yolo_box
    cv2.rectangle(
        annotated,
        (yx1, yy1),
        (yx2, yy2),
        (255, 255, 255),
        2,
    )

    cv2.putText(
        annotated,
        "YOLO",
        (yx1, max(15, yy1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    # VLM box
    vx1, vy1, vx2, vy2 = vlm_box
    cv2.rectangle(
        annotated,
        (vx1, vy1),
        (vx2, vy2),
        (0, 255, 255),
        2,
    )

    cv2.putText(
        annotated,
        "VLM",
        (vx1, min(height - 5, vy2 + 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
    )

    cv2.putText(
        annotated,
        f"IoU={score:.3f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    evidence_dir = (
        Path(__file__).resolve().parent
        / "evidence"
    )

    evidence_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        evidence_dir
        / "bbox_comparison.png"
    )

    cv2.imwrite(
        str(output_path),
        annotated,
    )

    print(
        f"[BBOX] comparison image saved="
        f"{output_path}"
    )