"""
Task 2(iii): Early YOLO verification.

Runs YOLO on a saved onboard-camera frame to verify that scene
objects are recognizable before integrating autonomous navigation.

Student A contribution.
"""

from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_ROOT / "task_2" / "stop_sign_test.png"
OUTPUT_DIR = PROJECT_ROOT / "task_2" / "yolo_test"


def main():
    print(f"[YOLO TEST] source={IMAGE_PATH}")
    model = YOLO("yolo11n.pt")

    results = model.predict(
        source=str(IMAGE_PATH),
        conf=0.25,
        save=True,
        project=str(OUTPUT_DIR),
        name="front_camera",
    )

    print("\n[YOLO TEST] detections:")

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]

            xyxy = box.xyxy[0].tolist()

            print(
                f"  class={class_name:<15} "
                f"conf={confidence:.3f} "
                f"bbox={[round(v) for v in xyxy]}"
            )


if __name__ == "__main__":
    main()