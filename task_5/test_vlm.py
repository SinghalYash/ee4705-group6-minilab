"""Standalone Task 5 VLM test."""

from pathlib import Path

from PIL import Image

from vlm_interface import ask_image


ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = ROOT / "task_2" / "front_camera_test.png"


def main():

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Camera test image not found: {IMAGE_PATH}"
        )

    image = Image.open(IMAGE_PATH).convert("RGB")

    question = input(
        "Visual question: "
    ).strip()

    if not question:
        question = "What can you see?"

    answer = ask_image(
        image,
        question,
    )

    print(f"Robot: {answer}")


if __name__ == "__main__":
    main()