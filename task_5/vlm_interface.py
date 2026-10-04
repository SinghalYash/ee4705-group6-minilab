"""OpenAI VLM interface for scene description and visual question answering."""

import base64
import io
import os

from PIL import Image
from openai import OpenAI


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)


def image_to_data_url(image):
    """Convert a PIL image to a JPEG data URL."""

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=85,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


def ask_image(image, question):
    """Ask the VLM a question about a PIL image."""

    image_url = image_to_data_url(image)

    print(f"[VLM] question={question}")

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are viewing the front camera image "
                            "from a simulated quadruped robot. "
                            "Answer using only information visible "
                            "in the image. Be concise.\n\n"
                            f"Question: {question}"
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "low",
                    },
                ],
            }
        ],
    )

    return response.output_text

def ask_robot(robot, question):
    """Ask a visual question about the robot's latest camera frame."""

    frame_data = robot.camera.get_latest_frame()

    if frame_data is None:
        return "The robot camera does not have a frame yet."

    rgb, timestamp = frame_data

    image = Image.fromarray(rgb)

    print(
        f"[VLM] camera frame t={timestamp:.2f}s"
    )

    return ask_image(
        image,
        question,
    )