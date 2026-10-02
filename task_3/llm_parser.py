import json
import os
from openai import OpenAI, BadRequestError
from command_schema import COMMAND_SCHEMA

from executor import execute_actions


# client = OpenAI()

client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
MODEL = "qwen3-vl-flash"


SYSTEM_PROMPT = """
You are a command parser for a simulated quadruped robot.

Convert the user's English instruction into structured robot actions.

AVAILABLE ACTIONS

1. move
   Fields:
   - vx: forward/backward velocity from -1.0 to 1.0
   - vy: left/right velocity from -1.0 to 1.0
   - wz: rotational velocity from -1.0 to 1.0
   - duration: duration in seconds

   Motion conventions:
   - forward: vx = +0.8
   - backward: vx = -0.8
   - move left: vy = +0.8
   - move right: vy = -0.8
   - use 0.4 magnitude when the user explicitly requests slow motion
   - otherwise use 0.8 as the default speed
   - unused velocity components are 0.0

2. turn
   Field:
   - angle_deg

   Turn conventions:
   - positive angle = left / counter-clockwise
   - negative angle = right / clockwise
   - "turn around", "turn back", or equivalent = 180 degrees

3. goto_object
   Fields:
   - class: object class
   - color: requested color, or null

4. stop

5. chat
   Field:
   - reply

RULES

- Preserve the order of multi-step commands.
- Interpret natural English paraphrases.
- Do not ask for speed when the user gives a clear movement direction.
  Use the default speed.
- If a movement duration is explicitly provided, use it.
- If essential information genuinely cannot be inferred safely, ask for clarification.
- Only accept commands the robot can perform.
- Reject impossible or unsafe commands.
- Reject empty commands.
- Reject non-English commands.
- Do not invent capabilities.
"""


def _call(messages, response_format):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format=response_format,
        extra_body={"enable_thinking": False},  # structured output isn't supported in Qwen thinking mode
    )


def parse_command(user_text, history=None):
    if history is None:
        history = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    try:
        response = _call(messages, {
            "type": "json_schema",
            "json_schema": {
                "name": "robot_command",
                "schema": COMMAND_SCHEMA,
                "strict": True,
            },
        })
    except BadRequestError:
        # Fallback: JSON mode + schema in the prompt (DashScope requires the word "JSON" in the prompt)
        messages[0] = {
            "role": "system",
            "content": SYSTEM_PROMPT
            + "\n\nOUTPUT FORMAT\nReply ONLY with a JSON object matching this JSON schema:\n"
            + json.dumps(COMMAND_SCHEMA),
        }
        response = _call(messages, {"type": "json_object"})

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]

    result = json.loads(raw)

    # Guard against schema drift in fallback mode
    result.setdefault("accepted", False)
    result.setdefault("actions", [])
    if not result["accepted"] and not result.get("reason"):
        result["reason"] = "invalid model output"

    return result

def print_command(result):
    """Print the parsed command in the format required by the MiniLab."""

    if not result["accepted"]:
        reason = result.get("reason") or "unknown"
        print(f"[CMD] rejected reason={reason}")
        return

    action_strings = []

    for action in result["actions"]:
        action_type = action["action"]

        if action_type == "move":
            action_strings.append(
                f"move("
                f"vx={action['vx']}, "
                f"vy={action['vy']}, "
                f"wz={action['wz']}, "
                f"{action['duration']} s)"
            )

        elif action_type == "turn":
            action_strings.append(
                f"turn({action['angle_deg']} deg)"
            )

        elif action_type == "goto_object":
            action_strings.append(
                f"goto_object("
                f"class={action['class']}, "
                f"color={action['color']})"
            )

        elif action_type == "stop":
            action_strings.append("stop")

        elif action_type == "chat":
            action_strings.append("chat")

    actions_text = ", ".join(action_strings)

    print(
        f"[CMD] actions={actions_text} "
        f"n={len(result['actions'])}"
    )


if __name__ == "__main__":
    command = input("Enter command: ")

    result = parse_command(command)

    print(json.dumps(result, indent=2))
    print_command(result)

    execute_actions(result)