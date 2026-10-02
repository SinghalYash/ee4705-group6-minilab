"""Ollama/Qwen parser for Task 3 evaluation."""

import json

from ollama import chat

from command_schema import COMMAND_SCHEMA
from llm_parser import SYSTEM_PROMPT


def parse_command_ollama(user_text, history=None):
    """Parse an English robot command using local Qwen3 via Ollama."""

    if history is None:
        history = []

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    response = chat(
        model="qwen3:4b",
        messages=messages,
        format=COMMAND_SCHEMA,
        options={
            "temperature": 0,
        },
    )

    return json.loads(response.message.content)


if __name__ == "__main__":
    command = input("Enter command: ")

    result = parse_command_ollama(command)

    print(json.dumps(result, indent=2))