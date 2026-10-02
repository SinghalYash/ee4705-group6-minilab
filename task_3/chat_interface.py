"""Terminal chat interface for Task 3."""

from llm_parser import parse_command, print_command
from executor import execute_actions

import json

import threading


def run_chat():
    """Run the Task 3 terminal chat interface."""

    print("EE4705 Robot Command Interface")
    print("Type an English command.")
    print("Type 'exit' to quit.\n")

    history = []

    while True:
        user_text = input("You> ").strip()

        if user_text.lower() == "exit":
            print("Exiting.")
            break

        result = parse_command(user_text, history)

        history.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": json.dumps(result),
            }
        )

        print_command(result)

        if result["accepted"]:
            execute_actions(result)

        print()

def start_chat_thread():
    """Start the terminal chat interface in a background thread."""

    thread = threading.Thread(
        target=run_chat,
        daemon=True,
    )

    thread.start()

    return thread

if __name__ == "__main__":
    run_chat()