"""Terminal chat interface for Task 3."""

from llm_parser import parse_command, print_command
from executor import execute_actions, stop_execution

import json
import threading
import speech_recognition as sr

def listen_for_command():
    """Capture one spoken English command from the microphone."""

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("[STT] Listening...")

            recognizer.adjust_for_ambient_noise(
                source,
                duration=0.5,
            )

            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=10,
            )

        print("[STT] Processing...")

        text = recognizer.recognize_google(
            audio,
            language="en-SG",
        )

        print(f'[STT] "{text}"')

        return text

    except sr.WaitTimeoutError:
        print("[STT] No speech detected.")
        return None

    except sr.UnknownValueError:
        print("[STT] Could not understand the speech.")
        return None

    except sr.RequestError as exc:
        print(f"[STT] Speech service error: {exc}")
        return None

    except OSError as exc:
        print(f"[STT] Microphone error: {exc}")
        return None

def run_chat():
    """Run the Task 3 terminal chat interface."""

    print("EE4705 Robot Command Interface")
    print("Type an English command.")
    print("Type 'exit' to quit.\n")

    history = []

    execution_thread = None

    while True:
        user_text = input("You> ").strip()

        if user_text.lower() == "/voice":

            spoken_text = listen_for_command()

            if not spoken_text:
                print()
                continue

            user_text = spoken_text

        if user_text.lower() == "exit":
            print("Exiting.")
            break

        if user_text.lower() in {
            "stop",
            "stop now",
            "emergency stop",
        }:
            print("[CMD] actions=stop n=1")
            stop_execution()
            print()
            continue

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

            actions = result.get("actions", [])

            # --------------------------------------------------
            # STOP is handled immediately by the chat thread
            # --------------------------------------------------

            if any(
                action["action"] == "stop"
                for action in actions
            ):
                stop_execution()

            # --------------------------------------------------
            # All other missions run in a worker thread
            # --------------------------------------------------

            else:
                if (
                    execution_thread is not None
                    and execution_thread.is_alive()
                ):
                    print(
                        "Robot: I am still executing the current "
                        "mission. Type 'stop' to cancel it."
                    )

                else:
                    execution_thread = threading.Thread(
                        target=execute_actions,
                        args=(result,),
                        name="task3-executor",
                        daemon=True,
                    )

                    execution_thread.start()

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