"""Evaluation suite for the Task 3 LLM command parser."""

import csv
import os
import sys
import time

from llm_parser import parse_command
from ollama_parser import parse_command_ollama

PARSERS = {
    "openai": parse_command,
    "ollama": parse_command_ollama,
}


TEST_CASES = [
    # ---------- Basic ----------
    {
        "input": "walk forward for three seconds",
        "category": "basic",
        "expected": [
            {"action": "move", "vx": 0.8, "vy": 0.0,
             "wz": 0.0, "duration": 3.0}
        ],
    },
    {
        "input": "walk backward for two seconds",
        "category": "basic",
        "expected": [
            {"action": "move", "vx": -0.8, "vy": 0.0,
             "wz": 0.0, "duration": 2.0}
        ],
    },
    {
        "input": "turn left 90 degrees",
        "category": "basic",
        "expected": [
            {"action": "turn", "angle_deg": 90.0}
        ],
    },
    {
        "input": "turn right 45 degrees",
        "category": "basic",
        "expected": [
            {"action": "turn", "angle_deg": -45.0}
        ],
    },

    # ---------- Multi-step ----------
    {
        "input": "walk forward for three seconds, then turn back",
        "category": "multi-step",
        "expected": [
            {"action": "move", "vx": 0.8, "vy": 0.0,
             "wz": 0.0, "duration": 3.0},
            {"action": "turn", "angle_deg": 180.0},
        ],
    },
    {
        "input": "move forward for two seconds then turn left 90 degrees",
        "category": "multi-step",
        "expected": [
            {"action": "move", "vx": 0.8, "vy": 0.0,
             "wz": 0.0, "duration": 2.0},
            {"action": "turn", "angle_deg": 90.0},
        ],
    },
    {
        "input": "turn right 90 degrees then walk forward for four seconds",
        "category": "multi-step",
        "expected": [
            {"action": "turn", "angle_deg": -90.0},
            {"action": "move", "vx": 0.8, "vy": 0.0,
             "wz": 0.0, "duration": 4.0},
        ],
    },
    {
        "input": "walk backward for two seconds, turn around, then walk forward for one second",
        "category": "multi-step",
        "expected": [
            {"action": "move", "vx": -0.8, "vy": 0.0,
             "wz": 0.0, "duration": 2.0},
            {"action": "turn", "angle_deg": 180.0},
            {"action": "move", "vx": 0.8, "vy": 0.0,
             "wz": 0.0, "duration": 1.0},
        ],
    },

    # ---------- Paraphrases ----------
    {
        "input": "could you walk forwards for five seconds?",
        "category": "paraphrase",
        "expected": [
            {"action": "move", "vx": 0.8, "vy": 0.0,
             "wz": 0.0, "duration": 5.0}
        ],
    },
    {
        "input": "go straight ahead for three seconds",
        "category": "paraphrase",
        "expected": [
            {"action": "move", "vx": 0.8, "vy": 0.0,
             "wz": 0.0, "duration": 3.0}
        ],
    },
    {
        "input": "take a few steps backwards for two seconds",
        "category": "paraphrase",
        "expected": [
            {"action": "move", "vx": -0.8, "vy": 0.0,
             "wz": 0.0, "duration": 2.0}
        ],
    },
    {
        "input": "make a quarter turn to the left",
        "category": "paraphrase",
        "expected": [
            {"action": "turn", "angle_deg": 90.0}
        ],
    },
    {
        "input": "face the opposite direction",
        "category": "paraphrase",
        "expected": [
            {"action": "turn", "angle_deg": 180.0}
        ],
    },
    {
        "input": "slowly move forward for four seconds",
        "category": "paraphrase",
        "expected": [
            {"action": "move", "vx": 0.4, "vy": 0.0,
             "wz": 0.0, "duration": 4.0}
        ],
    },

    # ---------- Invalid / out-of-scope ----------
    {
        "input": "fly to the roof",
        "category": "invalid",
        "expected": None,
    },
    {
        "input": "jump over the wall",
        "category": "invalid",
        "expected": None,
    },
    {
        "input": "pick up the chair",
        "category": "invalid",
        "expected": None,
    },
    {
        "input": "",
        "category": "invalid",
        "expected": None,
    },
    {
        "input": "sing me a song while running",
        "category": "invalid",
        "expected": None,
    },
    {
        "input": "camina hacia adelante por tres segundos",
        "category": "invalid",
        "expected": None,
    },
]

def evaluate_result(result, expected):
    """Compare the parsed result against the expected command."""

    # Invalid command should be rejected
    if expected is None:
        return result["accepted"] is False

    # Valid command should be accepted
    if not result["accepted"]:
        return False

    actual = result["actions"]

    # Wrong number of actions
    if len(actual) != len(expected):
        return False

    for actual_action, expected_action in zip(actual, expected):

        # Wrong action type
        if actual_action["action"] != expected_action["action"]:
            return False

        # Check every parameter specified in expected_action
        for key, expected_value in expected_action.items():

            if key == "action":
                continue

            actual_value = actual_action.get(key)

            if actual_value is None:
                return False

            # Numeric comparison
            if isinstance(expected_value, (int, float)):
                if abs(actual_value - expected_value) > 1e-6:
                    return False

            # String comparison
            elif actual_value != expected_value:
                return False

    return True

def run_evaluation(service):
    if service not in PARSERS:
        raise ValueError(
            f"Unknown service '{service}'. "
            f"Choose from: {', '.join(PARSERS)}"
        )

    parser = PARSERS[service]

    results = []

    print("Task 3 Parser Evaluation")
    print("=" * 60)

    for index, test in enumerate(TEST_CASES, start=1):

        print(
            f"[{index:02d}/{len(TEST_CASES)}] "
            f"{test['input']!r}"
        )

        start = time.perf_counter()

        try:
            result = parser(test["input"])
            error = None

        except Exception as exc:
            result = None
            error = str(exc)

        latency = time.perf_counter() - start

        if result is not None:
            passed = evaluate_result(
                result,
                test["expected"],
            )

            actual_actions = [
                action["action"]
                for action in result["actions"]
            ]

        else:
            passed = False
            actual_actions = []

        print(
            f"Category : {test['category']}\n"
            f"Expected : {test['expected']}\n"
            f"Actual   : {actual_actions}\n"
            f"Latency  : {latency:.3f} s\n"
            f"Result   : {'PASS' if passed else 'FAIL'}"
        )

        if error:
            print(f"Error    : {error}")

        results.append(
            {
                "input": test["input"],
                "category": test["category"],
                "expected": test["expected"],
                "actual": actual_actions,
                "passed": passed,
                "latency": latency,
                "error": error,
            }
        )

    passed_count = sum(
        result["passed"]
        for result in results
    )

    accuracy = passed_count / len(results) * 100

    average_latency = (
        sum(result["latency"] for result in results)
        / len(results)
    )

    # --------------------------------------------------
    # Save detailed results to CSV
    # --------------------------------------------------

    results_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
    )

    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(
        results_dir,
        f"{service}_results.csv",
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "input",
                "category",
                "expected",
                "actual",
                "passed",
                "latency",
                "error",
            ],
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(
            f"Passed          : {passed_count}/{len(results)}"
        )

        print(
            f"Parsing accuracy: {accuracy:.1f}%"
        )

        print(
            f"Average latency : {average_latency:.3f} s"
        )

    


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python evaluation.py "
            "<openai|ollama>"
        )
        raise SystemExit(1)

    service = sys.argv[1].lower()

    run_evaluation(service)