"""Task 3 command executor.

Executes structured commands produced by the LLM parser.
The temporary motion functions will later be replaced by
Task 2's actual motion-skills API.
"""

import time


# ---------------------------------------------------------
# Temporary Task 2 motion functions
# ---------------------------------------------------------

def move(vx, vy, wz, duration):
    """Temporary move function until Task 2 API is available."""

    print(
        f"    [SIM] moving "
        f"vx={vx} vy={vy} wz={wz} "
        f"for {duration} s"
    )

    time.sleep(duration)


def turn(angle_deg):
    """Temporary turn function until Task 2 API is available."""

    print(f"    [SIM] turning {angle_deg} deg")

    # Temporary delay only.
    time.sleep(1)


# ---------------------------------------------------------
# Task 3 executor
# ---------------------------------------------------------

def execute_actions(result):
    """Execute all accepted actions sequentially."""

    if not result["accepted"]:
        return

    actions = result["actions"]
    start_time = time.perf_counter()

    for index, action in enumerate(actions, start=1):

        action_type = action["action"]

        if action_type == "move":

            print(
                f"[EXEC] action={index}/{len(actions)} "
                f"move "
                f"vx={action['vx']} "
                f"vy={action['vy']} "
                f"wz={action['wz']} "
                f"t={action['duration']} s"
            )

            move(
                action["vx"],
                action["vy"],
                action["wz"],
                action["duration"],
            )

        elif action_type == "turn":

            print(
                f"[EXEC] action={index}/{len(actions)} "
                f"turn angle={action['angle_deg']} deg"
            )

            turn(action["angle_deg"])

    elapsed = time.perf_counter() - start_time

    print(
        f"[DONE] actions={len(actions)} "
        f"t={elapsed:.1f} s"
    )