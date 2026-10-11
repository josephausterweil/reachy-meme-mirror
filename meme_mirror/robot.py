"""Play a move on the real Reachy Mini from this computer (SDK remote client).

Only call this after the user explicitly asks for motion. Refuses while a robot
app (e.g. Face Greeter) holds the robot.

Usage: python -m meme_mirror.robot <move.json> [--sleep-after]
"""

import argparse
import json
import time

import requests
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMove
from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS, INIT_HEAD_POSE

DAEMON = "http://reachy-mini.local:8000"


class RobotBusy(RuntimeError):
    pass


def check_free(daemon=DAEMON):
    """Raise RobotBusy if the daemon isn't ready or an app is running."""
    status = requests.get(f"{daemon}/api/daemon/status", timeout=5).json()
    if status.get("state") != "running" or not status["backend_status"].get("ready"):
        raise RobotBusy(f"daemon not ready: {status.get('state')}")
    app = requests.get(f"{daemon}/api/apps/current-app-status", timeout=5).json()
    if app is not None:
        raise RobotBusy(f"an app is running: {app}")
    return status


def play(move_dict, sleep_after=False, daemon=DAEMON):
    """Wake if needed, glide to the move's first pose, play it, return to neutral."""
    status = check_free(daemon)
    was_asleep = status["backend_status"]["motor_control_mode"] == "disabled"
    move = RecordedMove(move_dict)
    with ReachyMini(connection_mode="network", media_backend="no_media", timeout=10) as mini:
        if was_asleep:
            mini.enable_motors()
            mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS_JOINT_POSITIONS, duration=2.0)
        head, antennas, body_yaw = move.evaluate(0.0)
        mini.goto_target(head=head, antennas=antennas, duration=0.8, body_yaw=body_yaw)
        mini.play_move(move, sound=False)
        mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS_JOINT_POSITIONS,
                         duration=1.0, body_yaw=0.0)
        time.sleep(0.3)
        if sleep_after:
            mini.goto_sleep()
            time.sleep(0.5)
            mini.disable_motors()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("move")
    ap.add_argument("--sleep-after", action="store_true")
    a = ap.parse_args()
    play(json.load(open(a.move)), sleep_after=a.sleep_after)
    print("done")
