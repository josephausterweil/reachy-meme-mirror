"""Play a move on the real Reachy Mini from this computer (SDK remote client).

Only call this after the user explicitly asks for motion. Refuses while a robot
app (e.g. Face Greeter) holds the robot.

The connection is kept open between plays (opening/closing costs ~0.7 s / ~0.9 s) and closed after
IDLE_CLOSE_S of inactivity.

Usage: python -m meme_mirror.robot <move.json> [--sleep-after]
"""

import argparse
import json
import threading
import time

import requests
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMove
from reachy_mini.reachy_mini import INIT_ANTENNAS_JOINT_POSITIONS, INIT_HEAD_POSE

from .config import robot_host

def daemon_url():
    """The robot daemon: localhost when we run on the robot as an app, else the configured host."""
    return "http://localhost:8000" if _attached else f"http://{robot_host()}:8000"
IDLE_CLOSE_S = 60.0
HOME_S = 0.8  # glide back to neutral after a move

_mini = None
_attached = False   # True when running as a Reachy Mini app: the app framework owns the connection
_last_used = 0.0
_conn_lock = threading.RLock()


def attach(mini):
    """Use the ReachyMini instance handed to a running app (meme_mirror installed on the robot)."""
    global _mini, _attached
    with _conn_lock:
        _mini, _attached = mini, True


def detach():
    global _mini, _attached
    with _conn_lock:
        _mini, _attached = None, False


def attached():
    return _attached


def play_sound_on_robot(path):
    """Play through the robot's own speaker (only when attached as an app with audio)."""
    if _attached and _mini is not None and path:
        _mini.media.play_sound(str(path))


class RobotBusy(RuntimeError):
    pass


def check_free(daemon=None):
    """Raise RobotBusy if the daemon isn't ready or an app is running."""
    daemon = daemon or daemon_url()
    status = requests.get(f"{daemon}/api/daemon/status", timeout=5).json()
    if status.get("state") != "running" or not status["backend_status"].get("ready"):
        raise RobotBusy(f"daemon not ready: {status.get('state')}")
    if not _attached:  # as the running app ourselves, the "an app is running" check would always trip
        app = requests.get(f"{daemon}/api/apps/current-app-status", timeout=5).json()
        if app is not None:
            raise RobotBusy(f"an app is running: {app}")
    return status


def _connection():
    """The kept-open client. A stale one is detected with a harmless query (no motion) and reopened."""
    global _mini, _last_used
    with _conn_lock:
        if _attached:
            return _mini
        if _mini is not None:
            try:
                _mini.get_current_head_pose()
            except Exception:
                close()
        if _mini is None:
            _mini = ReachyMini(host=robot_host(), connection_mode="network", media_backend="no_media", timeout=10)
            _mini.__enter__()
            threading.Thread(target=_idle_closer, daemon=True).start()
        _last_used = time.time()
        return _mini


def close():
    global _mini
    with _conn_lock:
        if _mini is not None and not _attached:
            try:
                _mini.__exit__(None, None, None)
            except Exception:
                pass
            finally:
                _mini = None


def _idle_closer():
    while True:
        time.sleep(5)
        with _conn_lock:
            if _mini is None:
                return
            if time.time() - _last_used > IDLE_CLOSE_S:
                close()
                return


def play(move_dict, sleep_after=False, daemon=None, on_start=None, return_home=True):
    """Wake if needed, glide to the move's first pose, play it, then (if return_home) glide back to neutral.
    `on_start()` is called right before the move itself starts (e.g. to start its sound).
    With return_home=False this returns as soon as the move ends; call return_to_neutral() afterwards."""
    status = check_free(daemon)
    was_asleep = status["backend_status"]["motor_control_mode"] == "disabled"
    move = RecordedMove(move_dict)
    mini = _connection()
    if was_asleep:
        mini.enable_motors()
        mini.goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS_JOINT_POSITIONS, duration=2.0)
    head, antennas, body_yaw = move.evaluate(0.0)
    mini.goto_target(head=head, antennas=antennas, duration=0.8, body_yaw=body_yaw)
    if on_start:
        on_start()
    mini.play_move(move, sound=False)
    if return_home or sleep_after:
        return_to_neutral()
    if sleep_after:
        go_to_sleep()


def return_to_neutral():
    _connection().goto_target(INIT_HEAD_POSE, antennas=INIT_ANTENNAS_JOINT_POSITIONS,
                              duration=HOME_S, body_yaw=0.0)


def go_to_sleep():
    mini = _connection()
    mini.goto_sleep()
    time.sleep(0.5)
    mini.disable_motors()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("move")
    ap.add_argument("--sleep-after", action="store_true")
    a = ap.parse_args()
    try:
        play(json.load(open(a.move)), sleep_after=a.sleep_after)
    finally:
        close()
    print("done")
