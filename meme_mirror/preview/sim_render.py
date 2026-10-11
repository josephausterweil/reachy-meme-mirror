"""Offscreen MuJoCo render of a trajectory, mirroring the SDK's MuJoCo backend.

The control path is the same as reachy_mini/daemon/backend/mujoco/backend.py:
head pose + body yaw -> AnalyticalKinematics.ik -> ctrl[:7], antennas -> ctrl[-2:]
(negated), physics at 500 Hz, control at 50 Hz. Frames are rendered at `fps`.

Usage: python -m meme_mirror.preview.sim_render <move.json> <out.mp4>   (also writes <out>_strip.png)
"""

import json
import os
import sys
from importlib.resources import files

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import reachy_mini  # noqa: E402
from reachy_mini.kinematics import AnalyticalKinematics  # noqa: E402
from reachy_mini.reachy_mini import (  # noqa: E402
    INIT_ANTENNAS_JOINT_POSITIONS,
    INIT_HEAD_POSE,
    SLEEP_ANTENNAS_JOINT_POSITIONS,
    SLEEP_HEAD_JOINT_POSITIONS,
)

PHYSICS_DT = 0.002
CONTROL_DECIMATION = 10  # 50 Hz control
SETTLE_S = 0.6  # glide from rest to the first pose before frames count


def load_move(path):
    """Return (times, heads[N,4,4], antennas[N,2], body_yaw[N]) from a Pollen-format move JSON."""
    d = json.load(open(path))
    data = d["set_target_data"]
    return (
        np.asarray(d["time"], float),
        np.asarray([s["head"] for s in data], float),
        np.asarray([s["antennas"] for s in data], float),
        np.asarray([s.get("body_yaw", 0.0) for s in data], float),
    )


class SimRenderer:
    def __init__(self, width=480, height=480, camera="studio_close"):
        root = str(files(reachy_mini).joinpath("descriptions/reachy_mini/mjcf/"))
        self.model = mujoco.MjModel.from_xml_path(f"{root}/scenes/empty.xml")
        self.model.opt.timestep = PHYSICS_DT
        self.data = mujoco.MjData(self.model)
        self.kin = AnalyticalKinematics()
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera = camera
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                 for i in range(self.model.nu)]
        self.qpos_addr = [self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]

    def _reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # MuJoCo antenna order is [left, right]; robot convention is [right, left]
        sleep = SLEEP_HEAD_JOINT_POSITIONS + [SLEEP_ANTENNAS_JOINT_POSITIONS[1],
                                              SLEEP_ANTENNAS_JOINT_POSITIONS[0]]
        self.data.qpos[self.qpos_addr] = sleep
        self.data.ctrl[:] = sleep
        mujoco.mj_forward(self.model, self.data)
        self._set_target(INIT_HEAD_POSE, np.array(INIT_ANTENNAS_JOINT_POSITIONS), 0.0)
        for _ in range(int(1.5 / PHYSICS_DT)):
            mujoco.mj_step(self.model, self.data)

    def _set_target(self, head, antennas, body_yaw):
        joints = self.kin.ik(head, body_yaw=body_yaw)
        if joints is not None and not np.any(np.isnan(joints)):
            self.data.ctrl[:7] = joints
        self.data.ctrl[-2:] = -np.asarray(antennas)

    def render(self, times, heads, antennas, body_yaw, out_path=None, fps=25):
        """Render the trajectory; write a video (mp4/gif by extension) if out_path. Returns the frames."""
        self._reset()
        n_settle = int(SETTLE_S / PHYSICS_DT)
        steps = n_settle + int(times[-1] / PHYSICS_DT)
        frame_every = int(round(1 / (fps * PHYSICS_DT)))
        frames = []
        for step in range(steps):
            if step % CONTROL_DECIMATION == 0:
                t = max(0.0, (step - n_settle) * PHYSICS_DT)
                i = min(np.searchsorted(times, t), len(times) - 1)
                self._set_target(heads[i], antennas[i], body_yaw[i])
            mujoco.mj_step(self.model, self.data)
            if step >= n_settle and (step - n_settle) % frame_every == 0:
                self.renderer.update_scene(self.data, camera=self.camera)
                frames.append(self.renderer.render().copy())
        if out_path:
            imageio.mimsave(out_path, frames, fps=fps)
        return frames


def strip(frames, n=6, crop=(60, 420)):
    """n evenly spaced frames, cropped around the robot, side by side."""
    a, b = crop
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return np.concatenate([frames[i][a:b, a:b] for i in idx], axis=1)


if __name__ == "__main__":
    out = sys.argv[2]
    frames = SimRenderer().render(*load_move(sys.argv[1]), out)
    imageio.imwrite(os.path.splitext(out)[0] + "_strip.png", strip(frames))
    print(f"wrote {len(frames)} frames to {out} (+ _strip.png)")
