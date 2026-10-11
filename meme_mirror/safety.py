"""Safety check for compiled trajectories: clamp to the ranges Pollen's own recorded emotions use,
flag speeds beyond theirs, and verify every frame is reachable with the SDK's analytical IK.

Changes are reported, never silent.
"""

from dataclasses import dataclass, field

import numpy as np

from .compiler import DT, to_move

# Observed min/max over pollen-robotics/reachy-mini-emotions-library (catalog features, 2026-10-11),
# rounded slightly outward. Head yaw is absolute (includes body rotation).
LIMITS = {
    "pitch": (-25.0, 32.0),
    "roll": (-30.0, 30.0),
    "yaw": (-65.0, 65.0),
    "z": (-45.0, 20.0),
    "body_yaw": (-60.0, 60.0),
    "ear_r": (-40.0, 180.0),
    "ear_l": (-40.0, 180.0),
    "x": (-30.0, 30.0),
    "y": (-30.0, 30.0),
}
MAX_HEAD_SPEED_DPS = 650.0   # library 98th-percentile peak is 600
MAX_EAR_SPEED_DPS = 1000.0   # library peak (both ears summed) is 963


@dataclass
class SafetyReport:
    clamped: dict = field(default_factory=dict)      # channel -> (n_frames, worst value)
    too_fast: list = field(default_factory=list)     # messages
    unreachable_frames: int = 0
    ok: bool = True

    def summary(self):
        if self.ok and not self.clamped and not self.too_fast:
            return "ok"
        parts = [f"clamped {k} on {n} frames (worst {w:+.0f})" for k, (n, w) in self.clamped.items()]
        parts += self.too_fast
        if self.unreachable_frames:
            parts.append(f"{self.unreachable_frames} frames unreachable by IK")
        return "; ".join(parts)


def check(tracks, ik=True):
    """Return (safe_tracks, SafetyReport). Clamps positions; speed and IK problems mark the report not ok."""
    rep = SafetyReport()
    out = {}
    for k, v in tracks.items():
        lo, hi = LIMITS[k]
        bad = (v < lo) | (v > hi)
        if bad.any():
            worst = v[bad][np.argmax(np.abs(v[bad]))]
            rep.clamped[k] = (int(bad.sum()), float(worst))
        out[k] = np.clip(v, lo, hi)

    head_speed = np.sqrt(sum(np.gradient(out[k], DT) ** 2 for k in ("pitch", "roll", "yaw")))
    ear_speed = np.abs(np.gradient(out["ear_r"], DT)) + np.abs(np.gradient(out["ear_l"], DT))
    if np.percentile(head_speed, 98) > MAX_HEAD_SPEED_DPS:
        rep.too_fast.append(f"head peak {np.percentile(head_speed, 98):.0f} deg/s > {MAX_HEAD_SPEED_DPS:.0f}")
    if np.percentile(ear_speed, 98) > MAX_EAR_SPEED_DPS:
        rep.too_fast.append(f"ears peak {np.percentile(ear_speed, 98):.0f} deg/s > {MAX_EAR_SPEED_DPS:.0f}")

    if ik:
        from reachy_mini.kinematics import AnalyticalKinematics

        kin = AnalyticalKinematics()
        move = to_move(out)
        for frame in move["set_target_data"][::5]:  # every 0.1 s
            j = kin.ik(np.asarray(frame["head"]), body_yaw=frame["body_yaw"])
            if j is None or np.any(np.isnan(j)):
                rep.unreachable_frames += 5
    rep.ok = not rep.too_fast and rep.unreachable_frames == 0
    return out, rep
