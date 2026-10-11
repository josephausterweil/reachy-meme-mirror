"""ExpressionSpec -> 50 Hz trajectory -> Pollen/Marionette move JSON (playable by RecordedMove/play_move)."""

import numpy as np
from reachy_mini.utils import create_head_pose

from . import catalog as catalog_mod
from .spec import NEUTRAL, ExpressionSpec, HoldSegment, KeyframeSegment, LibrarySegment

FPS = 50
DT = 1.0 / FPS
CHANNELS = list(NEUTRAL)  # pitch roll yaw z ear_r ear_l body_yaw x y
GLUE_S = 0.5  # blend into a library clip / back to neutral at the end


def _ease(n, kind):
    s = np.arange(1, n + 1) / n
    if kind == "linear":
        return s
    if kind == "snap":  # fast start, soft landing
        return 1 - (1 - s) ** 3
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)  # minimum-jerk


def _library_channels(move_id, catalog):
    """Catalog move -> dict of channel arrays resampled to FPS (ears as droop)."""
    ch = catalog_mod.load_channels(catalog[move_id]["file"])
    t = ch["t"] - ch["t"][0]
    tt = np.arange(0, t[-1], DT)
    out = {k: np.interp(tt, t, ch[k]) for k in ("pitch", "roll", "yaw", "z", "body_yaw", "x", "y")}
    out["ear_r"] = np.interp(tt, t, -ch["ant_r"])
    out["ear_l"] = np.interp(tt, t, ch["ant_l"])
    return out


def _append(tracks, seg):
    for k in CHANNELS:
        tracks[k].extend(seg[k])


def _glide(cur, target, seconds, ease="smooth"):
    n = max(1, int(round(seconds * FPS)))
    w = _ease(n, ease)
    return {k: list(cur[k] + w * (target[k] - cur[k])) for k in CHANNELS}


def compile_spec(spec: ExpressionSpec, catalog=None, glue=True, speed=1.0, amplitude=1.0):
    """Return dict of channel arrays at FPS. `speed` > 1 plays faster; `amplitude` scales away from neutral."""
    catalog = catalog if catalog is not None else catalog_mod.load()
    tracks = {k: [NEUTRAL[k]] for k in CHANNELS}
    cur = dict(NEUTRAL)

    for seg in spec.segments:
        if isinstance(seg, HoldSegment):
            n = max(1, int(round(seg.seconds * FPS)))
            _append(tracks, {k: [cur[k]] * n for k in CHANNELS})
        elif isinstance(seg, KeyframeSegment):
            for kf in seg.keyframes:
                target = dict(cur)
                for k in ("pitch", "roll", "yaw", "z", "ear_r", "ear_l", "body_yaw"):
                    v = getattr(kf, k)
                    if v is not None:
                        target[k] = v
                if kf.ears is not None:
                    target["ear_r"] = target["ear_l"] = kf.ears
                _append(tracks, _glide(cur, target, kf.dt, kf.ease))
                cur = target
        elif isinstance(seg, LibrarySegment):
            lib = _library_channels(seg.move_id, catalog)
            n = len(lib["pitch"])
            a = int(seg.start_s * FPS)
            b = n if seg.end_s is None else min(n, int(seg.end_s * FPS))
            src = np.arange(a, b)
            tt = np.arange(0, len(src) - 1, 1 / seg.time_scale)  # resample for time_scale
            clip = {k: np.interp(tt, np.arange(len(src)), lib[k][a:b]) for k in CHANNELS}
            clip = {k: NEUTRAL[k] + seg.amplitude * (clip[k] - NEUTRAL[k]) for k in CHANNELS}
            first = {k: clip[k][0] for k in CHANNELS}
            if glue and any(abs(first[k] - cur[k]) > 0.5 for k in CHANNELS):
                _append(tracks, _glide(cur, first, GLUE_S))
            _append(tracks, {k: list(clip[k]) for k in CHANNELS})
            cur = {k: clip[k][-1] for k in CHANNELS}

    if glue:
        _append(tracks, _glide(cur, NEUTRAL, GLUE_S))
    out = {k: np.asarray(v, float) for k, v in tracks.items()}
    if not glue:  # drop the seed neutral frame so a bare library clip round-trips exactly
        out = {k: v[1:] for k, v in out.items()}
    t = np.arange(len(out["pitch"])) * DT

    for ov in spec.overlays:
        end = t[-1] if ov.end_s is None else ov.end_s
        mask = (t >= ov.start_s) & (t <= end)
        tl = t[mask] - ov.start_s
        ramp = np.clip(np.minimum(tl, (end - ov.start_s) - tl) / 0.25, 0, 1)  # 0.25 s fade in/out
        if ov.shape == "sine":
            wave = np.sin(2 * np.pi * ov.freq_hz * tl)
        else:  # tremble: smoothed noise at ~freq_hz
            rng = np.random.default_rng(0)
            knots = rng.uniform(-1, 1, int(len(tl) * ov.freq_hz * DT) + 2)
            wave = np.interp(tl * ov.freq_hz, np.arange(len(knots)), knots)
        delta = ov.amplitude * ramp * wave
        if ov.channel == "ears":
            out["ear_r"][mask] += delta
            out["ear_l"][mask] += -delta if ov.antiphase_ears else delta
        else:
            out[ov.channel][mask] += delta

    if amplitude != 1.0:
        out = {k: NEUTRAL[k] + amplitude * (v - NEUTRAL[k]) for k, v in out.items()}
    if speed != 1.0:
        t_new = np.arange(0, t[-1], DT * speed)
        out = {k: np.interp(t_new, t, v) for k, v in out.items()}
    return out


def to_move(tracks, description=""):
    """Channel arrays -> Pollen/Marionette move dict."""
    n = len(tracks["pitch"])
    data = []
    for i in range(n):
        head = create_head_pose(x=tracks["x"][i] / 1000, y=tracks["y"][i] / 1000, z=tracks["z"][i] / 1000,
                                roll=tracks["roll"][i], pitch=tracks["pitch"][i], yaw=tracks["yaw"][i],
                                degrees=True, mm=False)
        data.append(dict(head=np.round(head, 6).tolist(),
                         antennas=[round(float(np.deg2rad(-tracks["ear_r"][i])), 6),
                                   round(float(np.deg2rad(tracks["ear_l"][i])), 6)],
                         body_yaw=round(float(np.deg2rad(tracks["body_yaw"][i])), 6),
                         check_collision=False))
    return dict(description=description, time=[round(i * DT, 4) for i in range(n)], set_target_data=data)
