"""Lightweight preview without MuJoCo (e.g. on the robot itself): a strip of simple drawings.

Each of the 6 frames shows the robot from the FRONT (left: roll tilt, yaw turn, height, both antennas) and from the
SIDE (right: pitch, i.e. looking up/down, and height). Good enough for the blind judge; people get the 3D viewer.
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (31, 45, 68)
BODY = (235, 235, 240)
DARK = (25, 25, 30)
ANT = (20, 20, 24)
CELL = 170
CELL_H = 215
BASE = 195


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _rot(pts, ang, cx, cy):
    c, s = math.cos(ang), math.sin(ang)
    return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in pts]


def _front(d, ox, f):
    """Front view: body, head rotated by roll, shifted by yaw (turn) and z (height), antennas by droop."""
    cx, base = ox + CELL / 2, BASE
    d.rounded_rectangle([cx - 34, base - 62, cx + 34, base], radius=26, fill=BODY)        # body
    body_shift = f["body_yaw"] * 0.25
    hx = cx + (f["yaw"] - f["body_yaw"]) * 0.35 + body_shift * 0.2
    hy = base - 92 - f["z"] * 0.9
    roll = math.radians(-f["roll"])
    w = 66 * max(0.55, math.cos(math.radians(f["yaw"])))                                 # turning narrows the face
    head = [(hx - w / 2, hy - 24), (hx + w / 2, hy - 24), (hx + w / 2, hy + 24), (hx - w / 2, hy + 24)]
    d.polygon(_rot(head, roll, hx, hy), fill=BODY)
    look = max(-1, min(1, f["pitch"] / 25))                                               # + = looking down
    for ex in (-0.22, 0.22):
        ex_x = hx + ex * w + (f["yaw"] / 60) * 6
        ey = hy + look * 8
        (px, py), = _rot([(ex_x, ey)], roll, hx, hy)
        d.ellipse([px - 8, py - 8, px + 8, py + 8], fill=DARK)
    for side, droop in ((-1, f["ear_r"]), (1, f["ear_l"])):                               # robot's right is viewer's left
        (ax, ay), = _rot([(hx + side * w * 0.32, hy - 24)], roll, hx, hy)
        ang = math.radians(droop) * side + roll                                           # 0 = straight up, + = outward/down
        ex, ey = ax + 52 * math.sin(ang), ay - 52 * math.cos(ang)
        d.line([ax, ay, ex, ey], fill=ANT, width=3)


def _side(d, ox, f):
    """Side view (robot facing right): head pitched down for + pitch, raised by z."""
    cx, base = ox + CELL / 2, BASE
    d.rounded_rectangle([cx - 26, base - 62, cx + 26, base], radius=22, fill=BODY)
    hx, hy = cx, base - 92 - f["z"] * 0.9
    ang = math.radians(f["pitch"])
    head = [(hx - 26, hy - 24), (hx + 26, hy - 24), (hx + 26, hy + 24), (hx - 26, hy + 24)]
    d.polygon(_rot(head, ang, hx, hy), fill=BODY)
    (lx, ly), = _rot([(hx + 26, hy)], ang, hx, hy)                                        # the face / lens
    d.ellipse([lx - 5, ly - 9, lx + 5, ly + 9], fill=DARK)
    (ax, ay), = _rot([(hx, hy - 24)], ang, hx, hy)
    droop = (f["ear_r"] + f["ear_l"]) / 2
    a2 = ang + math.radians(droop) * 0.6
    d.line([ax, ay, ax + 46 * math.sin(a2), ay - 46 * math.cos(a2)], fill=ANT, width=3)


def frame_values(tracks, i):
    return {k: float(tracks[k][i]) for k in ("pitch", "roll", "yaw", "z", "ear_r", "ear_l", "body_yaw")}


def strip(tracks, n=6, fps=50):
    """Channel arrays (as from compiler / move_to_tracks) -> PIL image of n evenly spaced frames."""
    T = len(tracks["pitch"])
    idx = np.linspace(0, T - 1, n).round().astype(int)
    img = Image.new("RGB", (n * 2 * CELL, CELL_H + 22), BG)
    d = ImageDraw.Draw(img)
    font = _font(13)
    for k, i in enumerate(idx):
        ox = k * 2 * CELL
        f = frame_values(tracks, i)
        _front(d, ox, f)
        _side(d, ox + CELL, f)
        d.line([ox + 2 * CELL - 1, 0, ox + 2 * CELL - 1, CELL_H + 22], fill=(60, 75, 100), width=2)
        d.text((ox + 8, CELL_H + 3), f"{k + 1}. t={i / fps:.1f}s   front | side", font=font, fill=(200, 210, 230))
    return img


def thumbnail(tracks, cols=3, rows=2, fps=50):
    """Card thumbnail: a near-square grid of FRONT views at evenly spaced times."""
    n = cols * rows
    T = len(tracks["pitch"])
    idx = np.linspace(0, T - 1, n).round().astype(int)
    img = Image.new("RGB", (cols * CELL, rows * (CELL_H + 18)), BG)
    d = ImageDraw.Draw(img)
    font = _font(13)
    for k, i in enumerate(idx):
        ox, oy = (k % cols) * CELL, (k // cols) * (CELL_H + 18)
        cell = Image.new("RGB", (CELL, CELL_H), BG)
        _front(ImageDraw.Draw(cell), 0, frame_values(tracks, i))
        img.paste(cell, (ox, oy))
        d.text((ox + 6, oy + CELL_H + 1), f"{k + 1}. {i / fps:.1f}s", font=font, fill=(200, 210, 230))
    return img
