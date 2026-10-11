"""Catalog of existing recorded moves: motion features, preview strips, and Claude-written descriptions.

    python -m meme_mirror.catalog build      # download, features, strips (no LLM)
    python -m meme_mirror.catalog describe   # Claude describes each move (cached; resumable)

Writes catalog.yaml (hand-editable: Joe's corrections go in `notes` / `exclude`, which rebuilds keep).
"""

import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog.yaml"
STRIPS = ROOT / "cache" / "strips"

SOURCES = {
    "pollen": "pollen-robotics/reachy-mini-emotions-library",
    "ac": "Anne-Charlotte/new-emotions",
    "stella": "stellaaaa/thinking-animations",
}
SKIP_PREFIXES = ("take-",)  # unnamed Marionette takes
MAX_DURATION_S = 20.0


# ---------- loading and features ----------

def move_files():
    """Yield (move_id, source, name, path) for every catalogued move."""
    for source, repo in SOURCES.items():
        root = snapshot_download(repo, repo_type="dataset")
        for path in sorted(glob.glob(f"{root}/**/*.json", recursive=True)):
            name = Path(path).stem
            if name.startswith(SKIP_PREFIXES):
                continue
            yield f"{source}/{name}", source, name, path


def load_channels(path):
    """Move JSON (path or dict) -> dict of per-frame channels in degrees / mm, plus time and description."""
    d = path if isinstance(path, dict) else json.load(open(path))
    data = d.get("set_target_data")
    if not data:
        return None
    t = np.asarray(d["time"], float)
    heads = np.asarray([s["head"] for s in data], float)
    rpy = R.from_matrix(heads[:, :3, :3]).as_euler("xyz", degrees=True)
    ant = np.rad2deg(np.asarray([s["antennas"] for s in data], float))
    return dict(
        t=t, description=d.get("description", ""),
        roll=rpy[:, 0], pitch=rpy[:, 1], yaw=rpy[:, 2],
        x=heads[:, 0, 3] * 1000, y=heads[:, 1, 3] * 1000, z=heads[:, 2, 3] * 1000,
        ant_r=ant[:, 0], ant_l=ant[:, 1],
        body_yaw=np.rad2deg(np.asarray([s.get("body_yaw", 0.0) for s in data], float)),
    )


def _dominant_freq(sig, dt):
    sig = sig - sig.mean()
    if len(sig) < 16 or np.abs(sig).max() < 2.0:
        return 0.0
    power = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2
    freqs = np.fft.rfftfreq(len(sig), dt)
    band = (freqs > 0.4) & (freqs < 6.0)
    if not band.any() or power[band].max() < 0.25 * power[1:].sum():
        return 0.0  # no clear rhythm
    return float(freqs[band][np.argmax(power[band])])


def features(ch):
    """Summary numbers an LLM (and Joe) can reason with. Angles in deg, height in mm.
    Antenna 'droop' > 0 means both antennas swung outward/down (right negative, left positive)."""
    t = ch["t"]
    dt = float(np.median(np.diff(t)))
    droop = (ch["ant_l"] - ch["ant_r"]) / 2
    asym = (ch["ant_l"] + ch["ant_r"]) / 2
    head_speed = np.sqrt(sum(np.gradient(ch[k], dt) ** 2 for k in ("roll", "pitch", "yaw")))
    ant_speed = np.abs(np.gradient(ch["ant_r"], dt)) + np.abs(np.gradient(ch["ant_l"], dt))
    f = dict(duration_s=round(float(t[-1]), 2))
    for k in ("pitch", "roll", "yaw", "z", "body_yaw"):
        f[k] = dict(mean=round(float(ch[k].mean()), 1), min=round(float(ch[k].min()), 1),
                    max=round(float(ch[k].max()), 1))
    f["antenna_droop"] = dict(mean=round(float(droop.mean()), 1), min=round(float(droop.min()), 1),
                              max=round(float(droop.max()), 1))
    f["antenna_asymmetry_mean_abs"] = round(float(np.abs(asym).mean()), 1)
    f["head_peak_speed_dps"] = round(float(np.percentile(head_speed, 98)), 0)
    f["antenna_peak_speed_dps"] = round(float(np.percentile(ant_speed, 98)), 0)
    f["still_fraction"] = round(float(((head_speed < 8) & (ant_speed < 15)).mean()), 2)
    f["rhythm_hz"] = dict(pitch=round(_dominant_freq(ch["pitch"], dt), 2),
                          roll=round(_dominant_freq(ch["roll"], dt), 2),
                          antennas=round(_dominant_freq(droop, dt), 2))
    return f


# ---------- build ----------

def build(render=True):
    old = yaml.safe_load(CATALOG.read_text()) if CATALOG.exists() else {}
    STRIPS.mkdir(parents=True, exist_ok=True)
    renderer = None
    if render:
        from meme_mirror.preview.sim_render import SimRenderer, load_move, strip
        import imageio.v2 as imageio
        renderer = SimRenderer()
    cat = {}
    for move_id, source, name, path in move_files():
        ch = load_channels(path)
        if ch is None or ch["t"][-1] > MAX_DURATION_S:
            continue
        sound = next((str(Path(path).with_suffix(s)) for s in (".ogg", ".wav")
                      if Path(path).with_suffix(s).exists()), None)
        strip_path = STRIPS / f"{move_id.replace('/', '__')}.png"
        if renderer and not strip_path.exists():
            imageio.imwrite(strip_path, strip(renderer.render(*load_move(path)), n=6))
        entry = dict(source=source, repo=SOURCES[source], name=name, file=path, sound=sound,
                     strip=str(strip_path), recorded_description=ch["description"],
                     features=features(ch))
        prev = old.get(move_id, {})
        for keep in ("description", "notes", "exclude"):  # LLM output and Joe's edits survive rebuilds
            if keep in prev:
                entry[keep] = prev[keep]
        cat[move_id] = entry
        print(f"{move_id:40s} {entry['features']['duration_s']:5.1f}s")
    save(cat)
    print(f"{len(cat)} moves -> {CATALOG}")


def save(cat):
    CATALOG.write_text(yaml.safe_dump(cat, sort_keys=False, allow_unicode=True, width=110))


def load():
    return yaml.safe_load(CATALOG.read_text())


# ---------- describe (LLM) ----------

class MoveDescription(BaseModel):
    move_id: str
    reads_as: list[str] = Field(description="2-5 emotions/attitudes a typical viewer would infer, most likely first")
    summary: str = Field(description="one sentence: what the robot visibly does, in plain words")
    valence: float = Field(ge=-1, le=1)
    arousal: float = Field(ge=0, le=1)
    intensity: int = Field(ge=1, le=5)
    timing: str = Field(description="how it unfolds over time, e.g. 'slow sink then hold', 'sudden recoil, quick recovery'")
    key_cues: list[str] = Field(description="the bodily cues that carry the meaning (e.g. 'antennas droop outward')")
    meme_uses: list[str] = Field(description="kinds of meme situations it could express, if any")


class Batch(BaseModel):
    moves: list[MoveDescription]


DESCRIBE_PROMPT = """You are cataloguing expressive motions of Reachy Mini, a small desk robot: a head on a
6-DoF platform (pitch: positive = looking down; roll; yaw; height z), two antennas (droop > 0 = both swing
outward/down, like drooping ears; negative = raised/inward), and a rotating body. No face, arms or voice.

Each image is a strip of 6 frames sampled evenly through one motion, seen from the front-left.
For each move you also get its library name, the recorder's description (may be empty or just a timestamp),
and numeric features (angles in degrees, height in mm).

Describe how a typical human viewer would READ each motion emotionally. Judge from the frames and numbers;
use the name as a hint, not as truth. Return one entry per move, with move_id copied exactly.

Moves:
"""


def describe(batch_size=6, workers=4, only_missing=True):
    from meme_mirror.llm import get_provider

    cat = load()
    todo = [m for m, e in cat.items() if not (only_missing and "description" in e)]
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    provider = get_provider("cataloger")

    def run(batch):
        lines, images = [], []
        for m in batch:
            e = cat[m]
            images.append(e["strip"])
            lines.append(json.dumps(dict(move_id=m, image=Path(e["strip"]).name, name=e["name"],
                                         recorded_description=e["recorded_description"],
                                         features=e["features"])))
        return provider.complete(DESCRIBE_PROMPT + "\n".join(lines), Batch, images=images)

    with ThreadPoolExecutor(workers) as pool:
        for batch, result in zip(batches, pool.map(run, batches)):
            got = {d.move_id: d for d in result.moves}
            for m in batch:
                if m in got:
                    cat[m]["description"] = got[m].model_dump(exclude={"move_id"})
                else:
                    print(f"  missing description for {m}")
            save(cat)  # incremental
            print(f"described {len(got)}/{len(batch)}: {', '.join(batch)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "describe"])
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    if a.cmd == "build":
        build(render=not a.no_render)
    else:
        describe()
