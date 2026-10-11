"""Candidate previews: a MuJoCo video + frame strip when MuJoCo can render here, else a drawn frame strip.
The web page can always show any move in its in-browser 3D viewer.

[preview] renderer = "auto" | "mujoco" | "schematic"   (in config.local.toml)
"""

import logging
from functools import lru_cache

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _mujoco():
    """A SimRenderer if MuJoCo is installed and can render offscreen here, else None."""
    from ..config import settings

    if settings().get("preview", {}).get("renderer", "auto") == "schematic":
        return None
    try:
        from .sim_render import SimRenderer

        return SimRenderer()
    except Exception as e:  # not installed, or no offscreen GL (e.g. on the robot)
        log.info("MuJoCo previews unavailable (%s); using drawn strips", e.__class__.__name__)
        return None


def renderer_name():
    return "mujoco" if _mujoco() else "schematic"


def render(move_path, tracks, out_dir, cid):
    """Write previews for one candidate; returns {"video": name|None, "strip": name}."""
    sim = _mujoco()
    strip_name = f"{cid}_strip.png"
    if sim is not None:
        import imageio.v2 as imageio

        from .sim_render import load_move, strip

        frames = sim.render(*load_move(move_path), out_dir / f"{cid}.mp4")
        imageio.imwrite(out_dir / strip_name, strip(frames))
        return {"video": f"{cid}.mp4", "strip": strip_name}
    from .schematic import strip as drawn
    from .schematic import thumbnail

    drawn(tracks).save(out_dir / strip_name)
    thumbnail(tracks).save(out_dir / f"{cid}_thumb.png")
    return {"video": None, "strip": strip_name, "thumb": f"{cid}_thumb.png"}
