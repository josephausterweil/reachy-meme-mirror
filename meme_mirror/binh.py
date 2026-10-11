"""The 'binh' generator path: model-written motion recipes -> binhpham's flow-matching generator (CPU) -> moves.

Uses a vendored copy of reachy-motion-generator (vendor/binh, Apache-2.0, Pham Tuan Binh). Their planner LLMs are
skipped: our proposer writes the recipe in their grammar, using their teacher system prompt (RECIPE_GUIDE).
Parsing/validating recipes needs only numpy+scipy; generating motion also needs PyTorch (CPU build is enough).
"""

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

_VENDOR = str(Path(__file__).resolve().parent / "vendor" / "binh")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from common.motion import to_move  # noqa: E402
from common.reach import Reach  # noqa: E402
from planner.dsl import check as _check  # noqa: E402
from planner.dsl import variants  # noqa: E402
from planner.prompt import SYSTEM as RECIPE_GUIDE  # noqa: E402,F401  (grammar + 14 worked examples)


def available() -> bool:
    """True if PyTorch is installed, so recipes can be turned into motion."""
    return importlib.util.find_spec("torch") is not None


@lru_cache(maxsize=1)
def _model():
    from generator.sample import load

    net, stats, dev = load("/nonexistent", dev="cpu")  # falls back to the HF copy of generator.pt
    return net, stats, dev, Reach()


def check(recipe: str) -> str | None:
    """None if the recipe parses and stays in range, else the error message."""
    return _check(recipe)


def generate(recipe: str, n: int = 1, seed: int = 0, description: str = "", cfg: float = 1.5):
    """Recipe -> list of n move dicts (Pollen format, 25 Hz, projected onto the reachable set).
    Variants differ in amplitude/tempo jitter and noise; odd variants are mirrored left-right."""
    if not available():
        raise RuntimeError("learned motion needs PyTorch: use 'Install' in the page, or pip install torch (CPU)")
    from generator.sample import generate_batch

    err = check(recipe)
    if err:
        raise ValueError(f"bad recipe: {err}")
    net, stats, dev, reach = _model()
    plans = variants(recipe, n, seed=seed, fc=2.0, kdt=0.25)  # as their server does: fast events stay fast
    trajs = generate_batch(net, stats, plans, dev, seeds=[seed * 1000 + i for i in range(n)], cfg=cfg)
    moves = []
    for A in trajs:
        move, _frac = reach.project(to_move(A, description))
        moves.append(move)
    return moves


def reach():
    """The reachability projector (needs no PyTorch)."""
    return _reach_singleton()


@lru_cache(maxsize=1)
def _reach_singleton():
    return Reach()
