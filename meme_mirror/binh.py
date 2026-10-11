"""The 'binh' generator path: Claude-written motion recipes -> binhpham's flow-matching generator (CPU) -> moves.

Uses third_party/reachy-motion-generator (Apache-2.0, pham-tuan-binh). Their planner LLMs are skipped:
our proposer writes the recipe in their grammar, using their teacher system prompt (RECIPE_GUIDE).
"""

from functools import lru_cache

from common.motion import to_move
from common.reach import Reach
from planner.dsl import check as _check
from planner.dsl import variants
from planner.prompt import SYSTEM as RECIPE_GUIDE  # noqa: F401  (grammar + 14 worked examples)


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
