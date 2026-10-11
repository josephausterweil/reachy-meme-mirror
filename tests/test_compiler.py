import numpy as np
import pytest

from meme_mirror import catalog as cat_mod
from meme_mirror.compiler import CHANNELS, DT, compile_spec, to_move
from meme_mirror.safety import check
from meme_mirror.spec import ExpressionSpec, NEUTRAL


@pytest.fixture(scope="module")
def catalog():
    return cat_mod.load()


def _channels_of(move):
    """Move dict -> channel arrays via the same reader the catalog uses."""
    import json, tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(move, fh)
    return cat_mod.load_channels(fh.name)


def test_library_round_trip(catalog):
    """A bare library segment (no glue, no scaling) reproduces the recorded move."""
    spec = ExpressionSpec(title="t", segments=[{"kind": "library", "move_id": "pollen/sad1"}])
    tracks = compile_spec(spec, catalog, glue=False)
    got = _channels_of(to_move(tracks))
    orig = cat_mod.load_channels(catalog["pollen/sad1"]["file"])
    t0 = orig["t"] - orig["t"][0]
    for k in ("pitch", "roll", "yaw", "z", "body_yaw"):
        ref = np.interp(got["t"], t0, orig[k])
        assert np.abs(got[k] - ref).max() < 0.05, k
    for k in ("ant_r", "ant_l"):
        ref = np.interp(got["t"], t0, orig[k])
        assert np.abs(got[k] - ref).max() < 0.05, k


def test_keyframes_and_glue():
    spec = ExpressionSpec(title="t", segments=[
        {"kind": "keyframes", "keyframes": [{"dt": 1.0, "pitch": 15, "ears": 120}]},
        {"kind": "hold", "seconds": 0.5}])
    tr = compile_spec(spec, catalog={})
    i_peak = int(1.0 / DT)
    assert abs(tr["pitch"][i_peak] - 15) < 1e-6 and abs(tr["ear_l"][i_peak] - 120) < 1e-6
    assert all(abs(tr[k][-1] - NEUTRAL[k]) < 1e-6 for k in CHANNELS)  # glued back to neutral
    assert len(tr["pitch"]) == pytest.approx((1.0 + 0.5 + 0.5) / DT + 1, abs=2)


def test_overlay_and_dials():
    spec = ExpressionSpec(title="t", segments=[{"kind": "hold", "seconds": 2}],
                          overlays=[{"channel": "roll", "amplitude": 5, "freq_hz": 2}])
    tr = compile_spec(spec, catalog={})
    assert 4 < np.abs(tr["roll"]).max() <= 5
    slow = compile_spec(spec, catalog={}, speed=0.5, amplitude=2)
    assert len(slow["roll"]) == pytest.approx(2 * len(tr["roll"]), rel=0.02)
    assert 8 < np.abs(slow["roll"]).max() <= 10


def test_safety_clamps_and_reports():
    spec = ExpressionSpec(title="t", segments=[
        {"kind": "keyframes", "keyframes": [{"dt": 0.1, "pitch": 80, "yaw": 90}]}])
    out, rep = check(compile_spec(spec, catalog={}))
    assert out["pitch"].max() <= 32 and "pitch" in rep.clamped and "yaw" in rep.clamped
    assert not rep.ok  # 80 deg in 0.1 s is far too fast
    assert "clamped pitch" in rep.summary()


def test_library_moves_pass_safety(catalog):
    for m in ("pollen/sad1", "pollen/furious1", "pollen/shy1"):
        spec = ExpressionSpec(title=m, segments=[{"kind": "library", "move_id": m}])
        _, rep = check(compile_spec(spec, catalog))
        assert rep.unreachable_frames == 0, (m, rep.summary())
