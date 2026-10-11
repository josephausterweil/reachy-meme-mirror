import json

import pytest

from meme_mirror import learning, pipeline


READING = dict(core_emotion="smug", nuance="quietly superior", arc="a slow lean back")


def _cand(cid, source, rating=None, comment=None, chosen=None, saved=None):
    fb = {k: v for k, v in dict(rating=rating, comment=comment, chosen=chosen).items() if v is not None}
    c = dict(id=cid, round=1, label=f"label {cid}", source=source, rationale="r",
             spec={"title": "t", "segments": [{"kind": "hold", "seconds": 1}]} if source != "binh" else None,
             recipe="hold 1" if source == "binh" else None, match=0.5)
    if fb:
        c["feedback"] = fb
    if saved:
        c["saved_as"] = saved
    return c


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SESSIONS", tmp_path / "sessions")

    def make(sid, cands):
        d = tmp_path / "sessions" / sid
        d.mkdir(parents=True)
        (d / "session.json").write_text(json.dumps(dict(id=sid, meme="m.png", notes="", reading=READING,
                                                        candidates=cands)))
    return make


def test_preference_signals():
    assert learning.preference(_cand("a", "own", rating=3)) == 3
    assert learning.preference(_cand("a", "own", rating=3, saved="x")) == 3.5
    assert learning.preference(_cand("a", "own", rating=5, saved="x")) == 5
    assert learning.preference(_cand("a", "own", saved="x")) == 4.5
    assert learning.preference(_cand("a", "own", chosen=True)) == 4
    assert learning.preference(_cand("a", "own")) is None


def test_examples_exclude_current_session_and_split_by_approach(sessions):
    sessions("s1", [_cand("c1", "own", rating=5, comment="love the lean"),
                    _cand("c2", "binh", rating=1, comment="too jittery"),
                    _cand("c3", "library")])
    sessions("s2", [_cand("c1", "binh", chosen=True)])
    spec = learning.examples_text("spec", current_session="s2")
    assert "love the lean" in spec and "too jittery" not in spec
    rec = learning.examples_text("recipe", current_session="s2")
    assert "too jittery" in rec and "Disliked" in rec
    assert "played on Reachy" not in rec  # s2 is the current session: excluded
    assert learning.examples_text("spec", current_session="s1") == ""


def test_suggested_mix_follows_preferences(sessions):
    assert learning.suggested_mix() == {"library": 1, "own": 1, "binh": 2}  # no data: original default
    sessions("s1", [_cand(f"o{i}", "own", rating=5) for i in range(4)] +
                   [_cand(f"b{i}", "binh", rating=2) for i in range(4)])
    st = learning.source_stats()
    assert st["own"]["n"] == 4 and st["own"]["mean"] == 5 and st["binh"]["mean"] == 2
    assert learning.suggested_mix() == {"library": 1, "own": 2, "binh": 1}


def test_lessons_need_three_ratings(sessions):
    sessions("s1", [_cand("c1", "own", rating=4), _cand("c2", "binh", rating=2)])
    assert learning.lessons() == []  # no LLM call below the threshold


def test_history_mentions_picks():
    s = pipeline.Session.__new__(pipeline.Session)
    s.data = dict(candidates=[_cand("r1c1", "own", rating=4, chosen=True, saved="x")])
    h = pipeline._history_text(s, "new", None)
    assert "PLAYED" in h and "SAVED" in h
