"""Learning from the user's feedback across memes (prompt-level only: no model training).

Source of truth: the feedback stored on candidates in every runs/sessions/*/session.json.
1. preference(c): one number per candidate from stars, the 'good fit' toggle, and saving (playing is not a vote).
2. examples_text(): best and worst past expressions from OTHER memes, as few-shot guidance for the proposer.
   lessons(): general lessons distilled by an LLM from all notes (cached until feedback changes).
3. source_stats() / suggested_mix(): which source (library / own / binh) the user likes best, to shift the default mix.
"""

import hashlib
import json

from pydantic import BaseModel, Field

from .config import user_name
from .llm import get_provider

SOURCES = ("library", "own", "binh")
PRIOR_MEAN, PRIOR_WEIGHT = 3.0, 2.0  # shrink small samples toward "average"


def _sessions_dir():
    from .pipeline import SESSIONS

    return SESSIONS


def preference(c):
    """Stars if given (+0.5 if saved, capped at 5); otherwise saved or marked good fit -> 4.5; else None."""
    fb = c.get("feedback", {})
    saved = bool(c.get("saved_as"))
    if fb.get("rating"):
        return min(5.0, fb["rating"] + (0.5 if saved else 0))
    if saved or fb.get("picked"):
        return 4.5
    return None


def rated_candidates(exclude_session=None):
    """[(session_data, candidate, pref)] for every candidate with any feedback signal."""
    out = []
    for path in sorted(_sessions_dir().glob("*/session.json")):
        s = json.loads(path.read_text())
        if s["id"] == exclude_session or not s.get("reading"):
            continue
        for c in s["candidates"]:
            p = preference(c)
            if p is not None:
                out.append((s, c, p))
    return out


# ---------- 1+2: examples from other memes ----------

def _motion(c):
    return json.dumps(c["spec"], separators=(",", ":")) if c.get("spec") else c.get("recipe")


def _example_line(s, c, p):
    r = s["reading"]
    fb = c.get("feedback", {})
    tags = [f"{p:g}/5"] + (["marked good fit"] if fb.get("picked") else []) + (["saved"] if c.get("saved_as") else [])
    return (f"- meme felt like: {r['core_emotion']} ({r['nuance'][:160]})\n"
            f"  expression: {c['label']} [{c['source']}; {', '.join(tags)}]\n"
            f"  {user_name()}'s note: {fb.get('comment') or '-'}\n"
            f"  motion: {_motion(c)}")


def examples_text(approach, current_session=None, n_good=4, n_bad=2):
    """Few-shot block for the proposer. approach: 'spec' (library/own) or 'recipe' (binh)."""
    want = ("library", "own") if approach == "spec" else ("binh",)
    rows = [(s, c, p) for s, c, p in rated_candidates(current_session) if c["source"] in want]
    good = sorted((r for r in rows if r[2] >= 4), key=lambda r: -r[2])[:n_good]
    bad = sorted((r for r in rows if r[2] <= 2 and r[1].get("feedback", {}).get("comment")),
                 key=lambda r: r[2])[:n_bad]
    if not good and not bad:
        return ""
    parts = [f"# What {user_name()} liked and disliked on OTHER memes (learn their taste; do not copy verbatim)"]
    if good:
        parts += ["Liked:"] + [_example_line(*r) for r in good]
    if bad:
        parts += ["Disliked:"] + [_example_line(*r) for r in bad]
    return "\n".join(parts)


class Lessons(BaseModel):
    lessons: list[str] = Field(max_length=10, description="short, general, actionable lessons about the user's taste")


LESSONS_PROMPT = """{name} rates expressive motions of Reachy Mini (a small robot: head with pitch/roll/yaw/height,
two antenna 'ears', rotating body; no face) that are meant to convey the feeling of memes.
Below is every rated expression with his stars and notes. Distill up to 10 GENERAL lessons about what
works for him and what doesn't (cues, timing, intensity, which kinds of motion read as which feelings).
Only state lessons the evidence supports; prefer their own words' meaning. Skip one-off details.

{rows}"""


def lessons():
    """General lessons from all feedback, cached by a hash of that feedback. Needs >= 3 rated candidates."""
    rows = rated_candidates()
    if len(rows) < 3:
        return []
    body = "\n".join(_example_line(*r) for r in rows)
    key = hashlib.sha256(body.encode()).hexdigest()[:16]
    cache = _sessions_dir().parent / "lessons.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        if d.get("key") == key:
            return d["lessons"]
    result = get_provider("lessons").complete(LESSONS_PROMPT.format(name=user_name(), rows=body), Lessons)
    cache.write_text(json.dumps(dict(key=key, n=len(rows), lessons=result.lessons), indent=1))
    return result.lessons


def lessons_text():
    ls = lessons()
    return (f"# Lessons from {user_name()}'s past ratings\n" + "\n".join(f"- {x}" for x in ls)) if ls else ""


# ---------- 3: which source does the user prefer? ----------

def source_stats():
    """Per source: n rated, raw mean preference, shrunk mean, mean blind match, n saved/played."""
    stats = {}
    rows = rated_candidates()
    for src in SOURCES:
        ps = [p for _, c, p in rows if c["source"] == src]
        matches = [c["match"] for _, c, _ in rows if c["source"] == src and c.get("match") is not None]
        stats[src] = dict(
            n=len(ps),
            mean=round(sum(ps) / len(ps), 2) if ps else None,
            shrunk=round((sum(ps) + PRIOR_MEAN * PRIOR_WEIGHT) / (len(ps) + PRIOR_WEIGHT), 2),
            match=round(sum(matches) / len(matches), 2) if matches else None,
            picked=sum(1 for _, c, _ in rows if c["source"] == src and c.get("feedback", {}).get("picked")),
            saved=sum(1 for _, c, _ in rows if c["source"] == src and c.get("saved_as")))
    return stats


def suggested_mix(k=4):
    """One slot per source (keep exploring all three), the rest to the best-liked source.
    With no ratings yet, the extra slot goes to binh (the original default)."""
    stats = source_stats()
    mix = {s: 1 for s in SOURCES}
    if sum(st["n"] for st in stats.values()) == 0:
        best = "binh"
    else:
        best = max(SOURCES, key=lambda s: stats[s]["shrunk"])
    mix[best] += max(0, k - len(SOURCES))
    return mix
