"""Meme -> emotion reading -> candidate expressions (library / own spec / binh recipe) -> previews + legibility.

    python -m meme_mirror.pipeline path/to/meme.png [--notes "it's smug, not happy"] [-k 4]

Sessions live in runs/sessions/<id>/ (session.json, previews, move files) so the web UI and CLI share them.
"""

import argparse
import json
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, create_model, field_validator

from . import catalog as catalog_mod
from .compiler import apply_dials, compile_spec, move_to_tracks, to_move
from .llm import get_provider
from .safety import check as safety_check
from .spec import ExpressionSpec

SESSIONS = Path(__file__).resolve().parents[1] / "runs" / "sessions"
Source = Literal["library", "own", "binh"]


# ---------- schemas ----------

class EmotionReading(BaseModel):
    meme_template: str | None = Field(None, description="name of the meme template if recognized")
    depicts: str = Field(description="what the image literally shows, one sentence")
    core_emotion: str = Field(description="the feeling the meme expresses, 1-3 words")
    nuance: str = Field(description="what makes this feeling specific (irony, self-deprecation, ...)")
    valence: float = Field(ge=-1, le=1)
    arousal: float = Field(ge=0, le=1)
    dominance: float = Field(ge=0, le=1, description="0 = small/helpless, 1 = in control/superior")
    directed_at: Literal["self", "viewer", "third_party", "situation"]
    arc: str = Field(description="how the feeling unfolds over a few seconds, e.g. 'calm, a flicker of panic, back to forced calm'")
    bodily_cues: list[str] = Field(description="how a person would show this with head, gaze and posture only (no face, no hands)")
    text_prompt: str = Field(description="'word. You ... one sentence of situation.' form, e.g. 'denial. You smile calmly while everything burns.'")
    distractors: list[str] = Field(description="5 other emotions a viewer could plausibly confuse this with")
    confidence: Literal["low", "medium", "high"]


class SpecCandidate(BaseModel):
    label: str = Field(description="short name, e.g. 'forced calm + side glance'")
    rationale: str = Field(description="why a viewer would feel the meme's emotion from this motion")
    spec: ExpressionSpec
    sound: str | None = Field(None, description="catalog move_id whose sound fits, or null for silence")


class RecipeCandidate(BaseModel):
    label: str
    rationale: str
    recipe: str = Field(description="motion recipe in the recipe grammar")
    sound: str | None = None

    @field_validator("recipe")
    @classmethod
    def _valid_recipe(cls, v):
        from .binh import check

        err = check(v)
        if err:
            raise ValueError(f"invalid recipe: {err}")
        return v


def _proposals(item, n):
    return create_model(f"{item.__name__}s_{n}",
                        candidates=(list[item], Field(min_length=n, max_length=n)))


class OptionProb(BaseModel):
    option: str
    p: float = Field(ge=0, le=1)


class MatchScore(BaseModel):
    score: float = Field(ge=0, le=1, description="0 = unrelated feeling, 1 = same emotional experience")
    reason: str


class JudgeResult(BaseModel):
    free_reading: str = Field(description="in a few words, what the robot seems to feel")
    probabilities: list[OptionProb]


# ---------- prompts ----------

ROBOT = """Reachy Mini is a small desk robot: a head on a 6-DoF platform (no face: two round camera "eyes"),
two antennas that act like ears, and a rotating body. No arms, no voice of its own (short sounds can be played)."""

SPEC_GUIDE = """# Approach "spec": an ExpressionSpec (our own compiler; reuse recorded library moves and/or keyframes)
Units: pitch deg (+ = head LOWERED, range -25..30), roll deg (±30), yaw deg (+ = turn to robot's left, ±60),
z mm (+ = taller, -45..20), ear_r / ear_l = ear droop deg (0 = straight up, + = swung outward/down like sad ears,
up to 175; negative = raised inward, down to -40; `ears` sets both), body_yaw deg (±60). Neutral: all 0, ears 10.
Segments run in order:
- {"kind":"library","move_id":<catalog id>,"time_scale":1,"amplitude":1,"start_s":0,"end_s":null}: a recorded move
  (time_scale > 1 = slower; amplitude scales it; start_s/end_s trim). Transitions are blended automatically.
- {"kind":"keyframes","keyframes":[{"dt":0.4,"pitch":15,"ears":140,"ease":"smooth"}, ...]}: each keyframe eases
  to the listed channels over dt seconds (unlisted channels hold). ease: smooth | linear | snap (fast start).
- {"kind":"hold","seconds":1.0}
Overlays add rhythm on top: {"channel":"roll","shape":"sine"|"tremble","amplitude":4,"freq_hz":2,"start_s":1,
"end_s":3,"antiphase_ears":false}. Give every segment an `intent`.
The same motion principles apply as for recipes below: tell a tiny story (onset -> expression -> settle),
commit to postures where ears, pitch and height agree, fast ear changes, stillness after a burst is expressive."""

RECIPE_INTRO = """# Approach "recipe": a motion recipe for a learned motion generator (adds organic, lifelike detail)
Write `recipe` in exactly this grammar:"""

INTERPRET_PROMPT = ROBOT + """

Look at this meme. Work out the emotional experience it conveys: what a person who shares or relates to it
feels, including irony or self-deprecation. Then describe how that feeling could show in a body with only
head, gaze and posture. {notes}"""

PROPOSE_TASK = """# The meme's emotion
{reading}
{notes}
# Your task
Propose exactly {k} candidate expressions for Reachy that would make a human watching the robot infer the
meme's emotional experience. {what}
Make the candidates genuinely different from each other (different cues, timing, or interpretation).
Keep each 3-10 s. Pick a fitting sound from the catalog (sounds are named after their move) or null.
{learned}
{history}"""

SPEC_WHAT = """Write each as an ExpressionSpec ("spec" approach).
- The first {n_lib} should use ONLY library segments (a single well-chosen recorded move, or a short sequence;
  light changes to time_scale / amplitude / trim are fine). Only use moves whose catalog reading genuinely fits;
  if nothing fits, build it from keyframes instead.
- The remaining {n_own} should be built mainly from keyframes and overlays (library pieces may be spliced in)."""

RECIPE_WHAT = "Write each as a motion recipe (the recipe grammar above)."

JUDGE_PROMPT = ROBOT + """

The image shows 6 frames sampled evenly through one motion of the robot ({duration:.1f} s), left to right,
seen from its front-left. Motion summary (degrees; pitch + = head lowered; ear droop + = ears hang outward/down):
{features}

Without any other context, what is the robot feeling? Give your free reading, then a probability for each
option (they should sum to 1): {options}"""


MATCH_PROMPT = """An observer watched a robot's motion without knowing what it was meant to express, and wrote:
"{free}"

The motion was meant to convey: {core} ({nuance}). Intended arc: {arc}

How close is what the observer perceived to the intended emotional experience? Score 0-1. Give credit for
the right feeling and arc even in different words; give less when only the surface movement matches."""


def catalog_digest(cat):
    lines = []
    for mid, e in cat.items():
        if e.get("exclude"):
            continue
        d = e.get("description", {})
        lines.append(f"{mid} ({e['features']['duration_s']:.1f}s): reads as {', '.join(d.get('reads_as', []))}. "
                     f"{d.get('summary', '')} Timing: {d.get('timing', '')}."
                     + (f" Note: {e['notes']}" if e.get("notes") else ""))
    return "# Catalog of recorded library moves\n" + "\n".join(lines)


# ---------- session ----------

def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "meme"


class Session:
    def __init__(self, path: Path):
        self.dir = path
        self.data = json.loads((path / "session.json").read_text())

    @classmethod
    def create(cls, meme_path, notes=""):
        sid = time.strftime("%Y%m%d-%H%M%S") + "-" + _slug(Path(meme_path).stem)
        d = SESSIONS / sid
        d.mkdir(parents=True)
        meme = d / ("meme" + Path(meme_path).suffix.lower())
        shutil.copy(meme_path, meme)
        (d / "session.json").write_text(json.dumps(dict(id=sid, meme=meme.name, notes=notes,
                                                        reading=None, candidates=[]), indent=1))
        return cls(d)

    @classmethod
    def load(cls, sid):
        return cls(SESSIONS / sid)

    def save(self):
        (self.dir / "session.json").write_text(json.dumps(self.data, indent=1))

    @property
    def reading(self):
        return EmotionReading.model_validate(self.data["reading"]) if self.data["reading"] else None

    def candidate(self, cid):
        return next(c for c in self.data["candidates"] if c["id"] == cid)


# ---------- steps ----------

def interpret(session: Session, notes=None):
    notes = session.data["notes"] if notes is None else notes
    extra = f"\nThe user adds: {notes}" if notes else ""
    reading = get_provider("interpreter").complete(
        INTERPRET_PROMPT.format(notes=extra), EmotionReading, images=[session.dir / session.data["meme"]])
    session.data["reading"] = reading.model_dump()
    session.data["notes"] = notes
    session.save()
    return reading


def _history_text(session, mode, anchor):
    cands = session.data["candidates"]
    if not cands:
        return ""
    lines = ["# Already tried for THIS meme (do NOT repeat these; learn from ratings, comments, and which ones "
             "Joe played or saved: those are his picks)"]
    for c in cands:
        fb = c.get("feedback", {})
        what = json.dumps(c["spec"]) if c.get("spec") else c.get("recipe")
        picked = ("; Joe PLAYED it on Reachy" if fb.get("chosen") else "") + \
                 ("; Joe SAVED it" if c.get("saved_as") else "")
        lines.append(f"- [{c['id']}] {c['label']} ({c['source']}): {c['rationale']}\n  motion: {what}\n"
                     f"  rating: {fb.get('rating', 'unrated')}/5{picked}; comment: {fb.get('comment', '-')}; "
                     f"blind match: {c.get('match', '?')}")
    if mode == "more_like" and anchor:
        lines.append(f"\nThe user wants MORE LIKE [{anchor}]: keep what works in it and propose variations that "
                     "push it further in the direction of their feedback.")
    else:
        lines.append("\nThe user wants NEW WAYS of expressing it: explore different interpretations and cues.")
    return "\n".join(lines)


def propose(session: Session, k=4, mix=None, mode="new", anchor=None, notes=""):
    """mix: counts per source, e.g. {"library": 1, "own": 1, "binh": 2} (k is ignored when mix is given)."""
    from . import learning

    mix = mix or learning.suggested_mix(k)
    n_spec, n_rec = mix.get("library", 0) + mix.get("own", 0), mix.get("binh", 0)
    reading = session.reading
    note_text = notes or session.data["notes"]
    common = dict(reading=reading.model_dump_json(indent=1, exclude={"distractors"}),
                  notes=f"\nThe user's notes: {note_text}\n" if note_text else "",
                  history=_history_text(session, mode, anchor))
    cat = catalog_mod.load()
    sounds = "# Sounds available (by move id)\n" + ", ".join(m for m, e in cat.items() if e.get("sound"))
    lessons = learning.lessons_text()
    jobs = []
    if n_spec:
        what = SPEC_WHAT.format(n_lib=mix.get("library", 0), n_own=mix.get("own", 0))
        learned = "\n\n".join(x for x in (lessons, learning.examples_text("spec", session.data["id"])) if x)
        jobs.append((PROPOSE_TASK.format(k=n_spec, what=what, learned=learned, **common),
                     _proposals(SpecCandidate, n_spec),
                     "\n\n".join([ROBOT, SPEC_GUIDE, catalog_digest(cat)])))
    if n_rec:
        from .binh import RECIPE_GUIDE

        learned = "\n\n".join(x for x in (lessons, learning.examples_text("recipe", session.data["id"])) if x)
        jobs.append((PROPOSE_TASK.format(k=n_rec, what=RECIPE_WHAT, learned=learned, **common),
                     _proposals(RecipeCandidate, n_rec),
                     "\n\n".join([ROBOT, RECIPE_INTRO, RECIPE_GUIDE, sounds])))
    provider = get_provider("proposer")
    with ThreadPoolExecutor(len(jobs)) as pool:
        results = list(pool.map(lambda j: provider.complete(j[0], j[1], system=j[2]), jobs))

    round_no = 1 + max([c["round"] for c in session.data["candidates"]], default=0)
    new = []
    for pc in (c for r in results for c in r.candidates):
        c = dict(id=f"r{round_no}c{len(new) + 1}", round=round_no, label=pc.label, rationale=pc.rationale,
                 sound=pc.sound if pc.sound in cat else None, mode=mode, anchor=anchor, notes=notes)
        if isinstance(pc, RecipeCandidate):
            c.update(source="binh", recipe=pc.recipe, spec=None)
        else:
            only_lib = all(seg.kind == "library" for seg in pc.spec.segments) and not pc.spec.overlays
            c.update(source="library" if only_lib else "own", spec=pc.spec.model_dump(), recipe=None)
        new.append(c)
    return new


def realize(session: Session, c, speed=1.0, amplitude=1.0, render=True, renderer=None):
    """Build the move for candidate c (dict), safety-check it, save move JSON + previews. Mutates c."""
    try:
        if c["source"] == "binh":
            from . import binh

            seed = c.setdefault("seed", random.randrange(10_000))
            tracks = move_to_tracks(binh.generate(c["recipe"], n=1, seed=seed, description=c["label"])[0])
        else:
            tracks = compile_spec(ExpressionSpec.model_validate(c["spec"]))
        tracks = apply_dials(tracks, speed, amplitude)
        tracks, rep = safety_check(tracks, ik=False)
        move, projected = _reach().project(to_move(tracks, description=f"{c['label']}: {c['rationale']}"))
    except Exception as e:  # bad recipe / unknown move id etc.: keep the candidate, mark it failed
        c.update(error=str(e), safety="failed")
        return c
    summary = rep.summary()
    if projected > 0.005:
        summary = (summary + "; " if summary != "ok" else "") + f"{100 * projected:.0f}% of frames moved to reachable poses"
    c.update(error=None, safety=summary, safe=rep.ok, dials=dict(speed=speed, amplitude=amplitude),
             duration_s=round(len(tracks["pitch"]) / 50, 2), projected=round(projected, 3),
             features=catalog_mod.features(dict(t=_t(tracks), **_chan(tracks))))
    (session.dir / f"{c['id']}.json").write_text(json.dumps(move))
    c["move_file"] = f"{c['id']}.json"
    if render:
        import imageio.v2 as imageio
        from .preview.sim_render import SimRenderer, load_move, strip

        renderer = renderer or SimRenderer()
        frames = renderer.render(*load_move(session.dir / c["move_file"]), session.dir / f"{c['id']}.mp4")
        imageio.imwrite(session.dir / f"{c['id']}_strip.png", strip(frames))
        c.update(video=f"{c['id']}.mp4", strip=f"{c['id']}_strip.png")
    return c


_REACH = None


def _reach():
    """binhpham's projection onto the robot's reachable set (SDK analytical IK)."""
    global _REACH
    if _REACH is None:
        from common.reach import Reach

        _REACH = Reach()
    return _REACH


def _t(tracks):
    import numpy as np

    return np.arange(len(tracks["pitch"])) / 50


def _chan(tracks):
    return dict(roll=tracks["roll"], pitch=tracks["pitch"], yaw=tracks["yaw"], z=tracks["z"],
                x=tracks["x"], y=tracks["y"], body_yaw=tracks["body_yaw"],
                ant_r=-tracks["ear_r"], ant_l=tracks["ear_l"])


def judge(session: Session, c):
    """Blind listener: from frames + motion summary only, P(meme's core emotion) among distractors."""
    if not c.get("strip"):
        return c
    reading = session.reading
    options = [reading.core_emotion] + reading.distractors[:5]
    random.Random(c["id"]).shuffle(options)
    f = c["features"]
    summary = json.dumps({k: f[k] for k in ("pitch", "roll", "yaw", "z", "antenna_droop", "head_peak_speed_dps",
                                            "still_fraction", "rhythm_hz")})
    res = get_provider("judge").complete(
        JUDGE_PROMPT.format(duration=c["duration_s"], features=summary, options="; ".join(options)),
        JudgeResult, images=[session.dir / c["strip"]])
    total = sum(o.p for o in res.probabilities) or 1
    probs = {o.option.lower(): o.p / total for o in res.probabilities}
    match = get_provider("grader").complete(
        MATCH_PROMPT.format(free=res.free_reading, core=reading.core_emotion, nuance=reading.nuance,
                            arc=reading.arc), MatchScore)
    c.update(judge_reading=res.free_reading, judge_probs=probs,
             legibility=round(probs.get(reading.core_emotion.lower(), 0.0), 2),
             match=round(match.score, 2), match_reason=match.reason)
    return c


def run_round(session: Session, k=4, mix=None, mode="new", anchor=None, notes="", do_judge=True, log=print):
    """propose -> realize (+render) -> judge; appends to the session. Returns the new candidates."""
    log(f"proposing {k} candidates ({mode})...")
    new = propose(session, k=k, mix=mix, mode=mode, anchor=anchor, notes=notes)
    from .preview.sim_render import SimRenderer

    renderer = SimRenderer()
    for c in new:
        log(f"  building {c['id']} [{c['source']}] {c['label']}")
        realize(session, c, renderer=renderer)
    if do_judge:
        log("  blind judging...")
        with ThreadPoolExecutor(4) as pool:
            list(pool.map(lambda c: judge(session, c), new))
    session.data["candidates"].extend(new)
    session.save()
    return new


def feedback(session: Session, cid, rating=None, comment=None, chosen=None, saved_as=None):
    c = session.candidate(cid)
    fb = c.setdefault("feedback", {})
    if saved_as:
        c["saved_as"] = saved_as
    if rating is not None:
        fb["rating"] = rating
    if comment:
        fb["comment"] = comment
    if chosen is not None:
        fb["chosen"] = chosen
    session.save()
    with open(SESSIONS.parent / "ratings.jsonl", "a") as fh:  # preference log across sessions
        fh.write(json.dumps(dict(session=session.data["id"], candidate=cid, source=c["source"],
                                 legibility=c.get("legibility"), match=c.get("match"), core_emotion=session.reading.core_emotion,
                                 saved_as=c.get("saved_as"), t=time.time(), **fb)) + "\n")


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meme")
    ap.add_argument("--notes", default="")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    s = Session.create(a.meme, a.notes)
    print(f"session {s.data['id']}")
    r = interpret(s)
    print(f"reading: {r.core_emotion} | {r.nuance} | arc: {r.arc} | prompt: {r.text_prompt}")
    for c in run_round(s, k=a.k, do_judge=not a.no_judge):
        print(f"[{c['id']}] {c['source']:7s} {c['label']}\n    why: {c['rationale']}\n"
              f"    safety: {c.get('safety')} | multiple-choice: {c.get('legibility')} | match: {c.get('match')}\n"
              f"    blind judge: {c.get('judge_reading')}"
              + (f"\n    ERROR: {c['error']}" if c.get("error") else ""))
    print(f"previews in {s.dir}")


if __name__ == "__main__":
    main()
