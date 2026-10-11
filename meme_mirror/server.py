"""Local web UI for meme_mirror.

    .venv/bin/uvicorn meme_mirror.server:app --port 8770      then open http://localhost:8770

Nothing moves the robot except the "Play on Reachy" button (POST .../play).
"""

import json
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import catalog as catalog_mod
from . import pipeline as P

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "store"
WEB = Path(__file__).resolve().parent / "web"

app = FastAPI(title="meme_mirror")
_locks: dict[str, threading.Lock] = {}
_robot_lock = threading.Lock()


def _lock(sid):
    return _locks.setdefault(sid, threading.Lock())


def _session(sid) -> P.Session:
    if not (P.SESSIONS / sid / "session.json").exists():
        raise HTTPException(404, f"no session {sid}")
    return P.Session.load(sid)


# ---------- sessions ----------

@app.get("/api/sessions")
def list_sessions():
    out = []
    for d in sorted(P.SESSIONS.glob("*/session.json"), reverse=True):
        s = json.loads(d.read_text())
        r = s.get("reading") or {}
        out.append(dict(id=s["id"], meme=s["meme"], core_emotion=r.get("core_emotion"),
                        n_candidates=len(s["candidates"])))
    return out


@app.post("/api/sessions")
def create_session(meme: UploadFile = File(...), notes: str = Form("")):
    suffix = Path(meme.filename or "meme.png").suffix.lower() or ".png"
    if suffix not in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        raise HTTPException(400, "upload an image (png/jpg/gif/webp)")
    with tempfile.TemporaryDirectory() as tmp:
        stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(meme.filename or "meme").stem)[:40] or "meme"
        path = Path(tmp) / f"{stem}{suffix}"
        path.write_bytes(meme.file.read())
        s = P.Session.create(path, notes)
    P.interpret(s)
    return s.data


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    return _session(sid).data


class ReadingEdit(BaseModel):
    reading: dict | None = None
    notes: str | None = None
    reinterpret: bool = False


@app.put("/api/sessions/{sid}/reading")
def edit_reading(sid: str, body: ReadingEdit):
    with _lock(sid):
        s = _session(sid)
        if body.reinterpret:
            P.interpret(s, notes=body.notes if body.notes is not None else s.data["notes"])
        else:
            if body.reading is not None:
                merged = {**s.data["reading"], **body.reading}
                s.data["reading"] = P.EmotionReading.model_validate(merged).model_dump()
            if body.notes is not None:
                s.data["notes"] = body.notes
            s.save()
        return s.data


class RoundRequest(BaseModel):
    mode: str = "new"            # "new" | "more_like"
    anchor: str | None = None    # candidate id for "more_like"
    notes: str = ""
    mix: dict | None = None      # {"library": 1, "own": 1, "binh": 2}
    judge: bool = True


@app.post("/api/sessions/{sid}/round")
def new_round(sid: str, body: RoundRequest):
    with _lock(sid):
        s = _session(sid)
        new = P.run_round(s, mix=body.mix, mode=body.mode, anchor=body.anchor, notes=body.notes,
                          do_judge=body.judge, log=lambda *_: None)
        return dict(session=s.data, new=[c["id"] for c in new])


# ---------- candidates ----------

class Feedback(BaseModel):
    rating: int | None = None
    comment: str | None = None
    chosen: bool | None = None


@app.post("/api/sessions/{sid}/candidates/{cid}/feedback")
def candidate_feedback(sid: str, cid: str, body: Feedback):
    with _lock(sid):
        s = _session(sid)
        P.feedback(s, cid, rating=body.rating, comment=body.comment, chosen=body.chosen)
        return s.candidate(cid)


class Dials(BaseModel):
    speed: float = 1.0
    amplitude: float = 1.0
    rejudge: bool = False


@app.post("/api/sessions/{sid}/candidates/{cid}/dials")
def candidate_dials(sid: str, cid: str, body: Dials):
    """Re-build the candidate with new speed/amplitude (no LLM unless rejudge)."""
    if not (0.4 <= body.speed <= 2.5 and 0.3 <= body.amplitude <= 1.6):
        raise HTTPException(400, "speed 0.4-2.5, amplitude 0.3-1.6")
    with _lock(sid):
        s = _session(sid)
        c = s.candidate(cid)
        P.realize(s, c, speed=body.speed, amplitude=body.amplitude)
        if body.rejudge:
            P.judge(s, c)
        s.save()
        return c


def _sound_path(move_id):
    if not move_id:
        return None
    e = catalog_mod.load().get(move_id)
    return e.get("sound") if e else None


def _play_sound(path):
    subprocess.Popen(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class PlayRequest(BaseModel):
    sound: bool = True
    sleep_after: bool = False


@app.post("/api/sessions/{sid}/candidates/{cid}/play")
def candidate_play(sid: str, cid: str, body: PlayRequest):
    from . import robot

    s = _session(sid)
    c = s.candidate(cid)
    if not c.get("move_file"):
        raise HTTPException(400, "candidate has no move (build failed)")
    if not _robot_lock.acquire(blocking=False):
        raise HTTPException(409, "Reachy is already playing something")
    try:
        sound = _sound_path(c.get("sound")) if body.sound else None
        move = json.loads((s.dir / c["move_file"]).read_text())
        robot.play(move, sleep_after=body.sleep_after, on_start=(lambda: _play_sound(sound)) if sound else None)
    except robot.RobotBusy as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        raise HTTPException(502, f"robot error: {e}")
    finally:
        _robot_lock.release()
    return dict(ok=True)


@app.post("/api/robot/sleep")
def robot_sleep():
    from reachy_mini import ReachyMini

    from . import robot

    with _robot_lock:
        try:
            robot.check_free()
            with ReachyMini(connection_mode="network", media_backend="no_media", timeout=10) as mini:
                mini.goto_sleep()
                mini.disable_motors()
        except robot.RobotBusy as e:
            raise HTTPException(409, str(e))
    return dict(ok=True)


@app.get("/api/robot/status")
def robot_status():
    from . import robot

    try:
        st = robot.check_free()
        return dict(ok=True, motors=st["backend_status"]["motor_control_mode"])
    except robot.RobotBusy as e:
        return dict(ok=False, reason=str(e))
    except Exception as e:
        return dict(ok=False, reason=f"unreachable: {e.__class__.__name__}")


# ---------- gallery ----------

class SaveRequest(BaseModel):
    name: str


@app.post("/api/sessions/{sid}/candidates/{cid}/save")
def candidate_save(sid: str, cid: str, body: SaveRequest):
    slug = re.sub(r"[^a-z0-9]+", "-", body.name.lower()).strip("-")[:50]
    if not slug:
        raise HTTPException(400, "give it a name")
    s = _session(sid)
    c = s.candidate(cid)
    if not c.get("move_file"):
        raise HTTPException(400, "candidate has no move")
    STORE.mkdir(exist_ok=True)
    move = json.loads((s.dir / c["move_file"]).read_text())
    move["description"] = f"{body.name}: {c['label']}"
    (STORE / f"{slug}.json").write_text(json.dumps(move))
    sound = _sound_path(c.get("sound"))
    if sound:
        shutil.copy(sound, STORE / f"{slug}{Path(sound).suffix}")
    shutil.copy(s.dir / c["video"], STORE / f"{slug}.mp4") if c.get("video") else None
    meta = dict(name=body.name, slug=slug, session=sid, candidate=cid, source=c["source"],
                core_emotion=s.reading.core_emotion, meme=f"{sid}/{s.data['meme']}",
                rationale=c["rationale"], match=c.get("match"), feedback=c.get("feedback", {}))
    (STORE / f"{slug}.meta.json").write_text(json.dumps(meta, indent=1))
    with _lock(sid):
        s = _session(sid)
        s.candidate(cid)["saved_as"] = slug
        s.save()
    return meta


@app.get("/api/gallery")
def gallery():
    return [json.loads(p.read_text()) for p in sorted(STORE.glob("*.meta.json"))] if STORE.exists() else []


# ---------- files ----------

@app.get("/files/{sid}/{name}")
def session_file(sid: str, name: str):
    path = (P.SESSIONS / sid / name).resolve()
    if P.SESSIONS.resolve() not in path.parents or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/store/{name}")
def store_file(name: str):
    path = (STORE / name).resolve()
    if STORE.resolve() not in path.parents or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/sound/{move_id:path}")
def sound_file(move_id: str):
    p = _sound_path(move_id)
    if not p:
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/api/sounds")
def sounds():
    return sorted(m for m, e in catalog_mod.load().items() if e.get("sound"))


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
