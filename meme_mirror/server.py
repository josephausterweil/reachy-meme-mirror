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

from .config import HOME

STORE = HOME / "store"
WEB = Path(__file__).resolve().parent / "web"

app = FastAPI(title="meme_mirror")


@app.exception_handler(Exception)
async def _readable_errors(request, exc):
    """Model/setup problems reach the page as a readable message instead of 'Internal Server Error'."""
    from fastapi.responses import JSONResponse

    from .llm import LLMError

    if isinstance(exc, LLMError):
        return JSONResponse({"detail": f"AI model problem: {exc}"}, status_code=502)
    return JSONResponse({"detail": f"{exc.__class__.__name__}: {str(exc)[:300]}"}, status_code=500)
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
    picked: bool | None = None   # 'good fit' toggle
    played: bool | None = None


@app.post("/api/sessions/{sid}/candidates/{cid}/feedback")
def candidate_feedback(sid: str, cid: str, body: Feedback):
    with _lock(sid):
        s = _session(sid)
        P.feedback(s, cid, rating=body.rating, comment=body.comment, picked=body.picked,
                   played=body.played)
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


def sound_mode():
    """Where move sounds play: robot | computer (where this server runs) | browser | off.
    'auto' (default) = the robot's speaker when we run on the robot as an app, else this computer."""
    from . import robot
    from .config import settings

    mode = settings().get("robot", {}).get("sound", "auto")
    if mode == "auto":
        return "robot" if robot.attached() else "computer"
    return mode


@app.post("/api/sessions/{sid}/candidates/{cid}/play")
def candidate_play(sid: str, cid: str, body: PlayRequest):
    """Starts the move and answers as soon as the motion itself starts (glide-in done), with its duration.
    The glide back to neutral afterwards runs in the background; the robot lock is held until it's done."""
    from . import robot

    s = _session(sid)
    c = s.candidate(cid)
    if not c.get("move_file"):
        raise HTTPException(400, "candidate has no move (build failed)")
    if not _robot_lock.acquire(timeout=3):  # a click during the previous move's glide home waits for it
        raise HTTPException(409, "Reachy is already playing something")
    move = json.loads((s.dir / c["move_file"]).read_text())
    mode = sound_mode() if body.sound else "off"
    sound = _sound_path(c.get("sound")) if mode != "off" else None
    started, error = threading.Event(), {}

    def on_start():
        if sound and mode == "computer":
            _play_sound(sound)
        elif sound and mode == "robot":
            robot.play_sound_on_robot(sound)
        started.set()

    def worker():
        try:
            robot.play(move, sleep_after=body.sleep_after, return_home=True, on_start=on_start)
        except Exception as e:
            error["e"] = e
        finally:
            started.set()
            _robot_lock.release()

    threading.Thread(target=worker, daemon=True).start()
    started.wait(timeout=30)
    if "e" in error:
        e = error["e"]
        raise HTTPException(409 if isinstance(e, robot.RobotBusy) else 502,
                            str(e) if isinstance(e, robot.RobotBusy) else f"robot error: {e}")
    return dict(ok=True, duration_s=round(move["time"][-1] - move["time"][0], 2),
                sound_url=f"/sound/{c['sound']}" if sound and mode == "browser" else None)


@app.post("/api/robot/sleep")
def robot_sleep():
    from . import robot

    if not _robot_lock.acquire(timeout=5):
        raise HTTPException(409, "Reachy is busy")
    try:
        robot.check_free()
        robot.go_to_sleep()
    except robot.RobotBusy as e:
        raise HTTPException(409, str(e))
    finally:
        _robot_lock.release()
    return dict(ok=True)


@app.on_event("shutdown")
def _close_robot():
    from . import robot

    robot.close()


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
        P.feedback(_session(sid), cid, saved_as=slug)
    return meta


@app.get("/api/learning")
def learning_summary(lessons: bool = True):
    """What the proposer learns from: per-source stats, the suggested mix, and distilled lessons."""
    from . import learning

    return dict(stats=learning.source_stats(), suggested_mix=learning.suggested_mix(),
                n_rated=len(learning.rated_candidates()),
                lessons=learning.lessons() if lessons else None)


# ---------- optional PyTorch for learned ("binh") motion ----------

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_torch_job = {"state": "idle", "log": ""}


def _install_cmd():
    import sys

    uv = shutil.which("uv") or next((p for p in ("/opt/uv/uv",) if Path(p).exists()), None)
    if uv:
        return [uv, "pip", "install", "--no-cache", "--python", sys.executable, "torch", "--index-url", TORCH_CPU_INDEX]
    return [sys.executable, "-m", "pip", "install", "--no-cache-dir", "torch", "--index-url", TORCH_CPU_INDEX]


@app.get("/api/binh")
def binh_status():
    from . import binh

    return dict(available=binh.available(), **_torch_job)


@app.post("/api/binh/install")
def binh_install():
    """Install the CPU build of PyTorch into this app's own Python (about 0.9 GB on disk)."""
    import importlib
    import sys

    from . import binh

    if binh.available():
        return binh_status()
    if _torch_job["state"] == "installing":
        return binh_status()
    free_gb = shutil.disk_usage(sys.prefix).free / 1e9
    if free_gb < 2.0:
        raise HTTPException(507, f"only {free_gb:.1f} GB free; PyTorch needs about 1 GB plus headroom")

    def run():
        _torch_job.update(state="installing", log="")
        p = subprocess.run(_install_cmd(), capture_output=True, text=True)
        importlib.invalidate_caches()
        ok = p.returncode == 0 and binh.available()
        _torch_job.update(state="done" if ok else "failed", log=(p.stdout + p.stderr)[-1500:])

    threading.Thread(target=run, daemon=True).start()
    return binh_status()


# ---------- LLM settings ----------

@app.get("/api/llm")
def llm_settings():
    from .config import secret, settings
    from .llm.presets import PRESETS, ROLE_TIER

    cfg = settings().get("llm", {})
    presets = {}
    for name, p in PRESETS.items():
        env = p.get("api_key_env")
        presets[name] = dict(label=p["label"], help=p["help"], needs_key=bool(env) and name != "custom",
                             key_optional=name == "custom", has_key=bool(env and secret(env)),
                             list_models=bool(p.get("list_models")), base_url=p.get("base_url"),
                             main_model=p.get("main_model", ""), fast_model=p.get("fast_model", ""))
    overrides = [r for r in ROLE_TIER if isinstance(cfg.get(r), dict) and cfg[r].get("provider")]
    preset = cfg.get("preset") or "claude-code"
    problem = None  # a quick readiness check (no model call): is the chosen setup even possible here?
    p = PRESETS.get(preset, {})
    if preset == "claude-code" and not shutil.which("claude"):
        problem = "Claude Code isn't installed here. Choose another provider (an API key, or a local model server)."
    elif p.get("api_key_env") and preset != "custom" and not secret(p["api_key_env"]):
        problem = f"{p['label']} needs an API key."
    elif p.get("list_models") and not (cfg.get("main_model") or p.get("main_model")):
        problem = "Choose a model."
    return dict(preset=preset, main_model=cfg.get("main_model", ""),
                fast_model=cfg.get("fast_model", ""), base_url=cfg.get("base_url", ""),
                presets=presets, overrides=overrides, ready=problem is None, problem=problem)


class LLMChoice(BaseModel):
    preset: str
    main_model: str = ""
    fast_model: str = ""
    base_url: str = ""
    api_key: str | None = None   # only sent when the user typed one; stored in secrets.local.toml


@app.post("/api/llm")
def save_llm(body: LLMChoice):
    from .config import set_secret, update_local
    from .llm.presets import PRESETS

    if body.preset not in PRESETS:
        raise HTTPException(400, f"unknown preset {body.preset}")
    env = PRESETS[body.preset].get("api_key_env")
    if body.api_key is not None and env:
        set_secret(env, body.api_key.strip())
    update_local({"llm": dict(preset=body.preset, main_model=body.main_model.strip(),
                              fast_model=body.fast_model.strip(), base_url=body.base_url.strip())})
    return llm_settings()


@app.get("/api/llm/models")
def llm_models(preset: str, base_url: str = ""):
    from .llm.openai_compat import OpenAICompatibleProvider
    from .llm.presets import PRESETS

    p = PRESETS.get(preset)
    if not p or not p.get("list_models"):
        return dict(models=[])
    url = base_url or p.get("base_url")
    if not url:
        raise HTTPException(400, "enter the server address first")
    try:
        prov = OpenAICompatibleProvider(model="list", base_url=url, api_key_env=p.get("api_key_env"), cache=False)
        return dict(models=prov.list_models())
    except Exception as e:
        raise HTTPException(502, f"could not list models at {url}: {e.__class__.__name__}: {str(e)[:200]}")


class _ColorAnswer(BaseModel):
    color: str


@app.post("/api/llm/test")
def llm_test():
    """One tiny image question per model tier, to check the setup end to end."""
    import time as _time

    from PIL import Image

    from .llm import LLMError, get_provider

    img = Path(tempfile.gettempdir()) / "meme_mirror_llm_test.png"
    Image.new("RGB", (64, 64), (220, 30, 30)).save(img)
    results = {}
    for tier, role in (("main", "interpreter"), ("fast", "judge")):
        t0 = _time.time()
        try:
            prov = get_provider(role)
            prov.cache = False
            ans = prov.complete("What color is this square? One word.", _ColorAnswer, images=[img])
            ok = "red" in ans.color.lower()
            results[tier] = dict(ok=ok, model=prov.model, provider=prov.name, seconds=round(_time.time() - t0, 1),
                                 detail=f"answered '{ans.color}'" + ("" if ok else " (expected red: can it see images?)"))
        except (LLMError, Exception) as e:
            results[tier] = dict(ok=False, detail=str(e)[:300], seconds=round(_time.time() - t0, 1))
    return results


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
