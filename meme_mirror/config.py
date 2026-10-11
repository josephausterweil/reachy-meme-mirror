"""Settings and where data lives.

Defaults ship in the package (defaults.toml); your own settings go in config.local.toml in the data folder.

Data folder (HOME): $MEME_MIRROR_HOME if set; else the repo root when running from a git checkout;
else ~/.meme_mirror (e.g. when installed as a Reachy Mini app). It holds runs/, store/, export/, cache/,
config.local.toml and secrets.local.toml.
"""

import os
import tomllib
from functools import lru_cache
from pathlib import Path

PKG = Path(__file__).resolve().parent
DEV = (PKG.parent / "pyproject.toml").exists() and (PKG.parent / ".git").exists()
if os.environ.get("MEME_MIRROR_HOME"):
    HOME = Path(os.environ["MEME_MIRROR_HOME"]).expanduser()
elif DEV:
    HOME = PKG.parent
else:
    HOME = Path.home() / ".meme_mirror"
HOME.mkdir(parents=True, exist_ok=True)
ROOT = HOME  # backwards-compatible name


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@lru_cache(maxsize=1)
def settings():
    cfg = tomllib.loads((PKG / "defaults.toml").read_text())
    local = HOME / "config.local.toml"
    if local.exists():
        cfg = _merge(cfg, tomllib.loads(local.read_text()))
    return cfg


def user_name():
    """How prompts refer to the person rating expressions (e.g. 'Sam'); default 'the user'."""
    return settings().get("user", {}).get("name") or "the user"


def robot_host():
    return settings().get("robot", {}).get("host", "reachy-mini.local")


SECRETS = HOME / "secrets.local.toml"  # API keys, git-ignored, readable only by you


def secret(name):
    """An API key: environment variable first, then secrets.local.toml."""
    import os

    if os.environ.get(name):
        return os.environ[name]
    if SECRETS.exists():
        return tomllib.loads(SECRETS.read_text()).get(name)
    return None


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def dump_toml(d, prefix=""):
    """Minimal TOML writer for nested tables of scalars/lists (enough for our config files)."""
    lines = [f"{k} = {_toml_value(v)}" for k, v in d.items() if not isinstance(v, dict)]
    for k, v in d.items():
        if isinstance(v, dict):
            name = f"{prefix}{k}"
            body = dump_toml(v, name + ".")
            if any(not isinstance(x, dict) for x in v.values()):
                lines.append(f"\n[{name}]")
            lines.append(body)
    return "\n".join(x for x in lines if x)


def update_local(updates):
    """Merge `updates` into config.local.toml (None values delete keys)."""
    path = HOME / "config.local.toml"
    cur = tomllib.loads(path.read_text()) if path.exists() else {}

    def apply(dst, src):
        for k, v in src.items():
            if v is None:
                dst.pop(k, None)
            elif isinstance(v, dict):
                apply(dst.setdefault(k, {}), v)
            else:
                dst[k] = v
    apply(cur, updates)
    path.write_text(dump_toml(cur).lstrip() + "\n")
    settings.cache_clear()


def set_secret(name, value):
    import os

    cur = tomllib.loads(SECRETS.read_text()) if SECRETS.exists() else {}
    if value:
        cur[name] = value
    else:
        cur.pop(name, None)
    fd = os.open(SECRETS, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # private from the start
    with os.fdopen(fd, "w") as fh:
        fh.write(dump_toml(cur) + "\n")
    os.chmod(SECRETS, 0o600)
