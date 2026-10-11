"""Build (and optionally upload) the Hugging Face Space that makes meme_mirror installable from the Reachy Mini
app store / dashboard.

    .venv/bin/python scripts/publish_space.py                       # stage into export/space/ and list files
    .venv/bin/python scripts/publish_space.py --push                # ...and upload to the Space (public)

The Space holds the installable package (pyproject.toml + meme_mirror/), its license files, an app-store README
(tagged reachy_mini_python_app) and the showcase page as its landing page.
"""

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "export" / "space"
SPACE = "JosephAusterweil/meme_mirror"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.local.toml")


def stage():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for f in ("pyproject.toml", "LICENSE", "NOTICE"):
        shutil.copy(REPO / f, OUT / f)
    shutil.copy(REPO / "scripts" / "space_README.md", OUT / "README.md")
    shutil.copytree(REPO / "meme_mirror", OUT / "meme_mirror", ignore=IGNORE)
    shutil.copy(REPO / "docs" / "index.html", OUT / "index.html")
    shutil.copytree(REPO / "docs" / "assets", OUT / "assets")
    return sorted(p.relative_to(OUT) for p in OUT.rglob("*") if p.is_file())


def push():
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(SPACE, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=SPACE, repo_type="space",
                      commit_message="Update meme_mirror", delete_patterns=["meme_mirror/**", "assets/**"])
    return f"https://huggingface.co/spaces/{SPACE}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    files = stage()
    size = sum((OUT / f).stat().st_size for f in files) / 1e6
    print(f"staged {len(files)} files ({size:.1f} MB) in {OUT}")
    for f in files:
        if not str(f).startswith(("meme_mirror/web/viewer/assets/meshes", "assets/clips")):
            print("  ", f)
    if a.push:
        print("uploaded:", push())
