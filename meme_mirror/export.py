"""Package saved expressions (store/) as a Hugging Face dataset in the Pollen/Marionette moves layout.

    python -m meme_mirror.export --repo YOUR_HF_NAME/reachy-meme-expressions            # build export/ only
    python -m meme_mirror.export --repo YOUR_HF_NAME/reachy-meme-expressions --push     # ...and upload it

The dataset is tagged `reachy_mini_community_moves`, so it shows up in Pollen's community moves gallery and
any Reachy Mini can play it:  RecordedMoves("YOUR_HF_NAME/reachy-meme-expressions").get("<name>")

Only the motion, its sound and a short public description are exported: personal notes, ratings and the
meme image itself stay local (memes are usually someone else's copyrighted image).
"""

import argparse
import json
import shutil
from pathlib import Path

from .config import ROOT

STORE = ROOT / "store"
OUT = ROOT / "export"

CARD = """---
license: apache-2.0
pretty_name: {pretty}
tags:
  - reachy_mini
  - reachy_mini_community_moves
task_categories:
  - robotics
---

# {pretty}

Expressive motions for [Reachy Mini](https://github.com/pollen-robotics/reachy_mini), each made to convey the
feeling of a meme. They were made with **meme_mirror**: an LLM reads the meme, proposes several expressions
(recorded library moves, keyframed motion, or recipes rendered by
[binhpham's motion generator](https://github.com/pham-tuan-binh/reachy-motion-generator)), and a person picks,
rates and refines them.

```python
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves

moves = RecordedMoves("{repo}")
with ReachyMini() as mini:
    mini.play_move(moves.get("{first}"), sound=True)
```

| move | feeling | seconds |
|---|---|---|
{rows}

Sounds are from [pollen-robotics/reachy-mini-emotions-library](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library) (Apache-2.0).
"""


def build(repo):
    metas = sorted(STORE.glob("*.meta.json"))
    if not metas:
        raise SystemExit("store/ is empty: save some expressions in the web UI first")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data").mkdir(parents=True)
    rows = []
    for mp in metas:
        meta = json.loads(mp.read_text())
        slug = meta["slug"]
        move = json.loads((STORE / f"{slug}.json").read_text())
        feeling = meta.get("core_emotion", "")
        move["description"] = f"{meta['name']}: {feeling}"  # public description only
        (OUT / "data" / f"{slug}.json").write_text(json.dumps(move))
        for ext in (".ogg", ".wav"):
            if (STORE / f"{slug}{ext}").exists():
                shutil.copy(STORE / f"{slug}{ext}", OUT / "data" / f"{slug}{ext}")
        rows.append(f"| `{slug}` | {feeling} | {move['time'][-1]:.1f} |")
    pretty = repo.split("/")[-1].replace("-", " ").title()
    (OUT / "README.md").write_text(CARD.format(pretty=pretty, repo=repo, first=json.loads(metas[0].read_text())["slug"],
                                               rows="\n".join(rows)))
    return [p.relative_to(OUT) for p in sorted(OUT.rglob("*")) if p.is_file()]


def push(repo):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo, repo_type="dataset",
                      commit_message="Update meme expressions", delete_patterns=["data/*"])
    return f"https://huggingface.co/datasets/{repo}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. YOUR_HF_NAME/reachy-meme-expressions")
    ap.add_argument("--push", action="store_true", help="upload (public) after building export/")
    a = ap.parse_args()
    files = build(a.repo)
    print(f"built {OUT} ({len(files)} files):")
    for f in files:
        print("  ", f)
    if a.push:
        print("uploaded:", push(a.repo))
