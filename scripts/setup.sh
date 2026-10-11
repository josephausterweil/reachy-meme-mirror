#!/usr/bin/env bash
# One-time setup: Python environment, CPU PyTorch, and binhpham's motion generator.
# Usage (from the repo root):  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GEN_REPO=https://github.com/pham-tuan-binh/reachy-motion-generator
GEN_COMMIT=96497d69a31cc89da209f224bd9c1fee1834b864   # version meme_mirror was tested with

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  PIP=(uv pip install --python .venv/bin/python)
else
  python3.12 -m venv .venv || python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  PIP=(.venv/bin/python -m pip install)
fi

echo "== meme_mirror and its dependencies"
"${PIP[@]}" -e ".[dev]"
echo "== PyTorch (CPU build is enough: the motion generator is small)"
"${PIP[@]}" torch --index-url https://download.pytorch.org/whl/cpu

echo "== binhpham's motion generator (Apache-2.0)"
if [ ! -d third_party/reachy-motion-generator ]; then
  git clone "$GEN_REPO" third_party/reachy-motion-generator
fi
git -C third_party/reachy-motion-generator checkout -q "$GEN_COMMIT"
"${PIP[@]}" --no-deps -e third_party/reachy-motion-generator

cat <<'EOF'

Done. Next:
  1. (optional) copy your name into config.local.toml:   printf '[user]\nname = "Sam"\n' > config.local.toml
  2. start the app:   .venv/bin/uvicorn meme_mirror.server:app --port 8770
  3. open http://localhost:8770, click "⚙ LLM" to pick an AI model, then drop in a meme.
EOF
