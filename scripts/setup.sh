#!/usr/bin/env bash
# One-time setup: Python environment with simulator previews and CPU PyTorch.
# Usage (from the repo root):  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 .venv
  PIP=(uv pip install --python .venv/bin/python)
else
  python3.12 -m venv .venv || python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  PIP=(.venv/bin/python -m pip install)
fi

echo "== meme_mirror and its dependencies (with simulator previews)"
"${PIP[@]}" -e ".[dev,sim]"
echo "== PyTorch, CPU build (for learned 'binh' motion; the generator is small)"
"${PIP[@]}" torch --index-url https://download.pytorch.org/whl/cpu

cat <<'EOF'

Done. Next:
  1. (optional) copy your name into config.local.toml:   printf '[user]\nname = "Sam"\n' > config.local.toml
  2. start the app:   .venv/bin/uvicorn meme_mirror.server:app --port 8770
  3. open http://localhost:8770, click "⚙ LLM" to pick an AI model, then drop in a meme.
EOF
