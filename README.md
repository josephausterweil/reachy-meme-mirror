# meme_mirror

Turn memes into Reachy Mini expressions. Plan: ~/.claude/plans/let-s-make-a-systematic-adaptive-heron.md

## Setup
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -e ".[dev]"
    uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
    git clone https://github.com/pham-tuan-binh/reachy-motion-generator third_party/reachy-motion-generator   # tested at 96497d6
    uv pip install --python .venv/bin/python scipy transformers
    uv pip install --python .venv/bin/python --no-deps -e third_party/reachy-motion-generator

## Tools so far
- `python -m meme_mirror.preview.sim_render move.json out.mp4`: offscreen MuJoCo render (+ `out_strip.png`).
- binhpham generator on the CPU from Claude-written recipes:
  `python third_party/reachy-motion-generator/pipeline.py --recipes recipes.json --out DIR --no-render --ckpt /nonexistent`
