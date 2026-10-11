"""Meme images for the LLM. Claude only sees the first frame of an animated GIF/WebP, so animated memes
are turned into a contact sheet: up to MAX_FRAMES evenly spaced frames in reading order, each labelled
with its number and timestamp."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence

MAX_FRAMES = 8
CELL_W = 360
COLS = 4


def _font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def frame_info(path):
    """(n_frames, total_seconds) for an image; (1, 0) when it isn't animated."""
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
        if n <= 1:
            return 1, 0.0
        total = sum(f.info.get("duration", 100) for f in ImageSequence.Iterator(im)) / 1000
        return n, total


def contact_sheet(path, out_path):
    """Write a labelled grid of evenly spaced frames. Returns (n_shown, n_frames, total_seconds)."""
    with Image.open(path) as im:
        durations = [f.info.get("duration", 100) for f in ImageSequence.Iterator(im)]
        n = len(durations)
        starts = [sum(durations[:i]) / 1000 for i in range(n)]
        picks = sorted({round(i * (n - 1) / (min(MAX_FRAMES, n) - 1)) for i in range(min(MAX_FRAMES, n))})
        frames = []
        for i in picks:
            im.seek(i)
            f = im.convert("RGB")
            f = f.resize((CELL_W, max(1, round(f.height * CELL_W / f.width))))
            frames.append((i, f))
    cell_h = max(f.height for _, f in frames) + 30
    rows = (len(frames) + COLS - 1) // COLS
    cols = min(COLS, len(frames))
    sheet = Image.new("RGB", (cols * (CELL_W + 8) + 8, rows * (cell_h + 8) + 8), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(18)
    for k, (i, f) in enumerate(frames):
        x = 8 + (k % COLS) * (CELL_W + 8)
        y = 8 + (k // COLS) * (cell_h + 8)
        draw.text((x, y), f"{k + 1}.  t = {starts[i]:.1f} s", font=font, fill="black")
        sheet.paste(f, (x, y + 28))
    sheet.save(out_path)
    return len(frames), n, sum(durations) / 1000


def meme_inputs(path, work_dir):
    """(images, note) to give the interpreter for this meme file."""
    path = Path(path)
    n, total = frame_info(path)
    if n <= 1:
        return [path], ""
    sheet = Path(work_dir) / "meme_frames.png"
    if not sheet.exists():
        contact_sheet(path, sheet)
    shown = min(MAX_FRAMES, n)
    note = (f"\nThis meme is ANIMATED ({n} frames, {total:.1f} s per loop). The image shows {shown} evenly spaced "
            "frames in order (left to right, then top to bottom), each labelled with its time. Read it as a short "
            "clip: what changes over time, and when, is part of the meme. Let the 'arc' follow its beats.")
    return [sheet], note
