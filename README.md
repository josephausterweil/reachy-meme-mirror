# meme_mirror

Turn memes into expressions for a [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) robot.

Drop a meme into a local web page. An AI model works out the feeling the meme expresses, then proposes
several ways Reachy could show it with its head, antennas and body. You compare them in a simulator, play
the best ones on the robot, rate them, and ask for better ones. Over time it learns your taste.

Each round mixes three kinds of candidates:

| tag | where the motion comes from |
|---|---|
| **library** | moves recorded by people: Pollen's emotions library and community moves |
| **own** | keyframes and rhythms written by the model, compiled by meme_mirror (can reuse library moves) |
| **binh** | a motion script written by the model, rendered by [binhpham's motion generator](https://github.com/pham-tuan-binh/reachy-motion-generator), which adds lifelike detail |

## What you need

- A Reachy Mini on your network. You can try everything except "Play on Reachy" without one.
- Linux or macOS with Python 3.12. No GPU needed.
- An AI model. Any of these works:
  - your own Claude Code login,
  - an API key from Anthropic, OpenAI, Google Gemini or OpenRouter,
  - a free local model through Ollama or LM Studio.

Privacy: your memes, notes and the prompts go to whichever AI provider you choose. With a local model
(Ollama, LM Studio, …) they never leave your computer.

## Setup

```bash
git clone https://github.com/josephausterweil/reachy-meme-mirror
cd reachy-meme-mirror
bash scripts/setup.sh
.venv/bin/uvicorn meme_mirror.server:app --port 8770
```

Open <http://localhost:8770>, click **⚙ LLM**, choose your AI model and click **Test**. Then drop in a meme.
The first run downloads the motion libraries from Hugging Face, which takes about a minute.

If your robot isn't reachable as `reachy-mini.local`, set its address in `config.local.toml`:

```toml
[robot]
host = "192.168.1.42"

[user]
name = "Sam"      # optional: how the prompts refer to you
```

## Using it

1. **Drop a meme** (PNG, JPG, WebP or animated GIF). Check the reading of what it expresses, and edit it or add
   notes about the feeling or how Reachy might show it.
2. **Propose expressions.** About 30 seconds later you get candidate cards with simulator videos, the reasoning
   behind each, and blind scores. A separate model watches each motion without seeing the meme and guesses the
   feeling; *blind match* is how close that guess is to the meme's feeling.
3. **Give feedback:** stars, a note, and **Good fit** for the ones that express the meme well. Use
   **▶ Play on Reachy** to see one on the robot; playing doesn't count as a vote.
4. **Ask for more:**
   - **Improve from my feedback** uses all your ratings and notes.
   - **More like this** builds on one card.
   - **Try new ways** explores different interpretations.
   - The **dials** change speed and size without asking the AI again.
5. **Save** the ones you like. They go to `store/` in the standard Reachy Mini moves format.

Your ratings carry over between memes. Later rounds see your best and worst past expressions, plus lessons
drawn from your notes, and the default mix shifts toward the kind of candidate you rate highest.

Only **▶ Play on Reachy** and **Sleep** move the robot. Playing is refused while another Reachy app is running.
All motion is clamped to the ranges Pollen's own recorded emotions use, and checked against the robot's
reachable poses.

## Choosing the AI model

The easy way is the **⚙ LLM** panel:

| preset | what you need |
|---|---|
| Claude (Claude Code login) | [Claude Code](https://claude.com/claude-code) installed and logged in (`claude`) |
| Claude (Anthropic API key) | a key from console.anthropic.com, and `pip install anthropic` |
| OpenAI | a key from platform.openai.com |
| Google Gemini | a key from aistudio.google.com |
| OpenRouter | one key for models from many companies |
| Ollama / LM Studio | the app running locally with a **vision** model, e.g. `ollama pull qwen2.5vl:7b` |
| Other OpenAI-compatible server | the server's `/v1` address: vLLM, llama.cpp `llama-server`, LocalAI, Groq, Mistral, DeepSeek, … |

There are two model slots. The **main** model reads memes and writes expressions. The **fast** model runs the blind
judge and lighter jobs. Both need to accept images. Bigger models write noticeably better motion; small local
models work, but expect more misses.

API keys are kept in `secrets.local.toml` on your computer (git-ignored, readable only by you), or read from
the usual environment variables (`OPENAI_API_KEY`, `GEMINI_API_KEY`, …).

### For experts: any model, per job

`config.local.toml` can point any single job at any backend. Jobs are `interpreter`, `proposer`, `judge`,
`grader`, `lessons` and `cataloger`.

```toml
[llm.proposer]                         # e.g. your fine-tuned model on a GPU box
provider = "openai_compatible"
base_url = "http://gpu-box:8000/v1"
model = "my-motion-model"
structured = "json_object"             # json_schema | json_object | prompt (default: auto-detect)
temperature = 0.7
headers = { "X-Team" = "robots" }

[llm.judge]                            # or any program at all
provider = "command"
command = ["python", "my_llm.py"]      # gets {"system","prompt","schema","images","model"} as JSON on stdin,
model = "my-judge"                     # prints one JSON object matching "schema" on stdout
```

Every reply is validated against a schema, and an invalid reply gets one automatic retry that shows the model
its mistake. Calls are logged in `runs/llm_calls.jsonl` and cached by input.

## Sharing expressions

Publish saved expressions as a Hugging Face dataset that any Reachy Mini can play. It's tagged so it appears in
Pollen's community moves gallery:

```bash
huggingface-cli login
.venv/bin/python -m meme_mirror.export --repo YOUR_HF_NAME/reachy-meme-expressions          # preview in export/
.venv/bin/python -m meme_mirror.export --repo YOUR_HF_NAME/reachy-meme-expressions --push   # publish (public)
```

Only the motion, its sound and a short description are published. Your notes, ratings and the meme images are
not part of the upload.

## How it works

`pipeline.py` runs each round:

1. **interpret**: a vision model reads the meme. An animated GIF is shown as a timed contact sheet of frames.
2. **propose**: two parallel calls, one writing specs (library + own) and one writing recipes (binh).
3. **realize**:
   - specs are compiled by `compiler.py` and recipes go to `binh.py`;
   - `safety.py` clamps the motion;
   - every frame is projected onto the robot's reachable poses;
   - `preview/sim_render.py` renders it offscreen in MuJoCo.
4. **judge**: a blind model reads 6 frames plus a motion summary, and a grader scores its guess against the
   intended feeling.

`catalog.yaml` describes the 115 library moves: features computed from the motion, and descriptions written by
a model from rendered frames. Add `notes:` or `exclude: true` to any entry; rebuilding keeps them.

Run the tests with `.venv/bin/python -m pytest`.

## Credits and license

Apache-2.0 (see `LICENSE` and `NOTICE`). This project builds on:

- Pollen Robotics' Reachy Mini SDK and emotions library,
- Pham Tuan Binh's reachy-motion-generator (recipe language, planner prompt and motion model),
- community moves by Anne-Charlotte and stellaaaa.

All are Apache-2.0.
