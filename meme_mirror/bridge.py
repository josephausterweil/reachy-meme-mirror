"""Use your computer's Claude Code login from meme_mirror running elsewhere (e.g. installed on the robot).

Runs a small OpenAI-compatible server on THIS computer that answers with your local `claude` CLI. It listens on
localhost only and requires a token; reach it from the robot through an SSH tunnel:

    python -m meme_mirror.bridge                                   # prints the token, serves 127.0.0.1:8099
    ssh -N -R 8099:localhost:8099 pollen@reachy-mini.local          # robot's localhost:8099 -> this computer

Then in the robot app's ⚙ LLM panel: "Other OpenAI-compatible server", address http://localhost:8099/v1,
key = the printed token, models "opus" (main) and "sonnet" (fast). For your own personal use only.
"""

import argparse
import base64
import json
import secrets
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Header, HTTPException

from .config import secret, set_secret
from .llm.base import LLMError
from .llm.claude_code import ClaudeCodeProvider

MODELS = ("opus", "sonnet", "haiku")
TOKEN_NAME = "BRIDGE_TOKEN"
api = FastAPI(title="meme_mirror Claude Code bridge")


def _token():
    tok = secret(TOKEN_NAME)
    if not tok:
        tok = "mmb-" + secrets.token_urlsafe(24)
        set_secret(TOKEN_NAME, tok)
    return tok


def _auth(authorization):
    if authorization != f"Bearer {_token()}":
        raise HTTPException(401, "bad or missing bridge token")


@api.get("/v1/models")
def models(authorization: str | None = Header(None)):
    _auth(authorization)
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in MODELS]}


@api.post("/v1/chat/completions")
def chat(body: dict, authorization: str | None = Header(None)):
    _auth(authorization)
    model = body.get("model", "sonnet")
    if model not in MODELS:
        raise HTTPException(400, f"model must be one of {MODELS}")
    system, texts, images = None, [], []
    with tempfile.TemporaryDirectory() as tmp:
        for msg in body.get("messages", []):
            content = msg.get("content")
            parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
            for p in parts:
                if p.get("type") == "text":
                    (texts if msg.get("role") != "system" else []).append(p["text"])
                    if msg.get("role") == "system":
                        system = (system + "\n" if system else "") + p["text"]
                elif p.get("type") == "image_url":
                    url = p["image_url"]["url"]
                    if not url.startswith("data:"):
                        raise HTTPException(400, "only inline (data:) images are supported")
                    head, data = url.split(",", 1)
                    ext = ".png" if "png" in head else ".jpg" if "jpeg" in head or "jpg" in head else ".webp"
                    path = Path(tmp) / f"img{len(images)}{ext}"
                    path.write_bytes(base64.b64decode(data))
                    images.append(path)
        rf = body.get("response_format") or {}
        schema = rf.get("json_schema", {}).get("schema") if rf.get("type") == "json_schema" else {"type": "object"}
        try:
            out = ClaudeCodeProvider(model=model, cache=False)._raw("\n\n".join(texts), None, schema, images, system)
        except LLMError as e:
            raise HTTPException(502, str(e))
    return {"object": "chat.completion", "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(out)}}]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8099)
    a = ap.parse_args()
    print(f"meme_mirror bridge on http://127.0.0.1:{a.port}/v1  token: {_token()}")
    uvicorn.run(api, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
