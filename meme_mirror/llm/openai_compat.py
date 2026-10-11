"""Any OpenAI-compatible chat-completions server: OpenAI, Google Gemini, OpenRouter, Groq, Mistral, DeepSeek,
xAI, Together, ... and local servers: Ollama, LM Studio, vLLM, llama.cpp `llama-server`, LocalAI.

Images are sent inline as data URLs. Structured output is negotiated: JSON-schema response_format if the
server accepts it, else JSON mode, else plain prompting (the reply is validated either way, with one retry).
"""

import base64
import json
import mimetypes
import os
import re

import requests

from .base import CachedProvider, LLMError

STRUCTURED_MODES = ("json_schema", "json_object", "prompt")


def _image_part(path):
    media = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}


def extract_json(text):
    """Pull the first JSON object out of a reply (handles ```json fences and chatter around it)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)  # reasoning models' scratchpad
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object in reply")


class OpenAICompatibleProvider(CachedProvider):
    name = "openai_compatible"

    def __init__(self, model, base_url="http://localhost:11434/v1", api_key=None, api_key_env=None,
                 structured="auto", timeout=600, max_tokens=16000, temperature=None, headers=None, cache=True):
        super().__init__(model, cache)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or (os.environ.get(api_key_env) if api_key_env else None)
        if not self.api_key and api_key_env:
            from ..config import secret

            self.api_key = secret(api_key_env)
        self.structured = structured
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.headers = headers or {}

    def _post(self, body):
        headers = {"content-type": "application/json", **self.headers}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
        except requests.RequestException as e:
            raise LLMError(f"cannot reach {self.base_url}: {e.__class__.__name__}")
        return r

    def _raw(self, prompt, schema, schema_json, images, system):
        instructions = ("\n\nReply with ONLY a JSON object that matches this JSON Schema (no prose, no markdown):\n"
                        + json.dumps(schema_json))
        content = [{"type": "text", "text": prompt + instructions}] + [_image_part(p) for p in images]
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": content if images else prompt + instructions}]
        modes = STRUCTURED_MODES if self.structured == "auto" else (self.structured,)
        last = None
        for mode in modes:
            body = dict(model=self.model, messages=messages, max_tokens=self.max_tokens)
            if self.temperature is not None:
                body["temperature"] = self.temperature
            if mode == "json_schema":
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": schema.__name__[:60], "schema": schema_json}}
            elif mode == "json_object":
                body["response_format"] = {"type": "json_object"}
            r = self._post(body)
            if r.status_code in (400, 404, 415, 422, 501) and mode != modes[-1]:
                last = r.text[:300]
                continue  # server doesn't support this response_format: try the next mode
            if r.status_code in (401, 403):
                raise LLMError(f"{self.base_url}: not authorized (check the API key)")
            if r.status_code >= 400:
                raise LLMError(f"{self.base_url}: HTTP {r.status_code}: {r.text[:300]}")
            if self.structured == "auto":
                self.structured = mode  # remember what works for this server
            choice = r.json()["choices"][0]
            text = choice["message"].get("content") or ""
            if isinstance(text, list):  # some servers return content parts
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            try:
                return extract_json(text)
            except (ValueError, json.JSONDecodeError) as e:
                return {"_unparsed": text[:2000], "_error": str(e)}
        raise LLMError(f"{self.base_url}: request rejected: {last}")

    def list_models(self):
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = requests.get(f"{self.base_url}/models", headers={**headers, **self.headers}, timeout=15)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))
