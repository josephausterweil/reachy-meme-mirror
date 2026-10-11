import json
import sys
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from pydantic import BaseModel

from meme_mirror.llm.command import CommandProvider
from meme_mirror.llm.openai_compat import OpenAICompatibleProvider, extract_json
from meme_mirror.llm.presets import PRESETS, role_config


class Answer(BaseModel):
    color: str


def _server(reject_schema):
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"data": [{"id": "m-vision"}, {"id": "m-text"}]}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append(dict(req, auth=self.headers.get("authorization")))
            rf = req.get("response_format", {}).get("type")
            if reject_schema and rf == "json_schema":
                self.send_response(400); self.end_headers(); self.wfile.write(b"response_format not supported")
                return
            reply = 'Sure!\n```json\n{"color": "red"}\n```'
            body = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
            self.send_response(200); self.end_headers(); self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


@pytest.mark.parametrize("reject_schema", [False, True])
def test_openai_compatible_negotiates_and_parses(tmp_path, reject_schema, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    srv, seen = _server(reject_schema)
    img = tmp_path / "x.png"
    from PIL import Image
    Image.new("RGB", (8, 8), "red").save(img)
    p = OpenAICompatibleProvider(model="m-vision", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                                 api_key_env="TEST_KEY", cache=False)
    assert p.complete("color?", Answer, images=[img], system="sys").color == "red"
    last = seen[-1]
    assert last["auth"] == "Bearer sk-test" and last["messages"][0] == {"role": "system", "content": "sys"}
    parts = last["messages"][1]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert p.structured == ("json_object" if reject_schema else "json_schema")
    assert p.list_models() == ["m-text", "m-vision"]
    srv.shutdown()


def test_extract_json_variants():
    assert extract_json('<think>{"no": 1}</think> here: {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_command_provider(tmp_path):
    script = tmp_path / "llm.py"
    script.write_text("import json,sys\nreq=json.load(sys.stdin)\n"
                      "assert 'color' in req['schema']['properties']\nprint(json.dumps({'color': 'blue'}))\n")
    p = CommandProvider([sys.executable, str(script)], cache=False)
    assert p.complete("color?", Answer).color == "blue"


def test_presets_and_overrides():
    assert role_config("judge", {"preset": "claude-code"}) == dict(provider="claude_code", model="sonnet")
    kw = role_config("proposer", {"preset": "ollama", "main_model": "big-vl"})
    assert kw == dict(provider="openai_compatible", model="big-vl", base_url="http://localhost:11434/v1")
    assert role_config("judge", {"preset": "ollama", "main_model": "big-vl"})["model"] == "big-vl"  # fast falls back
    with pytest.raises(ValueError):
        role_config("proposer", {"preset": "openai"})  # no model chosen yet
    expert = {"preset": "claude-code", "proposer": {"provider": "command", "command": "x", "model": "m"}}
    assert role_config("proposer", expert)["provider"] == "command"
    assert all(set(p) >= {"label", "provider", "help"} for p in PRESETS.values())


def test_toml_writer_roundtrip():
    from meme_mirror.config import dump_toml
    d = {"user": {"name": 'Jo "J"'}, "llm": {"preset": "ollama", "main_model": "a:b",
                                              "proposer": {"provider": "command", "command": ["python", "x.py"]}}}
    assert tomllib.loads(dump_toml(d)) == d
