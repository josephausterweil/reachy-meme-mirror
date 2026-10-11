"""Claude through the Claude Code CLI (`claude -p`): uses the user's Claude Code login, no API key.

Images are passed as absolute paths the model opens with the Read tool (the only tool enabled).
"""

import json
import subprocess
import tempfile
from pathlib import Path

from .base import CachedProvider, LLMError

WORKDIR = Path(tempfile.gettempdir()) / "meme_mirror_claude"  # neutral cwd: no project CLAUDE.md


def _cli_schema(node):
    """Drop keywords the CLI's strict JSON-Schema check rejects (pydantic still validates the reply)."""
    if isinstance(node, dict):
        return {k: _cli_schema(v) for k, v in node.items() if k != "discriminator"}
    if isinstance(node, list):
        return [_cli_schema(v) for v in node]
    return node


class ClaudeCodeProvider(CachedProvider):
    name = "claude_code"

    def __init__(self, model="sonnet", timeout=300, effort=None, cache=True):
        super().__init__(model, cache)
        self.timeout = timeout
        self.effort = effort
        WORKDIR.mkdir(exist_ok=True)

    def _raw(self, prompt, schema, schema_json, images, system):
        if images:
            prompt = ("First use the Read tool to look at these image files:\n"
                      + "\n".join(f"- {p}" for p in images) + "\n\n" + prompt)
        cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", self.model,
               "--json-schema", json.dumps(_cli_schema(schema_json)), "--tools", "Read",
               "--strict-mcp-config", "--no-session-persistence", "--setting-sources", ""]
        for d in sorted({str(p.parent) for p in images}):
            cmd += ["--add-dir", d]
        if system:
            cmd += ["--append-system-prompt", system]
        if self.effort:
            cmd += ["--effort", self.effort]
        try:
            out = subprocess.run(cmd, cwd=WORKDIR, stdin=subprocess.DEVNULL, capture_output=True,
                                 text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise LLMError(f"claude -p timed out after {self.timeout} s")
        except FileNotFoundError:
            raise LLMError("Claude Code (the `claude` command) isn't installed or isn't on PATH. Install and log in, "
                           "or click ⚙ LLM to choose another model.")
        try:
            res = json.loads(out.stdout)
        except json.JSONDecodeError:
            raise LLMError(f"claude -p failed (exit {out.returncode}): {out.stderr[-800:] or out.stdout[-800:]}")
        if res.get("is_error"):
            raise LLMError(f"claude -p error: {res.get('result')}")
        if res.get("structured_output") is not None:
            return res["structured_output"]
        try:  # fall back to JSON in the text result
            return json.loads(res.get("result", ""))
        except json.JSONDecodeError:
            return {"_unparsed": res.get("result")}
