"""Any program as the LLM (for experts): your own scripts, other CLIs, custom local pipelines.

    [llm.proposer]
    provider = "command"
    command = ["python", "/path/to/my_llm.py"]     # or a single shell string
    model = "my-model"                               # free label, used for caching / logs

The program gets one JSON object on stdin:
    {"system": str | null, "prompt": str, "schema": {JSON Schema}, "images": ["/abs/path.png", ...], "model": str}
and must print one JSON object (matching the schema) on stdout. Exit code != 0 is reported as an error.
"""

import json
import shlex
import subprocess

from .base import CachedProvider, LLMError
from .openai_compat import extract_json


class CommandProvider(CachedProvider):
    name = "command"

    def __init__(self, command, model="custom", timeout=600, cache=True):
        super().__init__(model, cache)
        self.command = shlex.split(command) if isinstance(command, str) else list(command)
        self.timeout = timeout

    def _raw(self, prompt, schema, schema_json, images, system):
        payload = json.dumps(dict(system=system, prompt=prompt, schema=schema_json,
                                  images=[str(p) for p in images], model=self.model))
        try:
            out = subprocess.run(self.command, input=payload, capture_output=True, text=True, timeout=self.timeout)
        except FileNotFoundError:
            raise LLMError(f"command not found: {self.command[0]}")
        except subprocess.TimeoutExpired:
            raise LLMError(f"command timed out after {self.timeout} s")
        if out.returncode != 0:
            raise LLMError(f"command failed (exit {out.returncode}): {out.stderr[-500:]}")
        try:
            return extract_json(out.stdout)
        except (ValueError, json.JSONDecodeError) as e:
            return {"_unparsed": out.stdout[:2000], "_error": str(e)}
