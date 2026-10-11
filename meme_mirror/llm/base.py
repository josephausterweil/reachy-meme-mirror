"""Provider interface: every LLM backend turns (prompt, images, schema) into a validated pydantic object."""

import hashlib
import json
import time
from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

RUNS = Path(__file__).resolve().parents[2] / "runs"


class LLMError(RuntimeError):
    pass


class Provider(Protocol):
    name: str
    model: str

    def complete(self, prompt: str, schema: type[T], images: list[Path] = (),
                 system: str | None = None) -> T: ...


class CachedProvider:
    """Shared plumbing: input-hash cache, call log, one retry that shows the model its validation error.

    Subclasses implement `_raw(prompt, schema, schema_json, images, system) -> dict`.
    """

    name = "base"

    def __init__(self, model: str, cache: bool = True):
        self.model = model
        self.cache = cache
        (RUNS / "llm_cache").mkdir(parents=True, exist_ok=True)

    def _raw(self, prompt, schema, schema_json, images, system) -> dict:
        raise NotImplementedError

    def _key(self, prompt, schema_json, images, system):
        h = hashlib.sha256()
        for part in (self.name, self.model, system or "", prompt, json.dumps(schema_json, sort_keys=True)):
            h.update(part.encode())
        for img in images:
            h.update(Path(img).read_bytes())
        return h.hexdigest()[:24]

    def complete(self, prompt, schema, images=(), system=None):
        images = [Path(i).resolve() for i in images]
        schema_json = schema.model_json_schema()
        key = self._key(prompt, schema_json, images, system)
        cached = RUNS / "llm_cache" / f"{key}.json"
        if self.cache and cached.exists():
            return schema.model_validate(json.loads(cached.read_text()))

        attempt_prompt = prompt
        for attempt in range(2):
            t0 = time.time()
            raw = self._raw(attempt_prompt, schema, schema_json, images, system)
            self._log(key, attempt, prompt, images, raw, time.time() - t0)
            try:
                obj = schema.model_validate(raw)
            except ValidationError as e:
                attempt_prompt = (f"{prompt}\n\nYour previous answer was invalid:\n{e}\n"
                                  "Answer again, following the schema exactly.")
                continue
            cached.write_text(obj.model_dump_json())
            return obj
        raise LLMError(f"{self.name}: invalid output after retry (see runs/llm_calls.jsonl, key {key})")

    def _log(self, key, attempt, prompt, images, raw, seconds):
        with open(RUNS / "llm_calls.jsonl", "a") as fh:
            fh.write(json.dumps(dict(key=key, provider=self.name, model=self.model, attempt=attempt,
                                     seconds=round(seconds, 1), images=[str(i) for i in images],
                                     prompt=prompt[:2000], reply=raw)) + "\n")
