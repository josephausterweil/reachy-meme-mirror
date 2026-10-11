"""Local Qwen3-VL placeholder, for when the GPU is free.

audio_helper/vision.py already runs Qwen/Qwen3-VL-4B-Instruct (it needs >= 11 GB free VRAM); a real
implementation would reuse that loader, ask for JSON matching the schema, and return the parsed dict.
"""

from .base import CachedProvider, LLMError


class QwenLocalProvider(CachedProvider):
    name = "qwen_local"

    def __init__(self, model="Qwen/Qwen3-VL-4B-Instruct", cache=True):
        super().__init__(model, cache)

    def _raw(self, prompt, schema, schema_json, images, system):
        raise LLMError("qwen_local is not implemented yet (GPU busy); use claude_code")
