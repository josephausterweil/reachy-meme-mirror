"""Claude through the Anthropic API (`pip install anthropic`; credentials from ANTHROPIC_API_KEY or `ant auth login`).

Not used yet (no key configured); select it in config.toml when wanted.
"""

import base64
import mimetypes

from .base import CachedProvider, LLMError


class AnthropicAPIProvider(CachedProvider):
    name = "anthropic_api"

    def __init__(self, model="claude-opus-5-5", effort="medium", max_tokens=16000, cache=True):
        super().__init__(model, cache)
        import anthropic

        self.client = anthropic.Anthropic()
        self.effort = effort
        self.max_tokens = max_tokens

    def _raw(self, prompt, schema, schema_json, images, system):
        content = []
        for p in images:
            media_type = mimetypes.guess_type(p)[0] or "image/png"
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": media_type,
                "data": base64.standard_b64encode(p.read_bytes()).decode()}})
        content.append({"type": "text", "text": prompt})
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      messages=[{"role": "user", "content": content}],
                      output_format=schema, output_config={"effort": self.effort})
        if system:
            kwargs["system"] = system
        response = self.client.messages.parse(**kwargs)
        if response.stop_reason == "refusal":
            raise LLMError(f"refused: {response.stop_details}")
        if response.parsed_output is None:
            raise LLMError(f"no parsed output (stop_reason={response.stop_reason})")
        return response.parsed_output.model_dump()
