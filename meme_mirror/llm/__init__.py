"""Swappable LLM backends, chosen per role in config.toml.

    [llm.interpreter]          # role name
    provider = "claude_code"   # claude_code | anthropic_api | qwen_local
    model = "opus"
    # any other keys are passed to the provider's constructor (timeout, effort, ...)
"""

import tomllib
from pathlib import Path

from .base import LLMError, Provider  # noqa: F401

CONFIG = Path(__file__).resolve().parents[2] / "config.toml"


def _providers():
    from .anthropic_api import AnthropicAPIProvider
    from .claude_code import ClaudeCodeProvider
    from .qwen_local import QwenLocalProvider

    return {p.name: p for p in (ClaudeCodeProvider, AnthropicAPIProvider, QwenLocalProvider)}


def get_provider(role: str, config_path: Path = CONFIG) -> Provider:
    cfg = tomllib.loads(config_path.read_text()).get("llm", {})
    opts = dict(cfg.get(role) or cfg.get("default") or {"provider": "claude_code"})
    name = opts.pop("provider")
    return _providers()[name](**opts)
