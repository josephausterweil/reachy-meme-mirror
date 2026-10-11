"""Swappable LLM backends. Most people pick a preset (presets.py, or ⚙ LLM in the web UI).

Providers: claude_code (local `claude` CLI login), anthropic_api, openai_compatible (OpenAI, Gemini, OpenRouter,
Ollama, LM Studio, vLLM, llama.cpp, ...), command (any program: JSON in on stdin, JSON out on stdout).
Experts can set any role directly in config.local.toml:

    [llm.proposer]
    provider = "openai_compatible"
    base_url = "http://my-gpu-box:8000/v1"
    model = "my-finetune"
    # other keys go to the provider (timeout, max_tokens, temperature, structured, headers, ...)
"""

from ..config import settings
from .base import LLMError, Provider  # noqa: F401
from .presets import PRESETS, role_config  # noqa: F401


def _providers():
    from .anthropic_api import AnthropicAPIProvider
    from .claude_code import ClaudeCodeProvider
    from .command import CommandProvider
    from .openai_compat import OpenAICompatibleProvider

    return {p.name: p for p in (ClaudeCodeProvider, AnthropicAPIProvider, OpenAICompatibleProvider, CommandProvider)}


def get_provider(role: str) -> Provider:
    """Provider for a role: an explicit [llm.<role>] table, else the chosen preset (see presets.py)."""
    try:
        opts = role_config(role, settings().get("llm", {}))
    except (KeyError, ValueError) as e:
        raise LLMError(str(e))
    name = opts.pop("provider")
    return _providers()[name](**opts)
