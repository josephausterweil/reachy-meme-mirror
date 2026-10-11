"""Ready-made LLM setups. Pick one in the web UI (⚙ LLM) or in config.local.toml:

    [llm]
    preset = "ollama"
    main_model = "qwen2.5vl:32b"     # reads memes + writes expressions (must accept images)
    fast_model = "qwen2.5vl:7b"      # blind judge, grader, lessons (images too); "" = use main_model

Experts can still override any single role with a full [llm.<role>] table (provider = ..., model = ...);
roles: interpreter, proposer, judge, grader, lessons, cataloger.
"""

# Which model tier each role uses. Every tier must accept images except where noted.
ROLE_TIER = {
    "interpreter": "main_model",   # image
    "proposer": "main_model",      # text only (long, structured output)
    "judge": "fast_model",         # image
    "grader": "fast_model",        # text only
    "lessons": "fast_model",       # text only
    "cataloger": "fast_model",     # image
}

PRESETS = {
    "claude-code": dict(
        label="Claude (Claude Code login)", provider="claude_code",
        main_model="opus", fast_model="sonnet",
        help="Uses the `claude` command-line tool and whoever is logged in to it. Nothing to paste here: "
             "install Claude Code and run `claude` once to log in."),
    "anthropic": dict(
        label="Claude (Anthropic API key)", provider="anthropic_api", api_key_env="ANTHROPIC_API_KEY",
        main_model="claude-opus-5-5", fast_model="claude-sonnet-5-5",
        help="Paste an API key from console.anthropic.com (billed per use)."),
    "openai": dict(
        label="OpenAI (API key)", provider="openai_compatible", base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY", list_models=True,
        help="Paste an API key from platform.openai.com, then pick vision-capable models from the list."),
    "gemini": dict(
        label="Google Gemini (API key)", provider="openai_compatible",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai", api_key_env="GEMINI_API_KEY",
        list_models=True,
        help="Paste an API key from aistudio.google.com, then pick models from the list."),
    "openrouter": dict(
        label="OpenRouter (one key, many models)", provider="openai_compatible",
        base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY", list_models=True,
        help="One key for models from many companies (openrouter.ai). Pick models that accept images."),
    "ollama": dict(
        label="Ollama (local, free)", provider="openai_compatible", base_url="http://localhost:11434/v1",
        list_models=True,
        help="Runs models on your own computer. Install Ollama, then e.g. `ollama pull qwen2.5vl:7b`. "
             "Pick a vision model; bigger models give better expressions."),
    "lmstudio": dict(
        label="LM Studio (local, free)", provider="openai_compatible", base_url="http://localhost:1234/v1",
        list_models=True,
        help="Start LM Studio's local server with a vision model loaded, then pick it from the list."),
    "custom": dict(
        label="Other OpenAI-compatible server", provider="openai_compatible", base_url="",
        api_key_env="CUSTOM_LLM_API_KEY", list_models=True,
        help="vLLM, llama.cpp llama-server, LocalAI, Groq, Mistral, DeepSeek, Together, xAI, ...: "
             "enter the server's /v1 address (and a key if it needs one)."),
}

DEFAULT_PRESET = "claude-code"


def role_config(role, llm_cfg):
    """Provider kwargs for a role: an explicit [llm.<role>] table wins, else the preset + chosen models."""
    explicit = llm_cfg.get(role)
    if isinstance(explicit, dict) and explicit.get("provider"):
        return dict(explicit)
    name = llm_cfg.get("preset") or DEFAULT_PRESET
    p = PRESETS[name]
    tier = ROLE_TIER.get(role, "main_model")
    model = llm_cfg.get(tier) or llm_cfg.get("main_model") or p.get(tier) or p.get("main_model")
    if not model:
        raise ValueError(f"LLM preset '{name}' needs a model: choose one in ⚙ LLM settings")
    kw = dict(provider=p["provider"], model=model)
    if p["provider"] == "openai_compatible":
        kw["base_url"] = llm_cfg.get("base_url") or p["base_url"]
        if not kw["base_url"]:
            raise ValueError("enter the server address in ⚙ LLM settings")
    if p.get("api_key_env"):
        kw["api_key_env"] = p["api_key_env"]
    return kw
