"""The chat model the agents reason with.

Everything goes through :func:`build_chat_model`, so swapping OpenAI for another
LangChain chat model later (Azure OpenAI, a local model via an OpenAI-compatible
endpoint, etc.) is a one-line change here — the agents and graph don't care.

Note: constructing ``ChatOpenAI`` does NOT call the network — only ``.invoke()``
does. That is why the graph can be *compiled* (and unit-tested) without a key.
"""

from __future__ import annotations

from .config import Settings, settings


class LLMNotConfigured(RuntimeError):
    """Raised when we try to run the agents without an OpenAI key."""


def build_chat_model(cfg: Settings | None = None):
    """Return a configured ChatOpenAI instance ready to bind tools to."""
    cfg = cfg or settings
    if not cfg.openai_api_key:
        raise LLMNotConfigured(
            "No OPENAI_API_KEY found. Copy .env.example to .env and add your key "
            "(you can reuse the same key as the other AIMP kits). The agents need "
            "an LLM to make decisions."
        )
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - import guard
        raise LLMNotConfigured(
            "langchain-openai is not installed. Run: pip install -r requirements.txt"
        ) from exc

    return ChatOpenAI(
        model=cfg.chat_model,
        temperature=cfg.temperature,
        api_key=cfg.openai_api_key,
        timeout=cfg.llm_timeout_seconds,
        max_retries=cfg.llm_max_retries,
    )
