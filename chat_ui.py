"""Popup-style chatbot UI for the Factory Agent workspace.

Run:
    streamlit run chat_ui.py
"""

from __future__ import annotations

import concurrent.futures
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from factory_agent import build_graph
from factory_agent.config import settings
from factory_agent.knowledge_rag import KNOWLEDGE_RAG
from factory_agent.manual_data import MANUAL_INDEX
from factory_agent.graph import _extract_entity
from factory_agent.llm import LLMNotConfigured, build_chat_model
from factory_agent.project_data import PROJECT_DATA
from factory_agent.tickets import TICKET_STORE
from factory_agent.tools import get_entity_full_context
from factory_agent.yield_data import YIELD_DATASET

CHECKPOINTER = MemorySaver()

NODE_LABELS = {
    "status_check": "Status check",
    "technician": "Agent Technician",
    "yield": "Agent Yield",
    "escalation": "Escalation",
    "fallback": "Fallback",
}

SAMPLE_DOWN_ENTITY = "TCB702"
SAMPLE_UP_ENTITY = "TSX509"


def _runtime_profile_presets() -> dict[str, dict[str, object]]:
    """Return runtime profile presets for response smoothness tuning."""
    fast_timeout = max(1.0, float(settings.ui_soft_timeout_seconds))
    rich_timeout = max(14.0, fast_timeout + 8.0)
    return {
        "fast": {
            "soft_timeout_seconds": fast_timeout,
            "manual_top_k": 2,
            "show_agent_trace": False,
        },
        "rich": {
            "soft_timeout_seconds": rich_timeout,
            "manual_top_k": 3,
            "show_agent_trace": True,
        },
    }


def _apply_runtime_profile(name: str, announce: bool = True) -> str:
    """Apply a runtime profile to session-scoped responsiveness knobs."""
    candidate = (name or "").strip().lower()
    presets = _runtime_profile_presets()
    if candidate not in presets:
        return "Invalid profile. Use /profile fast or /profile rich."

    preset = presets[candidate]
    st.session_state.runtime_profile = candidate
    st.session_state.soft_timeout_seconds = float(preset["soft_timeout_seconds"])
    st.session_state.manual_top_k = int(preset["manual_top_k"])
    st.session_state.show_agent_trace = bool(preset["show_agent_trace"])
    st.session_state.response_cache = {}

    if not announce:
        return ""

    trace_label = "on" if st.session_state.show_agent_trace else "off"
    return (
        f"Runtime profile switched to: {candidate}. "
        f"Soft timeout={st.session_state.soft_timeout_seconds:.0f}s, "
        f"local context top_k={st.session_state.manual_top_k}, "
        f"agent trace={trace_label}."
    )


def _set_trace_mode(mode: str) -> str:
    """Enable/disable trace panels without restarting the app."""
    candidate = (mode or "").strip().lower()
    if candidate in {"on", "1", "true", "yes"}:
        st.session_state.show_agent_trace = True
    elif candidate in {"off", "0", "false", "no"}:
        st.session_state.show_agent_trace = False
    elif candidate in {"toggle", ""}:
        st.session_state.show_agent_trace = not bool(
            st.session_state.get("show_agent_trace", settings.ui_show_agent_trace)
        )
    else:
        return "Invalid trace mode. Use /trace on, /trace off, or /trace toggle."

    state = "on" if st.session_state.show_agent_trace else "off"
    return f"Agent trace is now: {state}."


def _inject_theme() -> None:
    """Apply an industrial visual theme with clear hierarchy and spacing."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --tcb-ink: #0f2f3a;
            --tcb-muted: #49616b;
            --tcb-accent: #0f766e;
            --tcb-accent-2: #f28a14;
            --tcb-surface: #ffffff;
            --tcb-border: #cfe1df;
        }

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
            color: var(--tcb-ink);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 12% 10%, rgba(15, 118, 110, 0.16), transparent 34%),
                radial-gradient(circle at 88% 0%, rgba(242, 138, 20, 0.18), transparent 30%),
                linear-gradient(180deg, #f7fbfb 0%, #edf5f4 100%);
        }

        .main .block-container {
            max-width: 920px;
            padding-top: 1.25rem;
            padding-bottom: 1.25rem;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', 'IBM Plex Sans', sans-serif;
            letter-spacing: 0.01em;
        }

        .tcb-status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.25rem 0 0.8rem 0;
        }

        .tcb-chip {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid var(--tcb-border);
            border-radius: 999px;
            color: var(--tcb-ink);
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.18rem 0.62rem;
            backdrop-filter: blur(2px);
        }

        .tcb-chip strong {
            color: var(--tcb-accent);
        }

        .stButton > button {
            border-radius: 11px;
            border: 1px solid #bdd7d4;
            background: #ffffff;
            color: var(--tcb-ink);
            font-weight: 600;
            transition: all 0.18s ease;
        }

        .stButton > button:hover {
            border-color: var(--tcb-accent);
            color: var(--tcb-accent);
            box-shadow: 0 4px 14px rgba(15, 118, 110, 0.12);
            transform: translateY(-1px);
        }

        [data-testid="stChatInputTextArea"] textarea {
            border-radius: 12px;
            border: 1px solid #bedad6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _resolve_foundry_logo() -> Path | None:
    """Find a logo file to render for app branding."""
    candidates = [
        Path(settings.foundry_logo_path),
        Path(__file__).resolve().parent / "assets" / "foundry-logo.svg",
        Path(__file__).resolve().parent / "assets" / "foundry-logo.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _render_brand_header() -> None:
    col_logo, col_text = st.columns([1, 5])
    with col_logo:
        logo = _resolve_foundry_logo()
        if logo is not None:
            st.image(str(logo), width=210)
        else:
            st.markdown("### Foundry")
    with col_text:
        st.title(settings.app_title)
        st.caption(
            "Equipment Performance Sustaining — enter an entity and the agents "
            "check status, tickets, and yield, then escalate if needed."
        )


def _rag_state_label() -> str:
    status = KNOWLEDGE_RAG.status_text().lower()
    if "ready" in status:
        return "ready"
    if "not loaded yet" in status:
        return "lazy"
    return "degraded"


def _render_status_strip() -> None:
    """Show lightweight runtime state for transparency and faster debugging."""
    style = st.session_state.get("prompt_style", "base")
    profile = st.session_state.get("runtime_profile", "fast")
    timeout_seconds = float(
        st.session_state.get("soft_timeout_seconds", settings.ui_soft_timeout_seconds)
    )
    trace_on = bool(st.session_state.get("show_agent_trace", settings.ui_show_agent_trace))
    llm_state = "enabled" if settings.llm_enabled else "missing key"
    rag_state = _rag_state_label()
    cache_items = len(st.session_state.get("response_cache", {}))
    cache_hits = int(st.session_state.get("cache_hits", 0))
    cache_misses = int(st.session_state.get("cache_misses", 0))

    st.markdown(
        (
            "<div class='tcb-status-strip'>"
            f"<span class='tcb-chip'>Style: <strong>{style}</strong></span>"
            f"<span class='tcb-chip'>Profile: <strong>{profile}</strong></span>"
            f"<span class='tcb-chip'>Soft timeout: <strong>{timeout_seconds:.0f}s</strong></span>"
            f"<span class='tcb-chip'>Trace: <strong>{'on' if trace_on else 'off'}</strong></span>"
            f"<span class='tcb-chip'>LLM: <strong>{llm_state}</strong></span>"
            f"<span class='tcb-chip'>RAG: <strong>{rag_state}</strong></span>"
            f"<span class='tcb-chip'>UI cache: <strong>{cache_items}</strong> ({cache_hits}H/{cache_misses}M)</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_quick_actions() -> str | None:
    """Provide single-click prompts for common actions."""
    queued: str | None = None
    current_style = st.session_state.get("prompt_style", "base")
    switch_to = "natural" if current_style == "base" else "base"
    current_profile = st.session_state.get("runtime_profile", "fast")
    profile_target = "rich" if current_profile == "fast" else "fast"
    trace_on = bool(st.session_state.get("show_agent_trace", settings.ui_show_agent_trace))
    trace_target = "off" if trace_on else "on"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Try DOWN entity", key="quick_down"):
            queued = SAMPLE_DOWN_ENTITY
    with col2:
        if st.button("Try UP entity", key="quick_up"):
            queued = SAMPLE_UP_ENTITY
    with col3:
        if st.button("System status", key="quick_status"):
            queued = "/status"
    with col4:
        if st.button(f"Switch to {switch_to}", key="quick_style"):
            queued = f"/style {switch_to}"

    col5, col6 = st.columns(2)
    with col5:
        if st.button(f"Profile {profile_target}", key="quick_profile"):
            queued = f"/profile {profile_target}"
    with col6:
        if st.button(f"Trace {trace_target}", key="quick_trace"):
            queued = f"/trace {trace_target}"

    return queued


def _has_streamlit_context() -> bool:
    """True when the app is executed through `streamlit run`."""
    return (
        threading.current_thread().name == "ScriptRunner.scriptThread"
        or "streamlit.web.bootstrap" in sys.modules
    )


def _start_graph():
    """Build the LangGraph instance, returning (graph, error_message)."""
    if not settings.llm_enabled:
        return None, (
            "No OPENAI_API_KEY found in .env. Add it, then refresh this page "
            "to enable live answers."
        )
    try:
        style = st.session_state.get("prompt_style", "base")
        return build_graph(
            build_chat_model(),
            checkpointer=CHECKPOINTER,
            prompt_style=style,
        ), ""
    except LLMNotConfigured as exc:
        return None, str(exc)


def _reset_state() -> None:
    """Reset graph and chat history for a new session."""
    PROJECT_DATA.reload()
    YIELD_DATASET.reload()
    MANUAL_INDEX.reload()
    KNOWLEDGE_RAG.reload()
    TICKET_STORE.reset()
    st.session_state.chat_log = []
    st.session_state.response_cache = {}
    st.session_state.cache_hits = 0
    st.session_state.cache_misses = 0
    st.session_state.thread_id = str(uuid4())
    st.session_state.graph, st.session_state.startup_error = _start_graph()


def _init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
    if "prompt_style" not in st.session_state:
        st.session_state.prompt_style = (
            settings.prompt_style if settings.prompt_style in {"base", "natural"} else "base"
        )
    if "response_cache" not in st.session_state:
        st.session_state.response_cache = {}
    if "cache_hits" not in st.session_state:
        st.session_state.cache_hits = 0
    if "cache_misses" not in st.session_state:
        st.session_state.cache_misses = 0
    if "runtime_profile" not in st.session_state:
        st.session_state.runtime_profile = "fast"
    if "soft_timeout_seconds" not in st.session_state:
        st.session_state.soft_timeout_seconds = float(settings.ui_soft_timeout_seconds)
    if "manual_top_k" not in st.session_state:
        st.session_state.manual_top_k = 2
    if "show_agent_trace" not in st.session_state:
        st.session_state.show_agent_trace = bool(settings.ui_show_agent_trace)
    if st.session_state.get("initialized"):
        return
    st.session_state.initialized = True
    _reset_state()
    _apply_runtime_profile(st.session_state.runtime_profile, announce=False)


def _switch_prompt_style(style: str) -> str:
    """Switch prompt style and rebuild the graph in-place."""
    candidate = style.strip().lower()
    if candidate not in {"base", "natural"}:
        return "Invalid style. Use /style base or /style natural."
    st.session_state.prompt_style = candidate
    st.session_state.response_cache = {}
    st.session_state.graph, st.session_state.startup_error = _start_graph()
    return f"Prompt style switched to: {candidate}."


def _reload_runtime_data() -> str:
    """Reload data/indexes and refresh the graph without clearing chat history."""
    PROJECT_DATA.reload()
    YIELD_DATASET.reload()
    MANUAL_INDEX.reload()
    KNOWLEDGE_RAG.reload()
    st.session_state.response_cache = {}
    st.session_state.graph, st.session_state.startup_error = _start_graph()
    return "Runtime data and indexes reloaded."


def _deterministic_entity_reply(question: str, reason: str = "", manual_top_k: int = 2) -> str:
    """Return a fast local-data answer without any LLM call."""
    entity = _extract_entity(question)
    if not entity:
        note = f" ({reason})" if reason else ""
        return (
            "Please enter a valid entity code like TCB706 or TSX509."
            f"{note}"
        )

    top_k = max(1, min(int(manual_top_k), 4))
    context = get_entity_full_context.invoke({"entity": entity, "manual_top_k": top_k})
    lines = [f"Mode: deterministic local-data fallback{f' ({reason})' if reason else ''}", context]
    return "\n\n".join(lines)


def _normalize_prompt(prompt: str) -> str:
    return " ".join((prompt or "").strip().split()).lower()


def _cache_key(style: str, prompt: str) -> tuple[str, str]:
    return style.strip().lower(), _normalize_prompt(prompt)


def _cache_get(key: tuple[str, str]) -> dict | None:
    cache = st.session_state.response_cache
    cached = cache.get(key)
    if cached is None:
        st.session_state.cache_misses += 1
        return None

    st.session_state.cache_hits += 1
    cache.pop(key, None)
    cache[key] = cached
    result = {
        "reply": cached.get("reply", ""),
        "steps": [dict(step) for step in cached.get("steps", [])],
        "meta": dict(cached.get("meta", {})),
    }
    result["meta"]["cached"] = True
    return result


def _cache_put(key: tuple[str, str], result: dict) -> None:
    limit = max(0, int(settings.ui_response_cache_size))
    if limit <= 0:
        return

    cache = st.session_state.response_cache
    payload = {
        "reply": result.get("reply", ""),
        "steps": [dict(step) for step in result.get("steps", [])],
        "meta": dict(result.get("meta", {})),
    }
    payload["meta"]["cached"] = False

    if key in cache:
        cache.pop(key, None)
    elif len(cache) >= limit:
        oldest = next(iter(cache), None)
        if oldest is not None:
            cache.pop(oldest, None)
    cache[key] = payload


def _run_graph_turn(question: str, graph, thread_id: str, fallback_manual_top_k: int) -> dict:
    """Run one natural-style request through the graph."""
    if graph is None:
        return {
            "reply": (
                "I cannot reach the live model right now. Add OPENAI_API_KEY in .env "
                "and refresh to continue."
            ),
            "steps": [],
            "meta": {"path": "no-graph"},
        }

    inputs = {"messages": [HumanMessage(content=question)], "next": ""}
    run_config = {
        "recursion_limit": settings.recursion_limit,
        "configurable": {"thread_id": thread_id},
    }
    steps: list[dict[str, str]] = []

    try:
        for step in graph.stream(inputs, config=run_config):
            for node, update in step.items():
                msgs = update.get("messages") if isinstance(update, dict) else None
                if msgs and msgs[-1].content.strip():
                    steps.append({"node": node, "content": msgs[-1].content.strip()})
    except Exception as exc:  # noqa: BLE001 - keep UI responsive on API/runtime errors
        text = str(exc)
        lowered = text.lower()

        if "401" in text or "api_key" in lowered or "authentication" in lowered:
            fallback = (
                "I cannot authenticate with the model right now. "
                "Please check OPENAI_API_KEY in .env, then try again."
            )
            return {
                "reply": fallback,
                "steps": [{"node": "fallback", "content": fallback}],
                "meta": {"path": "auth-error"},
            }
        elif "timed out" in lowered or "timeout" in lowered:
            reply = _deterministic_entity_reply(
                question,
                reason="model timeout",
                manual_top_k=fallback_manual_top_k,
            )
            return {
                "reply": reply,
                "steps": [{"node": "fallback", "content": reply}],
                "meta": {"path": "model-timeout"},
            }
        elif "rate limit" in lowered or "429" in lowered:
            reply = _deterministic_entity_reply(
                question,
                reason="rate limit",
                manual_top_k=fallback_manual_top_k,
            )
            return {
                "reply": reply,
                "steps": [{"node": "fallback", "content": reply}],
                "meta": {"path": "rate-limit"},
            }
        else:
            reply = _deterministic_entity_reply(
                question,
                reason="temporary model issue",
                manual_top_k=fallback_manual_top_k,
            )
            return {
                "reply": reply,
                "steps": [{"node": "fallback", "content": reply}],
                "meta": {"path": "model-error"},
            }

    # A conversational reply = what the specialists and escalation said.
    spoken = [s["content"] for s in steps]
    if spoken:
        reply = "\n\n".join(spoken)
    elif steps:
        reply = steps[-1]["content"]
    else:
        reply = "I didn't produce a response for that. Try rephrasing your request."
    return {"reply": reply, "steps": steps, "meta": {"path": "graph"}}


def _run_turn(question: str) -> dict:
    """Run one request with fast cache checks and soft-timeout protection."""
    style = st.session_state.get("prompt_style", "base")
    key = _cache_key(style, question)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    started = time.perf_counter()
    manual_top_k = max(1, min(int(st.session_state.get("manual_top_k", 2)), 4))
    if style == "base":
        # Base mode stays deterministic and local for guaranteed responsiveness.
        reply = _deterministic_entity_reply(
            question,
            reason="base mode",
            manual_top_k=manual_top_k,
        )
        result = {
            "reply": reply,
            "steps": [{"node": "fallback", "content": reply}],
            "meta": {"path": "base-deterministic"},
        }
    else:
        timeout_seconds = max(
            1.0,
            float(st.session_state.get("soft_timeout_seconds", settings.ui_soft_timeout_seconds)),
        )
        graph = st.session_state.get("graph")
        thread_id = str(st.session_state.get("thread_id", ""))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-turn")
        future = pool.submit(_run_graph_turn, question, graph, thread_id, manual_top_k)
        try:
            result = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            reply = _deterministic_entity_reply(
                question,
                reason=f"UI soft timeout ({timeout_seconds:.0f}s)",
                manual_top_k=manual_top_k,
            )
            result = {
                "reply": reply,
                "steps": [{"node": "fallback", "content": reply}],
                "meta": {"path": "ui-soft-timeout"},
            }
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    meta = dict(result.get("meta", {}))
    meta["latency_ms"] = elapsed_ms
    meta["style"] = style
    meta.setdefault("cached", False)
    result["meta"] = meta

    _cache_put(key, result)
    return result


def _runtime_status_text() -> str:
    """Human-readable runtime snapshot for /status command."""
    style = st.session_state.get("prompt_style", "base")
    graph_state = "ready" if st.session_state.get("graph") is not None else "unavailable"
    llm_state = "configured" if settings.llm_enabled else "missing OPENAI_API_KEY"
    runtime_profile = st.session_state.get("runtime_profile", "fast")
    soft_timeout = float(
        st.session_state.get("soft_timeout_seconds", settings.ui_soft_timeout_seconds)
    )
    trace_label = "on" if st.session_state.get("show_agent_trace", settings.ui_show_agent_trace) else "off"
    cache_size = len(st.session_state.get("response_cache", {}))
    hits = int(st.session_state.get("cache_hits", 0))
    misses = int(st.session_state.get("cache_misses", 0))

    return (
        "Runtime status:\n"
        f"- Prompt style: {style}\n"
        f"- Runtime profile: {runtime_profile}\n"
        f"- Soft timeout: {soft_timeout:.0f}s\n"
        f"- Agent trace: {trace_label}\n"
        f"- Graph: {graph_state}\n"
        f"- LLM: {llm_state}\n"
        f"- UI response cache: {cache_size} item(s), hits={hits}, misses={misses}\n"
        f"- Data: {PROJECT_DATA.health_report()}\n"
        f"- Yield: {YIELD_DATASET.status_text()}\n"
        f"- Manuals: {MANUAL_INDEX.status_text()}\n"
        f"- RAG: {KNOWLEDGE_RAG.status_text()}"
    )


def _handle_command(prompt: str) -> str | None:
    """Process slash commands and return reply text when handled."""
    text = (prompt or "").strip()
    lower = text.lower()

    if lower in {"/help", "/h"}:
        return (
            "Commands:\n"
            "- /style base\n"
            "- /style natural\n"
            "- /profile fast\n"
            "- /profile rich\n"
            "- /trace on|off|toggle\n"
            "- /status\n"
            "- /rag\n"
            "- /reload"
        )
    if lower.startswith("/style"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /style base or /style natural"
        return _switch_prompt_style(parts[1])
    if lower in {"/status", "/health"}:
        return _runtime_status_text()
    if lower.startswith("/profile"):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /profile fast or /profile rich"
        return _apply_runtime_profile(parts[1])
    if lower.startswith("/trace"):
        parts = text.split(maxsplit=1)
        mode = parts[1] if len(parts) > 1 else "toggle"
        return _set_trace_mode(mode)
    if lower == "/rag":
        return KNOWLEDGE_RAG.status_text()
    if lower in {"/reload", "/refresh"}:
        return _reload_runtime_data()
    return None


def _meta_caption(meta: dict) -> str:
    pieces: list[str] = []
    path = str(meta.get("path", "")).strip()
    if path:
        pieces.append(f"path={path}")
    latency_ms = meta.get("latency_ms")
    if isinstance(latency_ms, int):
        pieces.append(f"latency={latency_ms}ms")
    if meta.get("cached"):
        pieces.append("cache=hit")
    return " | ".join(pieces)


def _render_assistant(message: dict) -> None:
    """Render assistant reply with optional runtime metadata and node trace."""
    st.markdown(message.get("content", ""))
    meta = message.get("meta") or {}
    caption = _meta_caption(meta)
    if caption:
        st.caption(caption)

    steps = message.get("steps") or []
    show_trace = bool(st.session_state.get("show_agent_trace", settings.ui_show_agent_trace))
    if show_trace and steps:
        with st.expander("Agent trace", expanded=False):
            for step in steps:
                node = step.get("node", "")
                label = NODE_LABELS.get(node, node)
                st.markdown(f"**{label}**")
                st.markdown(step.get("content", ""))


def main() -> None:
    st.set_page_config(page_title=settings.app_title, page_icon="🏭", layout="centered")
    _init_session()
    _inject_theme()

    _render_brand_header()
    _render_status_strip()

    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.caption(
            "Enter an entity code — status routes it to Agent Technician (DOWN) or Agent Yield (UP). "
            f"Style: {st.session_state.prompt_style}. "
            "Use /style, /profile, /trace, /status, /rag, /reload, or /help."
        )
    with top_right:
        if st.button("New Chat"):
            _reset_state()
            st.rerun()

    queued_prompt = _render_quick_actions()

    if st.session_state.startup_error:
        st.warning(st.session_state.startup_error)

    for message in st.session_state.chat_log:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant(message)
            else:
                st.markdown(message["content"])

    prompt = st.chat_input(
        "Input entity (e.g. TCB706)..."
    )
    if not prompt and queued_prompt:
        prompt = queued_prompt
    if not prompt:
        return

    st.session_state.chat_log.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    command_reply = _handle_command(prompt)
    if command_reply is not None:
        command_message = {
            "role": "assistant",
            "content": command_reply,
            "steps": [],
            "meta": {"path": "command"},
        }
        st.session_state.chat_log.append(command_message)
        with st.chat_message("assistant"):
            _render_assistant(command_message)
        st.rerun()
        return

    with st.chat_message("assistant"):
        if st.session_state.get("prompt_style", "base") == "natural":
            with st.spinner("Coordinating specialist agents..."):
                result = _run_turn(prompt)
        else:
            result = _run_turn(prompt)
        _render_assistant(
            {
                "content": result["reply"],
                "steps": result["steps"],
                "meta": result.get("meta", {}),
            }
        )

    st.session_state.chat_log.append(
        {
            "role": "assistant",
            "content": result["reply"],
            "steps": result["steps"],
            "meta": result.get("meta", {}),
        }
    )
    st.rerun()


if __name__ == "__main__":
    if _has_streamlit_context():
        main()
    else:
        script = Path(__file__).resolve()
        venv_python = script.parents[2] / ".venv" / "Scripts" / "python.exe"
        python_bin = str(venv_python) if venv_python.exists() else sys.executable
        cmd = [python_bin, "-m", "streamlit", "run", str(script), *sys.argv[1:]]
        print("Launching Streamlit app...")
        raise SystemExit(subprocess.call(cmd))
