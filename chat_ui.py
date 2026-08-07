"""Popup-style chatbot UI for the Factory Agent workspace.

Run:
    streamlit run chat_ui.py
"""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from factory_agent import build_graph
from factory_agent.config import settings
from factory_agent.knowledge_rag import KNOWLEDGE_RAG
from factory_agent.manual_data import MANUAL_INDEX
from factory_agent.graph import _extract_entity
from factory_agent.llm import LLMNotConfigured, build_chat_model
from factory_agent.project_data import PROJECT_DATA
from factory_agent.tickets import TICKET_STORE
from factory_agent.yield_data import YIELD_DATASET

# Chat memory is rebuilt from persisted chat history each turn, so no in-RAM
# checkpointer is required for continuity across app restarts.
CHECKPOINTER = None
CHAT_MEMORY_PATH = Path(__file__).resolve().parent / ".chat_memory.json"


def _persistable_chat_log() -> list[dict[str, str]]:
    """Return a compact chat log safe to store on disk."""
    stored: list[dict[str, str]] = []
    for msg in st.session_state.get("chat_log", []):
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            stored.append({"role": role, "content": content})
    return stored


def _save_persistent_chat_state() -> None:
    """Persist lightweight chat state to disk for restart continuity."""
    payload = {
        "thread_id": str(st.session_state.get("thread_id", "")),
        "last_entity": str(st.session_state.get("last_entity", "")),
        "chat_log": _persistable_chat_log(),
        "saved_at": int(time.time()),
    }
    try:
        CHAT_MEMORY_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001 - memory persistence should not break chat flow
        return


def _clear_persistent_chat_state() -> None:
    """Delete persisted chat memory if it exists."""
    try:
        if CHAT_MEMORY_PATH.exists():
            CHAT_MEMORY_PATH.unlink()
    except Exception:  # noqa: BLE001 - best effort cleanup
        return


def _load_persistent_chat_state() -> dict:
    """Load persisted chat memory from disk, returning a normalized payload."""
    if not CHAT_MEMORY_PATH.exists():
        return {}

    try:
        raw = json.loads(CHAT_MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - corrupted memory file should be ignored
        return {}

    log: list[dict[str, str]] = []
    for item in raw.get("chat_log", []) if isinstance(raw.get("chat_log"), list) else []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            log.append({"role": role, "content": content})

    return {
        "thread_id": str(raw.get("thread_id", "")).strip(),
        "last_entity": str(raw.get("last_entity", "")).strip().upper(),
        "chat_log": log,
    }


def _derive_last_entity_from_chat(chat_log: list[dict[str, str]]) -> str:
    """Infer the last mentioned entity from recent user messages."""
    for msg in reversed(chat_log):
        if str(msg.get("role", "")).strip().lower() != "user":
            continue
        entity = _extract_entity(str(msg.get("content", "")))
        if entity:
            return entity
    return ""


def _set_last_entity_from_text(text: str) -> None:
    entity = _extract_entity(text)
    if entity:
        st.session_state.last_entity = entity


def _resolve_entity_with_memory(question: str) -> str:
    """Resolve entity from current question, falling back to last remembered one."""
    direct = _extract_entity(question)
    if direct:
        return direct
    if not _should_reuse_last_entity(question):
        return ""
    remembered = str(st.session_state.get("last_entity", "")).strip().upper()
    return remembered


def _cache_prompt_with_context(prompt: str) -> str:
    """Make cache keys context-aware when a prompt relies on remembered entity."""
    direct = _extract_entity(prompt)
    if direct:
        return prompt
    if not _should_reuse_last_entity(prompt):
        return prompt
    remembered = str(st.session_state.get("last_entity", "")).strip().upper()
    return f"{prompt} [entity:{remembered}]" if remembered else prompt


def _chat_history_messages(max_messages: int = 24) -> list:
    """Convert recent chat log entries into LangChain messages for model context."""
    history = st.session_state.get("chat_log", [])[-max(2, max_messages) :]
    converted: list = []
    for msg in history:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
    return converted


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
    project_dir = Path(__file__).resolve().parent
    assets_dir = project_dir / "assets"
    configured_logo = Path(settings.foundry_logo_path)
    candidates = [
        project_dir / "image (4).png",
        project_dir / "image (4).jpg",
        project_dir / "image (4).jpeg",
        project_dir / "image (4).webp",
        project_dir / "image (4).svg",
        configured_logo,
        assets_dir / "image (4).png",
        assets_dir / "image (4).jpg",
        assets_dir / "image (4).jpeg",
        assets_dir / "image (4).webp",
        assets_dir / "image (4).svg",
        assets_dir / "foundry-logo.svg",
        assets_dir / "foundry-logo.png",
    ]
    if not configured_logo.is_absolute():
        candidates.insert(6, (project_dir / configured_logo))
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
            st.markdown("### EPS")
    with col_text:
        st.title(settings.app_title)


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


def _reset_state(clear_persistent: bool = True) -> None:
    """Reset graph and chat history for a new session."""
    st.session_state.prompt_style = "natural"
    PROJECT_DATA.reload()
    YIELD_DATASET.reload()
    MANUAL_INDEX.reload()
    KNOWLEDGE_RAG.reload()
    TICKET_STORE.reset()
    st.session_state.chat_log = []
    st.session_state.response_cache = {}
    st.session_state.cache_hits = 0
    st.session_state.cache_misses = 0
    st.session_state.last_entity = ""
    st.session_state.thread_id = str(uuid4())
    st.session_state.graph, st.session_state.startup_error = _start_graph()
    if clear_persistent:
        _clear_persistent_chat_state()
    _save_persistent_chat_state()


def _init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
    # Keep Streamlit behavior aligned with CLI-style multi-step outputs.
    st.session_state.prompt_style = "natural"
    if "response_cache" not in st.session_state:
        st.session_state.response_cache = {}
    if "cache_hits" not in st.session_state:
        st.session_state.cache_hits = 0
    if "cache_misses" not in st.session_state:
        st.session_state.cache_misses = 0
    if "soft_timeout_seconds" not in st.session_state:
        st.session_state.soft_timeout_seconds = float(settings.ui_soft_timeout_seconds)
    if "manual_top_k" not in st.session_state:
        st.session_state.manual_top_k = 2
    if "last_entity" not in st.session_state:
        st.session_state.last_entity = ""

    if st.session_state.get("initialized"):
        st.session_state.graph, st.session_state.startup_error = _start_graph()
        return

    st.session_state.initialized = True
    persisted = _load_persistent_chat_state()
    if persisted:
        remembered_thread = persisted.get("thread_id", "")
        if remembered_thread:
            st.session_state.thread_id = remembered_thread
        st.session_state.chat_log = persisted.get("chat_log", [])
        remembered_entity = persisted.get("last_entity", "")
        st.session_state.last_entity = remembered_entity or _derive_last_entity_from_chat(
            st.session_state.chat_log
        )

        PROJECT_DATA.reload()
        YIELD_DATASET.reload()
        MANUAL_INDEX.reload()
        KNOWLEDGE_RAG.reload()
        st.session_state.response_cache = {}
        st.session_state.cache_hits = 0
        st.session_state.cache_misses = 0
        st.session_state.graph, st.session_state.startup_error = _start_graph()
    else:
        _reset_state(clear_persistent=False)


def _reload_runtime_data() -> str:
    """Reload data/indexes and refresh the graph without clearing chat history."""
    PROJECT_DATA.reload()
    YIELD_DATASET.reload()
    MANUAL_INDEX.reload()
    KNOWLEDGE_RAG.reload()
    st.session_state.response_cache = {}
    st.session_state.graph, st.session_state.startup_error = _start_graph()
    return "Runtime data and indexes reloaded."


def _list_preview(items: list[str], max_items: int = 12) -> str:
    if not items:
        return "none"
    preview = ", ".join(items[:max_items])
    if len(items) > max_items:
        preview += f", ... (+{len(items) - max_items} more)"
    return preview


def _local_intent_reply(question: str) -> str | None:
    """Answer common status-intent questions directly from local datasets."""
    q = _normalize_prompt(question)

    down_markers = [
        "which entity is down",
        "which entities are down",
        "which tool is down",
        "which tools are down",
        "entity down",
        "entities down",
        "tool down",
        "tools down",
        "currently down",
        "list down",
        "what is down",
        "what are down",
        "what tool is down",
        "what tools are down",
        "what tools is down",
    ]
    up_markers = [
        "which entity is up",
        "which entities are up",
        "which tool is up",
        "which tools are up",
        "entity up",
        "entities up",
        "tool up",
        "tools up",
        "currently up",
        "list up",
        "what is up",
        "what are up",
        "what tool is up",
        "what tools are up",
        "what tools is up",
    ]
    escalation_markers = [
        "escalat",
        "should be escalated",
        "wanted to be escalated",
        "need escalation",
        "require escalation",
        "requires escalation",
        "down-tool",
        "down tool",
    ]
    highest_yield_markers = [
        "highest yield",
        "best yield",
        "top yield",
        "max yield",
    ]
    lowest_yield_markers = [
        "lowest yield",
        "worst yield",
        "min yield",
        "minimum yield",
    ]
    underperforming_markers = [
        "underperform",
        "low yield",
        "below threshold",
        "below target",
        "poor yield",
        "weak yield",
    ]

    asks_down = any(marker in q for marker in down_markers)
    asks_up = any(marker in q for marker in up_markers)
    asks_escalation = any(marker in q for marker in escalation_markers)
    asks_highest_yield = any(marker in q for marker in highest_yield_markers)
    asks_lowest_yield = any(marker in q for marker in lowest_yield_markers)
    asks_underperforming = any(marker in q for marker in underperforming_markers)

    if not (
        asks_down
        or asks_up
        or asks_escalation
        or asks_highest_yield
        or asks_lowest_yield
        or asks_underperforming
    ):
        return None

    status_rows = PROJECT_DATA.status_rows
    if not status_rows:
        return "I cannot read status.csv right now. Please run /reload and try again."

    down_entities = sorted({r.entity for r in status_rows if r.status == "DOWN"})
    up_entities = sorted({r.entity for r in status_rows if r.status == "UP"})
    goal = float(settings.yield_threshold)
    yield_pairs = [
        (row.entity, YIELD_DATASET.entity_yield(row.entity)) for row in status_rows
    ]
    yield_pairs = [(entity, yv) for entity, yv in yield_pairs if yv is not None]

    sections: list[str] = ["Here is the latest local-data snapshot:"]

    if asks_down:
        if down_entities:
            sections.append(
                f"● DOWN entities ({len(down_entities)}):\n\n{_list_preview(down_entities)}"
            )
        else:
            sections.append("● DOWN entities:\n\nNone right now. All tracked entities are UP.")

    if asks_up:
        if up_entities:
            sections.append(
                f"● UP entities ({len(up_entities)}):\n\n{_list_preview(up_entities, max_items=18)}"
            )
        else:
            sections.append("● UP entities:\n\nNone found in the current status dataset.")

    if asks_escalation:
        escalation_lines: list[str] = []
        for row in sorted(status_rows, key=lambda x: x.entity):
            reasons: list[str] = []
            if row.status == "DOWN":
                reasons.append("status is DOWN")
            yv = YIELD_DATASET.entity_yield(row.entity)
            if yv is not None and yv < goal:
                reasons.append(f"yield {yv:.1f}% below {goal:.0f}%")
            if reasons:
                escalation_lines.append(f"- {row.entity}: {'; '.join(reasons)}")

        if escalation_lines:
            max_lines = 14
            preview_lines = escalation_lines[:max_lines]
            if len(escalation_lines) > max_lines:
                preview_lines.append(f"- ... (+{len(escalation_lines) - max_lines} more)")
            sections.append(
                "● Entities that should be escalated now:\n\n" + "\n".join(preview_lines)
            )
        else:
            sections.append(
                "● Entities that should be escalated now:\n\n"
                "None at the moment (no DOWN status and no yield below threshold)."
            )

    if asks_highest_yield:
        if yield_pairs:
            top_entity, top_yield = max(yield_pairs, key=lambda p: p[1])
            sections.append(
                f"● Highest yield entity:\n\n{top_entity} at {top_yield:.1f}%"
            )
        else:
            sections.append("● Highest yield entity:\n\nYield data is unavailable right now.")

    if asks_lowest_yield:
        if yield_pairs:
            bottom_entity, bottom_yield = min(yield_pairs, key=lambda p: p[1])
            sections.append(
                f"● Lowest yield entity:\n\n{bottom_entity} at {bottom_yield:.1f}%"
            )
        else:
            sections.append("● Lowest yield entity:\n\nYield data is unavailable right now.")

    if asks_underperforming:
        underperformers = sorted(
            [(entity, yv) for entity, yv in yield_pairs if yv < goal],
            key=lambda p: p[1],
        )
        if underperformers:
            lines = [f"- {entity}: {yv:.1f}% (below {goal:.0f}%)" for entity, yv in underperformers[:14]]
            if len(underperformers) > 14:
                lines.append(f"- ... (+{len(underperformers) - 14} more)")
            sections.append("● Underperforming entities:\n\n" + "\n".join(lines))
        else:
            sections.append(
                f"● Underperforming entities:\n\nNone right now (all yields are at or above {goal:.0f}%)."
            )

    if asks_down:
        total_tools = len(status_rows)
        down_count = len(down_entities)
        if down_count:
            down_ratio = (down_count / total_tools) * 100 if total_tools else 0.0
            down_set = set(down_entities)
            down_yields = sorted(
                [(entity, yv) for entity, yv in yield_pairs if entity in down_set],
                key=lambda p: p[1],
            )
            down_with_missing_yield = sorted(
                [entity for entity in down_entities if entity not in {name for name, _ in down_yields}]
            )

            analysis_lines = [
                f"DOWN ratio: {down_count}/{total_tools} tools ({down_ratio:.1f}%)."
            ]

            low_yield_down = [
                f"{entity} ({yv:.1f}%)" for entity, yv in down_yields if yv < goal
            ]
            if low_yield_down:
                analysis_lines.append(
                    f"DOWN tools below {goal:.0f}% yield: {', '.join(low_yield_down[:6])}."
                )

            if down_yields:
                priority = ", ".join(
                    [f"{entity} ({yv:.1f}%)" for entity, yv in down_yields[:3]]
                )
                analysis_lines.append(f"Priority check order: {priority}.")
            else:
                analysis_lines.append(
                    "Priority check order: all DOWN tools (yield values unavailable)."
                )

            if down_with_missing_yield:
                analysis_lines.append(
                    "Missing yield for DOWN tools: "
                    f"{_list_preview(down_with_missing_yield, max_items=6)}."
                )

            analysis_lines.append(
                "Recommended action: dispatch technician checks and keep/open down-tool tickets "
                "until status returns UP."
            )

            sections.append(
                "● analysis:\n\n" + "\n".join([f"- {line}" for line in analysis_lines])
            )
        else:
            sections.append(
                "● analysis:\n\n"
                "- No tools are currently DOWN, so no immediate down-tool recovery action is needed."
            )

    return "\n\n".join(sections)


def _should_reuse_last_entity(question: str) -> bool:
    """Only reuse remembered entity for true follow-up phrasing."""
    q = _normalize_prompt(question)
    if not q:
        return False

    # Global/multi-entity intents should not be forced to the remembered entity.
    global_markers = [
        "which entity",
        "which entities",
        "which tool",
        "which tools",
        "list",
        "highest yield",
        "best yield",
        "top yield",
        "max yield",
        "lowest yield",
        "worst yield",
        "min yield",
        "minimum yield",
        "underperform",
        "below threshold",
        "below target",
        "what is down",
        "what are down",
        "what tool is down",
        "what tools are down",
        "what tools is down",
        "what is up",
        "what are up",
        "what tool is up",
        "what tools are up",
        "what tools is up",
    ]
    if any(marker in q for marker in global_markers):
        return False

    followup_markers = [
        "what about it",
        "how about it",
        "what about its",
        "how about its",
        "about it",
        "about its",
        "should it",
        "is it",
        "does it",
        "can it",
        "that entity",
        "this entity",
        "that tool",
        "this tool",
        "that one",
        "this one",
        "same entity",
        "same tool",
    ]
    return any(marker in q for marker in followup_markers)


def _deterministic_entity_reply(question: str, reason: str = "", manual_top_k: int = 2) -> str:
    """Return a fast local-data answer in CLI-style section format."""
    del manual_top_k, reason  # retained in signature for call-site compatibility

    natural = _local_intent_reply(question)
    if natural is not None:
        return natural

    entity = _resolve_entity_with_memory(question)
    if not entity:
        return "Please enter a valid entity code like TCB706 or TSX509."

    key = entity.strip().upper()
    status = (PROJECT_DATA.entity_status_value(key) or "UNKNOWN").upper()
    yield_value = YIELD_DATASET.entity_yield(key)
    goal = float(settings.yield_threshold)

    if yield_value is None:
        technician_text = (
            f"The entity {key} is currently {status}, but no yield record is available. "
            "Please refresh local data and retry."
        )
        summary_text = f"Checked status of {key}; yield data unavailable."
        escalation_text = (
            f"No escalation ticket was created for {key} because yield data is missing."
        )
        return (
            f"● technician:\n\n{technician_text}\n\n"
            f"Summary: {summary_text}\n\n"
            f"● escalation:\n\n{escalation_text}"
        )

    yield_low = yield_value < goal
    is_down = status == "DOWN"
    ticket_needed = is_down or yield_low

    if ticket_needed:
        if is_down and yield_low:
            technician_text = (
                f"The entity {key} is currently DOWN, and it has a yield of {yield_value:.1f}%, "
                f"which is below the acceptable threshold of {goal:.1f}%. "
                "A down-tool ticket is required for further action."
            )
            summary_text = (
                f"Checked status of {key}; tool is DOWN and yield is low, down-tool ticket needed."
            )
        elif is_down:
            technician_text = (
                f"The entity {key} is currently DOWN. "
                f"Its latest yield is {yield_value:.1f}% against a {goal:.1f}% threshold. "
                "A down-tool ticket is required for further action."
            )
            summary_text = f"Checked status of {key}; tool is DOWN, down-tool ticket needed."
        else:
            technician_text = (
                f"The entity {key} is currently UP, but it has a yield of {yield_value:.1f}%, "
                f"which is below the acceptable threshold of {goal:.1f}%. "
                "A down-tool ticket is required for further action."
            )
            summary_text = f"Checked status of {key}; yield is low, down-tool ticket needed."

        existing_down_ticket = next(
            (t for t in reversed(TICKET_STORE.for_entity(key)) if t.ticket_type == "down-tool"),
            None,
        )
        if existing_down_ticket is None:
            reason_bits: list[str] = []
            if is_down:
                reason_bits.append("entity is DOWN")
            if yield_low:
                reason_bits.append(f"yield {yield_value:.1f}% below {goal:.1f}%")
            TICKET_STORE.create(
                entity=key,
                ticket_type="down-tool",
                reason="; ".join(reason_bits) or "manual review",
            )

        escalation_text = (
            f"I created a down-tool ticket for {key} due to its low yield of {yield_value:.1f}%. "
            "No escalation ticket was necessary."
            if yield_low
            else f"I created a down-tool ticket for {key} because it is currently DOWN. "
            "No escalation ticket was necessary."
        )
    else:
        technician_text = (
            f"The entity {key} is currently UP, and its yield of {yield_value:.1f}% "
            f"meets the acceptable threshold of {goal:.1f}%."
        )
        summary_text = f"Checked status of {key}; yield is healthy, no down-tool ticket needed."
        escalation_text = f"No escalation ticket was necessary for {key}."

    return (
        f"● technician:\n\n{technician_text}\n\n"
        f"Summary: {summary_text}\n\n"
        f"● escalation:\n\n{escalation_text}"
    )


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


def _run_graph_turn(
    question: str,
    graph,
    thread_id: str,
    fallback_manual_top_k: int,
    history_messages: list | None = None,
) -> dict:
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

    inputs_messages = list(history_messages or [])
    if not inputs_messages:
        inputs_messages = [HumanMessage(content=question)]
    else:
        # Ensure latest user prompt is present in the graph input.
        if not any(
            getattr(msg, "type", "") == "human"
            and str(getattr(msg, "content", "")).strip() == question.strip()
            for msg in inputs_messages
        ):
            inputs_messages.append(HumanMessage(content=question))

    inputs = {"messages": inputs_messages, "next": ""}
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

    cli_style = _format_cli_style_reply(steps)
    if cli_style:
        reply = cli_style
    else:
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
    _set_last_entity_from_text(question)
    key = _cache_key(style, _cache_prompt_with_context(question))
    cached = _cache_get(key)
    if cached is not None:
        return cached

    started = time.perf_counter()

    # Deterministic handling for status-intent prompts keeps UX fast and grounded.
    local_intent = _local_intent_reply(question)
    if local_intent is not None:
        result = {
            "reply": local_intent,
            "steps": [{"node": "fallback", "content": local_intent}],
            "meta": {"path": "local-intent"},
        }
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        meta = dict(result.get("meta", {}))
        meta["latency_ms"] = elapsed_ms
        meta["style"] = style
        meta.setdefault("cached", False)
        result["meta"] = meta
        _cache_put(key, result)
        return result

    resolved_question = question
    if not _extract_entity(question):
        remembered = str(st.session_state.get("last_entity", "")).strip().upper()
        if remembered and _should_reuse_last_entity(question):
            resolved_question = f"{question}\n\nContext: the current entity is {remembered}."

    manual_top_k = max(1, min(int(st.session_state.get("manual_top_k", 2)), 4))
    if style == "base":
        # Base mode stays deterministic and local for guaranteed responsiveness.
        reply = _deterministic_entity_reply(
            resolved_question,
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
        history_messages = _chat_history_messages()
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-turn")
        future = pool.submit(
            _run_graph_turn,
            resolved_question,
            graph,
            thread_id,
            manual_top_k,
            history_messages,
        )
        try:
            result = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            reply = _deterministic_entity_reply(
                resolved_question,
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
    soft_timeout = float(
        st.session_state.get("soft_timeout_seconds", settings.ui_soft_timeout_seconds)
    )
    cache_size = len(st.session_state.get("response_cache", {}))
    hits = int(st.session_state.get("cache_hits", 0))
    misses = int(st.session_state.get("cache_misses", 0))

    return (
        "Runtime status:\n"
        f"- Prompt style: {style}\n"
        f"- Soft timeout: {soft_timeout:.0f}s\n"
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
            "- /reset\n"
            "- /status\n"
            "- /rag\n"
            "- /reload"
        )
    if lower in {"/reset", "/new", "/newchat"}:
        _reset_state()
        return "Conversation reset."
    if lower in {"/status", "/health"}:
        return _runtime_status_text()
    if lower == "/rag":
        return KNOWLEDGE_RAG.status_text()
    if lower in {"/reload", "/refresh"}:
        return _reload_runtime_data()
    return None


def _render_assistant(message: dict) -> None:
    """Render assistant reply with clean, user-facing output only."""
    st.markdown(message.get("content", ""))


def _format_cli_style_reply(steps: list[dict[str, str]]) -> str:
    """Match the concise CLI-style sections shown in image (2)."""
    blocks: list[str] = []
    for step in steps:
        node = step.get("node", "").strip().lower()
        if node in {"", "status_check", "fallback"}:
            continue
        content = step.get("content", "").strip()
        if not content:
            continue
        blocks.append(f"● {node}:\n\n{content}")
    return "\n\n".join(blocks)


def main() -> None:
    st.set_page_config(page_title=settings.app_title, page_icon="🏭", layout="centered")
    _init_session()
    _inject_theme()

    _render_brand_header()

    st.caption("Input entity > (for example: TCB706)")
    if st.button("New Chat"):
        _reset_state()
        st.rerun()

    if st.session_state.startup_error:
        st.warning(st.session_state.startup_error)

    for message in st.session_state.chat_log:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_assistant(message)
            else:
                st.markdown(message["content"])

    prompt = st.chat_input("Input entity (e.g. TCB706)...")
    if not prompt:
        return

    st.session_state.chat_log.append({"role": "user", "content": prompt})
    _save_persistent_chat_state()
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
        _save_persistent_chat_state()
        with st.chat_message("assistant"):
            _render_assistant(command_message)
        st.rerun()
        return

    with st.chat_message("assistant"):
        with st.spinner("Coordinating specialist agents..."):
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
    _save_persistent_chat_state()
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
