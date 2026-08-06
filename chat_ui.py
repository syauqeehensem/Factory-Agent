"""Popup-style chatbot UI for the Factory Agent workspace.

Run:
    streamlit run chat_ui.py
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from uuid import uuid4

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from factory_agent import build_graph
from factory_agent.config import settings
from factory_agent.llm import LLMNotConfigured, build_chat_model
from factory_agent.project_data import PROJECT_DATA
from factory_agent.yield_data import YIELD_DATASET

CHECKPOINTER = MemorySaver()

NODE_LABELS = {
    "supervisor": "Floor Supervisor",
    "agent_technician": "Agent Technician",
    "agent_yield": "Agent Yield",
}


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
            "Human-like, memory-aware multi-agent manufacturing assistant "
            "powered by Project Data and secure tool use."
        )


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
        return build_graph(build_chat_model(), checkpointer=CHECKPOINTER), ""
    except LLMNotConfigured as exc:
        return None, str(exc)


def _reset_state() -> None:
    """Reset graph and chat history for a new session."""
    PROJECT_DATA.reload()
    YIELD_DATASET.reload()
    st.session_state.chat_log = []
    st.session_state.thread_id = str(uuid4())
    st.session_state.graph, st.session_state.startup_error = _start_graph()


def _init_session() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
    if st.session_state.get("initialized"):
        return
    st.session_state.initialized = True
    _reset_state()


def _run_turn(question: str) -> dict:
    """Run one request through the graph, returning a conversational reply + steps."""
    graph = st.session_state.graph
    if graph is None:
        return {
            "reply": (
                "I cannot reach the live model right now. Add OPENAI_API_KEY in .env "
                "and refresh to continue."
            ),
            "steps": [],
        }

    inputs = {"messages": [HumanMessage(content=question)], "next": ""}
    run_config = {
        "recursion_limit": settings.recursion_limit,
        "configurable": {"thread_id": st.session_state.thread_id},
    }
    steps: list[dict[str, str]] = []

    try:
        for step in graph.stream(inputs, config=run_config):
            for node, update in step.items():
                msgs = update.get("messages") if isinstance(update, dict) else None
                if msgs and msgs[-1].content.strip():
                    steps.append({"node": node, "content": msgs[-1].content.strip()})
    except Exception as exc:  # noqa: BLE001 - keep UI responsive on API/runtime errors
        hint = ""
        text = str(exc)
        if "401" in text or "api_key" in text.lower() or "authentication" in text.lower():
            hint = " Check OPENAI_API_KEY in .env (invalid or expired keys return 401)."
        return {"reply": f"Something went wrong: {exc}.{hint}", "steps": []}

    # A conversational reply = what the specialists said; routing chatter stays hidden.
    spoken = [s["content"] for s in steps if s["node"] != "supervisor"]
    if spoken:
        reply = "\n\n".join(spoken)
    elif steps:
        reply = steps[-1]["content"]
    else:
        reply = "I didn't produce a response for that. Try rephrasing your request."
    return {"reply": reply, "steps": steps}


def _render_assistant(message: dict) -> None:
    """Render assistant reply."""
    st.markdown(message.get("content", ""))


def main() -> None:
    st.set_page_config(page_title=settings.app_title, page_icon="🏭", layout="centered")
    _init_session()

    _render_brand_header()

    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.caption("Project-Data-only assistants: Agent Technician + Agent Yield")
    with top_right:
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

    prompt = st.chat_input(
        "Ask naturally: line status, ticket issue, manual troubleshooting, or yield trend..."
    )
    if not prompt:
        return

    st.session_state.chat_log.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = _run_turn(prompt)
        _render_assistant({"content": result["reply"], "steps": result["steps"]})

    st.session_state.chat_log.append(
        {"role": "assistant", "content": result["reply"], "steps": result["steps"]}
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
