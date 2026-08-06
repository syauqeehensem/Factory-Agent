"""Interactive console: enter an entity, get the sustaining decision.

Run it::

    python cli.py

Type a tool/entity code (e.g. TCB706). The status check routes DOWN tools to
Agent Technician and UP tools to Agent Yield, then Escalation opens a ticket if
one is required. Commands: /reset and /quit.
"""

from __future__ import annotations

import sys
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from factory_agent import build_graph
from factory_agent.config import settings
from factory_agent.knowledge_rag import KNOWLEDGE_RAG
from factory_agent.llm import LLMNotConfigured, build_chat_model
from factory_agent.manual_data import MANUAL_INDEX
from factory_agent.project_data import PROJECT_DATA
from factory_agent.yield_data import YIELD_DATASET

CHECKPOINTER = MemorySaver()

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass


def _run(graph, text: str, thread_id: str) -> None:
    inputs = {"messages": [HumanMessage(content=text)], "next": ""}
    run_config = {
        "recursion_limit": settings.recursion_limit,
        "configurable": {"thread_id": thread_id},
    }
    try:
        for step in graph.stream(inputs, config=run_config):
            for node, update in step.items():
                msgs = update.get("messages") if isinstance(update, dict) else None
                if msgs:
                    print(f"\n● {node}:\n  {msgs[-1].content.strip()}")
    except Exception as exc:  # noqa: BLE001 - keep the REPL alive
        hint = " (check OPENAI_API_KEY in .env)" if "401" in str(exc) else ""
        print(f"\n[error] {exc}{hint}")


def main() -> int:
    print("=" * 64)
    print("  TCB Chatbot — Equipment Performance Sustaining")
    print("=" * 64)
    if not settings.llm_enabled:
        print("No OPENAI_API_KEY found. Add one to .env to run live answers.")
    print("Commands: /reset  /style base  /style natural  /quit\n")

    PROJECT_DATA.reload()
    YIELD_DATASET.reload()
    MANUAL_INDEX.reload()
    KNOWLEDGE_RAG.reload()
    thread_id = str(uuid4())
    prompt_style = settings.prompt_style if settings.prompt_style in {"base", "natural"} else "base"
    print(f"Current prompt style: {prompt_style}")
    graph = None
    if settings.llm_enabled:
        try:
            graph = build_graph(
                build_chat_model(), checkpointer=CHECKPOINTER, prompt_style=prompt_style
            )
        except LLMNotConfigured as exc:
            print(f"[warn] {exc}")

    while True:
        try:
            text = input("Input entity > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0
        if not text:
            continue
        if text in {"/quit", "/exit", "/q"}:
            print("Bye!")
            return 0
        if text == "/reset":
            PROJECT_DATA.reload()
            YIELD_DATASET.reload()
            MANUAL_INDEX.reload()
            KNOWLEDGE_RAG.reload()
            thread_id = str(uuid4())
            print("Conversation memory reset.")
            continue
        if text.startswith("/style"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /style base  OR  /style natural")
                continue
            candidate = parts[1].strip().lower()
            if candidate not in {"base", "natural"}:
                print("Invalid style. Use: base or natural")
                continue
            prompt_style = candidate
            if settings.llm_enabled:
                try:
                    graph = build_graph(
                        build_chat_model(),
                        checkpointer=CHECKPOINTER,
                        prompt_style=prompt_style,
                    )
                except LLMNotConfigured as exc:
                    print(f"[warn] {exc}")
                    graph = None
            print(f"Prompt style switched to: {prompt_style}")
            continue
        if graph is None:
            print("Agents are offline (no API key).")
            continue
        _run(graph, text, thread_id)
        print()


if __name__ == "__main__":
    sys.exit(main())
