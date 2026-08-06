"""Interactive console: hand situations to the agent team, one at a time.

Run it::

    python cli.py

Type a floor situation (e.g. "CNC-02 is overheating") and watch the team handle
it. Commands: /world (show factory), /audit (show audit trail), /reset, /quit.

The factory state persists across turns within a session, so you can follow up
("now order a spare belt for CONV-01") and the agents see the running history.
"""

from __future__ import annotations

import sys
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from factory_agent import build_graph
from factory_agent.config import settings
from factory_agent.llm import LLMNotConfigured, build_chat_model
from factory_agent.mock_factory import WORLD, reset_world
from factory_agent.security import AUDIT_LOG, reset_audit

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
    print("  Factory Agent — interactive multi-agent console")
    print("=" * 64)
    if not settings.llm_enabled:
        print("No OPENAI_API_KEY found. Add one to .env to run the agents.\n"
              "You can still explore with /world.")
    print("Commands: /world  /audit  /reset  /quit\n")

    reset_world()
    reset_audit()
    thread_id = str(uuid4())
    graph = None
    if settings.llm_enabled:
        try:
            graph = build_graph(build_chat_model(), checkpointer=CHECKPOINTER)
        except LLMNotConfigured as exc:
            print(f"[warn] {exc}")

    while True:
        try:
            text = input("floor > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0
        if not text:
            continue
        if text in {"/quit", "/exit", "/q"}:
            print("Bye!")
            return 0
        if text == "/world":
            print(WORLD.summary())
            continue
        if text == "/audit":
            for e in AUDIT_LOG:
                print(f"  {e}")
            if not AUDIT_LOG:
                print("  (audit trail empty)")
            continue
        if text == "/reset":
            reset_world()
            reset_audit()
            thread_id = str(uuid4())
            print("Factory, audit trail, and conversation memory reset.")
            continue
        if graph is None:
            print("Agents are offline (no API key). Use /world to explore.")
            continue
        _run(graph, text, thread_id)
        print()


if __name__ == "__main__":
    sys.exit(main())
