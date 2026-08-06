"""End-to-end demo: a Project-Data-only conversation handled by two agents.

Run it::

    python run_demo.py                         # default line-status question
    python run_demo.py --ask "Which entities are down and why?"
    python run_demo.py --status                # data-source status only
    python run_demo.py --graph                 # print the graph structure (no LLM)
"""

from __future__ import annotations

import argparse
import sys

from langchain_core.messages import HumanMessage

from factory_agent import build_graph
from factory_agent.config import settings
from factory_agent.llm import LLMNotConfigured, build_chat_model
from factory_agent.manual_data import MANUAL_INDEX
from factory_agent.project_data import PROJECT_DATA
from factory_agent.yield_data import YIELD_DATASET

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

DEFAULT_SCENARIO = "TCB706"


def _print_project_data_status() -> None:
    PROJECT_DATA.reload()
    YIELD_DATASET.reload()
    MANUAL_INDEX.reload()
    print(PROJECT_DATA.health_report())
    print(YIELD_DATASET.status_text())
    print(MANUAL_INDEX.status_text())


def _print_graph_structure() -> None:
    """Compile with a dummy key (no network) just to show the wiring."""
    from langchain_openai import ChatOpenAI

    dummy = ChatOpenAI(model=settings.chat_model, api_key="sk-dummy-for-compile-only")
    graph = build_graph(dummy)
    print("Graph nodes:", list(graph.get_graph().nodes))
    print("\nMermaid diagram (paste into mermaid.live):\n")
    print(graph.get_graph().draw_mermaid())


def main() -> int:
    parser = argparse.ArgumentParser(description="TCB Chatbot — two-agent demo")
    parser.add_argument("--ask", default=DEFAULT_SCENARIO, help="An entity code to review (e.g. TCB706)")
    parser.add_argument("--status", action="store_true", help="Show Project Data status and exit")
    parser.add_argument("--graph", action="store_true", help="Print the graph structure and exit")
    args = parser.parse_args()

    if args.status:
        _print_project_data_status()
        return 0
    if args.graph:
        _print_graph_structure()
        return 0

    PROJECT_DATA.reload()
    YIELD_DATASET.reload()
    MANUAL_INDEX.reload()

    print("=" * 70)
    print("  TCB Chatbot — Equipment Performance Sustaining demo")
    print("=" * 70)
    _print_project_data_status()
    print(f"\nMode: {'LLM (' + settings.chat_model + ')' if settings.llm_enabled else 'NO KEY'}"
          f" | recursion limit: {settings.recursion_limit}")
    print(f"\n>>> Incoming: {args.ask}\n")

    try:
        model = build_chat_model()
        graph = build_graph(model)
    except LLMNotConfigured as exc:
        print(f"[cannot run] {exc}")
        return 1

    inputs = {"messages": [HumanMessage(content=args.ask)], "next": ""}
    print("--- agent collaboration (each step is one node) ---")
    try:
        for step in graph.stream(inputs, config={"recursion_limit": settings.recursion_limit}):
            for node, update in step.items():
                msgs = update.get("messages") if isinstance(update, dict) else None
                if msgs:
                    print(f"\n● {node}:\n  {msgs[-1].content.strip()}")
    except Exception as exc:  # noqa: BLE001 - surface auth/other errors cleanly
        hint = ""
        if "401" in str(exc) or "api_key" in str(exc).lower() or "authentication" in str(exc).lower():
            hint = ("\nThis looks like an API-key problem. Put a valid key in .env "
                    "(OPENAI_API_KEY=...). A wrong/expired key returns a 401.")
        print(f"\n[run error] {exc}{hint}")
        return 1

    print("\n" + "=" * 70)
    print("RESULT — latest two-agent response")
    print("=" * 70)
    print("Run completed. Use the Streamlit UI or cli.py for multi-turn conversation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
