"""End-to-end demo: a predictive-maintenance triage handled by the agent team.

Run it::

    python run_demo.py                         # default CNC-01 high-vibration scenario
    python run_demo.py --ask "CONV-01 is noisy, please investigate"
    python run_demo.py --list                  # just show the simulated factory
    python run_demo.py --graph                 # print the graph structure (no LLM)

Watch the Floor Supervisor delegate to the Maintenance Scheduler and Parts
Procurement Agent in a loop, then see the resulting work orders, purchase orders,
and the secure audit trail.
"""

from __future__ import annotations

import argparse
import sys

from langchain_core.messages import HumanMessage

from factory_agent import build_graph
from factory_agent.config import settings
from factory_agent.llm import LLMNotConfigured, build_chat_model
from factory_agent.mock_factory import WORLD, reset_world
from factory_agent.security import AUDIT_LOG, reset_audit

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

DEFAULT_SCENARIO = (
    "Machine CNC-01 has triggered a high-vibration alert on the floor. Please triage "
    "it and take whatever maintenance and parts actions are needed to resolve it."
)


def _print_world() -> None:
    print("Simulated factory floor:")
    for m in WORLD.machines.values():
        r = m.reading
        mapping = f" | yield entity: {m.yield_entity}" if m.yield_entity else ""
        print(f"  {m.machine_id} ({m.name}): vib {r.vibration_mm_s} mm/s, "
              f"{r.temperature_c} C, {r.runtime_hours} h  ->  {m.status}{mapping}")
    print("Parts inventory:")
    for p in WORLD.parts.values():
        print(f"  {p.part_number} ({p.description}): {p.on_hand} on hand @ ${p.unit_cost:.0f}")


def _print_graph_structure() -> None:
    """Compile with a dummy key (no network) just to show the wiring."""
    from langchain_openai import ChatOpenAI

    dummy = ChatOpenAI(model=settings.chat_model, api_key="sk-dummy-for-compile-only")
    graph = build_graph(dummy)
    print("Graph nodes:", list(graph.get_graph().nodes))
    print("\nMermaid diagram (paste into mermaid.live):\n")
    print(graph.get_graph().draw_mermaid())


def main() -> int:
    parser = argparse.ArgumentParser(description="Factory Agent — multi-agent demo")
    parser.add_argument("--ask", default=DEFAULT_SCENARIO, help="The situation to hand the team")
    parser.add_argument("--list", action="store_true", help="Show the simulated factory and exit")
    parser.add_argument("--graph", action="store_true", help="Print the graph structure and exit")
    args = parser.parse_args()

    if args.list:
        _print_world()
        return 0
    if args.graph:
        _print_graph_structure()
        return 0

    reset_world()
    reset_audit()

    print("=" * 70)
    print("  Factory Agent — multi-agent production orchestration demo")
    print("=" * 70)
    _print_world()
    print(f"\nMode: {'LLM (' + settings.chat_model + ')' if settings.llm_enabled else 'NO KEY'}"
          f" | auto-approve limit: ${settings.auto_approve_limit:.0f}"
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
    print("RESULT — factory state after the run")
    print("=" * 70)
    print(WORLD.summary())
    print("\nSecure audit trail (every action the agents took):")
    for entry in AUDIT_LOG:
        print(f"  {entry}")
    if not AUDIT_LOG:
        print("  (no actions were taken)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
