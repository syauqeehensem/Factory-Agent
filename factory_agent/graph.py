"""Build the status-driven Equipment Performance Sustaining graph.

Flow (matches the design diagram):

    START → status_check ─(DOWN)→ technician ─→ escalation ─→ END
                         ├(UP)───→ yield ──────→ escalation ─→ END
                         └(not found)─────────────────────────→ END

``status_check`` is a deterministic router: it pulls the entity code from the
user's message, looks up its UP/DOWN status in ``data/status.csv``, and routes
to Agent Technician (excursion management) or Agent Yield (continue sustaining).
Both specialists hand off to a shared Escalation step that opens a ticket when
one is required.
"""

from __future__ import annotations

# Allow `python factory_agent/graph.py` by bootstrapping package context.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "factory_agent"

import re
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from .agents import build_escalation_agent, build_technician_agent, build_yield_agent
from .llm import build_chat_model
from .project_data import PROJECT_DATA
from .state import FactoryState

# Entity codes look like TCB706 / TSX509 (2-4 letters + 2-4 digits).
_ENTITY_RE = re.compile(r"[A-Za-z]{2,4}\d{2,4}")


def _extract_entity(text: str) -> str:
    match = _ENTITY_RE.search(text or "")
    return match.group(0).upper() if match else ""


def _status_check_node():
    """Deterministic 'Tools status check': entity → UP/DOWN → specialist."""

    def status_check(state: FactoryState) -> dict:
        latest_user = ""
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", "") == "human":
                latest_user = str(getattr(msg, "content", ""))
                break

        entity = _extract_entity(latest_user)
        if not entity:
            return {
                "next": "FINISH",
                "messages": [
                    AIMessage(
                        content="Please enter a tool/entity code, e.g. TCB706 or TSX509.",
                        name="status_check",
                    )
                ],
            }

        status = PROJECT_DATA.entity_status_value(entity)
        if status is None:
            known = ", ".join(PROJECT_DATA.known_entities())
            return {
                "next": "FINISH",
                "entity": entity,
                "messages": [
                    AIMessage(
                        content=(
                            f"I couldn't find {entity} in the status data. "
                            f"Known entities include: {known}."
                        ),
                        name="status_check",
                    )
                ],
            }

        # UP → continue sustaining (yield); DOWN/other → excursion management (technician).
        route = "yield" if status == "UP" else "technician"
        return {"next": route, "entity": entity, "status": status}

    return status_check


def _worker_node(agent, name: str):
    """Wrap a ReAct agent as a graph node that reports its final answer."""

    def worker(state: FactoryState) -> dict:
        result = agent.invoke({"messages": state["messages"]})
        final = result["messages"][-1]
        return {"messages": [AIMessage(content=final.content, name=name)]}

    return worker


def build_graph(model=None, checkpointer: Any | None = None, prompt_style: str | None = None):
    """Compile and return the runnable status-driven graph.

    Pass a ``model`` to inject your own (tests pass a dummy-key ChatOpenAI so the
    graph compiles offline); otherwise one is built from your settings. Optionally
    pass a LangGraph ``checkpointer`` to persist conversation state by thread id.
    """
    model = model or build_chat_model()

    builder = StateGraph(FactoryState)
    builder.add_node("status_check", _status_check_node())
    builder.add_node(
        "technician",
        _worker_node(build_technician_agent(model, prompt_style=prompt_style), "technician"),
    )
    builder.add_node(
        "yield",
        _worker_node(build_yield_agent(model, prompt_style=prompt_style), "yield"),
    )
    builder.add_node(
        "escalation",
        _worker_node(build_escalation_agent(model, prompt_style=prompt_style), "escalation"),
    )

    builder.add_edge(START, "status_check")
    builder.add_conditional_edges(
        "status_check",
        lambda state: state["next"],
        {"technician": "technician", "yield": "yield", "FINISH": END},
    )
    # Both specialists hand off to the shared escalation/action step.
    builder.add_edge("technician", "escalation")
    builder.add_edge("yield", "escalation")
    builder.add_edge("escalation", END)
    return builder.compile(checkpointer=checkpointer)
