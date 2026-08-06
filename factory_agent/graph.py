"""Build the stateful, cyclic multi-agent graph.

This is the heart of the kit. The **Floor Supervisor** is a router node: it reads
the conversation so far and decides which specialist should act next, or that the
job is done. Each specialist node runs its ReAct agent, appends a summary to the
shared messages, and routes **back to the supervisor** — that loop is the cycle:

    START → supervisor ─(conditional)→ agent_technician ─→ supervisor
                       └───────────────→ agent_yield ─────────→ supervisor
                       └──"FINISH"──────────────────────────────→ END

The supervisor uses *structured output* so its routing choice is a typed value,
not free text we have to parse — robust and easy to extend with more agents.
"""

from __future__ import annotations

# Allow `python factory_agent/graph.py` by bootstrapping package context.
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "factory_agent"

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from .agents import (
    build_technician_agent,
    build_yield_agent,
)
from .llm import build_chat_model
from .state import FactoryState

# The agents the supervisor can delegate to (plus the FINISH sentinel).
WORKERS = ["agent_technician", "agent_yield"]
_ROUTE_OPTIONS = WORKERS + ["FINISH"]

TECH_KEYWORDS = {
    "status",
    "ticket",
    "down",
    "up",
    "entity",
    "tsx",
    "tcb",
    "manual",
    "technician",
    "troubleshoot",
    "fault",
    "error",
    "vision",
}

YIELD_KEYWORDS = {
    "yield",
    "lot",
    "baseline",
    "hotspot",
    "trend",
    "scrap",
    "loss",
    "codeqty",
    "oldqty",
    "prodgroup",
}


def _contains_any(text: str, words: set[str]) -> bool:
    lowered = text.lower()
    return any(w in lowered for w in words)


def _supervisor_node():
    """Create a fast deterministic supervisor to avoid routing loops."""

    def supervisor(state: FactoryState) -> dict:
        messages = state["messages"]
        latest_user = ""
        latest_user_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if getattr(msg, "type", "") == "human":
                latest_user_idx = idx
                latest_user = str(getattr(msg, "content", ""))
                break

        turn_messages = messages[latest_user_idx + 1 :] if latest_user_idx >= 0 else []

        answered_by = {
            getattr(msg, "name", "")
            for msg in turn_messages
            if getattr(msg, "type", "") == "ai"
        }

        wants_tech = _contains_any(latest_user, TECH_KEYWORDS)
        wants_yield = _contains_any(latest_user, YIELD_KEYWORDS)

        nxt = "FINISH"
        reason = "No more specialist actions needed."

        if "agent_technician" not in answered_by and (wants_tech or not wants_yield):
            nxt = "agent_technician"
            reason = "Handle line status/tickets/manual troubleshooting first."
        elif "agent_yield" not in answered_by and wants_yield:
            nxt = "agent_yield"
            reason = "Add yield analysis for trend and hotspot context."

        note = f"[Floor Supervisor] → {nxt}: {reason}"
        return {"next": nxt, "messages": [AIMessage(content=note, name="floor_supervisor")]}

    return supervisor


def _worker_node(agent, name: str):
    """Wrap a ReAct agent as a graph node that reports back to the supervisor."""

    def worker(state: FactoryState) -> dict:
        result = agent.invoke({"messages": state["messages"]})
        # Surface only the agent's final answer on the shared scratchpad; its
        # intermediate tool calls are captured in the audit log.
        final = result["messages"][-1]
        return {"messages": [AIMessage(content=final.content, name=name)]}

    return worker


def build_graph(model=None, checkpointer: Any | None = None):
    """Compile and return the runnable multi-agent graph.

    Pass a ``model`` to inject your own (tests pass a dummy-key ChatOpenAI so the
    graph compiles offline); otherwise one is built from your settings. Optionally
    pass a LangGraph ``checkpointer`` to persist conversation state by thread id.
    """
    model = model or build_chat_model()

    builder = StateGraph(FactoryState)
    builder.add_node("supervisor", _supervisor_node())
    builder.add_node("agent_technician", _worker_node(build_technician_agent(model), "agent_technician"))
    builder.add_node("agent_yield", _worker_node(build_yield_agent(model), "agent_yield"))

    builder.add_edge(START, "supervisor")
    # Every worker hands control back to the supervisor — this forms the cycle.
    for worker in WORKERS:
        builder.add_edge(worker, "supervisor")
    # The supervisor's typed decision selects the next node (or ends the run).
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "agent_technician": "agent_technician",
            "agent_yield": "agent_yield",
            "FINISH": END,
        },
    )
    return builder.compile(checkpointer=checkpointer)
