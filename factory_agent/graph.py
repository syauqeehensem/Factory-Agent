"""Build the stateful, cyclic multi-agent graph.

This is the heart of the kit. The **Floor Supervisor** is a router node: it reads
the conversation so far and decides which specialist should act next, or that the
job is done. Each specialist node runs its ReAct agent, appends a summary to the
shared messages, and routes **back to the supervisor** — that loop is the cycle:

    START → supervisor ─(conditional)→ maintenance_scheduler ─→ supervisor
                       └───────────────→ yield_specialist ───────→ supervisor
                       └───────────────→ parts_procurement ────→ supervisor
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

from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .agents import (
    build_maintenance_agent,
    build_procurement_agent,
    build_yield_specialist_agent,
)
from .llm import build_chat_model
from .state import FactoryState

# The agents the supervisor can delegate to (plus the FINISH sentinel).
WORKERS = ["maintenance_scheduler", "yield_specialist", "parts_procurement"]
_ROUTE_OPTIONS = WORKERS + ["FINISH"]


class Router(TypedDict):
    """The Floor Supervisor's typed routing decision."""

    next: Literal["maintenance_scheduler", "yield_specialist", "parts_procurement", "FINISH"]
    reason: str


SUPERVISOR_PROMPT = (
    "You are the Floor Supervisor orchestrating a maintenance response. You manage "
    "three specialists:\n"
    "- maintenance_scheduler: triages alerts, creates and schedules work orders.\n"
    "- yield_specialist: analyzes historical/project yield data by machine/tool code.\n"
    "- parts_procurement: checks inventory and orders parts.\n\n"
    "Given the conversation so far, decide who should act NEXT to move the task "
    "forward, or answer FINISH when the issue is fully handled.\n"
    "Typical flow: maintenance_scheduler diagnoses and raises/schedules the work "
    "order (noting the part needed) → yield_specialist checks if yield evidence "
    "suggests an isolated vs systematic issue → parts_procurement ensures the needed "
    "part is in stock or ordered → FINISH.\n"
    "Conversation behavior:\n"
    "- Speak in a calm, colleague-like tone; avoid robotic template wording.\n"
    "- Keep continuity with prior turns and avoid repeating already-settled decisions.\n"
    "- Prefer concise routing choices with practical reasons.\n"
    "- If the user asks about TSX/TCB line entities, prefer maintenance_scheduler first "
    "so it can use Project Data status/ticket tools before deeper delegation.\n"
    "- If the user asks a broad follow-up, delegate to the best next specialist rather than restarting the entire flow.\n"
    "Do not pick an agent that has already completed its part. Choose FINISH once a "
    "work order is scheduled, yield assessment is complete when relevant, AND the "
    "required part is in stock or on order (or is blocked pending your approval)."
)


def _supervisor_node(model):
    """Create the supervisor node function bound to a model."""
    router_model = model.with_structured_output(Router)

    def supervisor(state: FactoryState) -> dict:
        decision = router_model.invoke(
            [SystemMessage(content=SUPERVISOR_PROMPT), *state["messages"]]
        )
        nxt = decision["next"]
        note = f"[Floor Supervisor] → {nxt}: {decision['reason']}"
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
    builder.add_node("supervisor", _supervisor_node(model))
    builder.add_node("maintenance_scheduler", _worker_node(build_maintenance_agent(model), "maintenance_scheduler"))
    builder.add_node("yield_specialist", _worker_node(build_yield_specialist_agent(model), "yield_specialist"))
    builder.add_node("parts_procurement", _worker_node(build_procurement_agent(model), "parts_procurement"))

    builder.add_edge(START, "supervisor")
    # Every worker hands control back to the supervisor — this forms the cycle.
    for worker in WORKERS:
        builder.add_edge(worker, "supervisor")
    # The supervisor's typed decision selects the next node (or ends the run).
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {"maintenance_scheduler": "maintenance_scheduler",
            "yield_specialist": "yield_specialist",
         "parts_procurement": "parts_procurement",
         "FINISH": END},
    )
    return builder.compile(checkpointer=checkpointer)
