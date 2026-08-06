"""The shared state that flows through the graph.

In LangGraph, every node receives the current state and returns a partial update.
A *reducer* decides how each update is merged in. Here:

* ``messages`` uses the built-in :func:`add_messages` reducer, so each node's
  messages are *appended* to the running conversation (not overwritten). This is
  the shared "scratchpad" all agents read and write.
* ``next`` is a plain field the Floor Supervisor sets to name the agent that
  should act next (or ``"FINISH"``). Plain fields are simply replaced.

Keeping graph state this small is deliberate: source data lives behind tools
that read Project Data files (status/tickets/yield/manuals). The graph carries
the conversation and routing decision; data tools provide the evidence.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class FactoryState(TypedDict):
    """State shared across the Floor Supervisor and specialist agents."""

    # The growing conversation/scratchpad. add_messages appends + de-duplicates.
    messages: Annotated[list[AnyMessage], add_messages]
    # The status check writes the next node to run, or "FINISH".
    next: str
    # The entity under review and its looked-up UP/DOWN status.
    entity: str
    status: str
