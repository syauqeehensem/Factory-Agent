"""The shared state that flows through the graph.

In LangGraph, every node receives the current state and returns a partial update.
A *reducer* decides how each update is merged in. Here:

* ``messages`` uses the built-in :func:`add_messages` reducer, so each node's
  messages are *appended* to the running conversation (not overwritten). This is
  the shared "scratchpad" all agents read and write.
* ``next`` is a plain field the Floor Supervisor sets to name the agent that
  should act next (or ``"FINISH"``). Plain fields are simply replaced.

Keeping graph state this small is deliberate: the *factory's* state (machines,
work orders, purchase orders) lives behind tools in ``mock_factory.py`` — exactly
as it would behind real factory software. The graph carries the conversation and
the routing decision; the tools carry the side effects.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class FactoryState(TypedDict):
    """State shared across the Floor Supervisor and specialist agents."""

    # The growing conversation/scratchpad. add_messages appends + de-duplicates.
    messages: Annotated[list[AnyMessage], add_messages]
    # The Floor Supervisor writes the next agent to run, or "FINISH".
    next: str
