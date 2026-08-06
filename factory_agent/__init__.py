"""Factory Agent — a stateful multi-agent production-orchestration kickstart.

Capstone Level 2 (agentic track) starter. A small, readable LangGraph app where a
**Floor Supervisor** delegates to two specialist agents — a **Maintenance
Scheduler** and a **Parts Procurement Agent** — that collaborate in a *cyclic*
graph and act on the factory through **secure, audited tool calls** (read sensor
data, raise work orders, order parts).

It is built to be BOTH a learning tool and a project seed:

1. Learn the LangGraph mental model — State + reducers, nodes, conditional edges,
   and cycles (worker → supervisor → worker) — in one short, commented codebase.
2. Start your capstone from here — swap the simulated factory for real systems,
   add agents/tools, and tighten the security layer.

Run ``python run_demo.py`` for an end-to-end predictive-maintenance triage.
"""

from .graph import build_graph
from .mock_factory import WORLD

__all__ = ["build_graph", "WORLD"]
__version__ = "0.1.0"
