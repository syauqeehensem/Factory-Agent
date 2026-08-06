"""TCB Chatbot — Project-Data-only two-agent orchestration.

A small, readable LangGraph app where a supervisor routes between:

- Agent Technician: status/ticket/manual troubleshooting
- Agent Yield: yield trend and hotspot analysis

All evidence comes from files under Project Data.
"""

from .graph import build_graph

__all__ = ["build_graph"]
__version__ = "0.1.0"
