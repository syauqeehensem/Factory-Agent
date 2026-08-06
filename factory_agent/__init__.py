"""TCB Chatbot — Equipment Performance Sustaining.

A small, readable LangGraph app. A deterministic status check reads an entity's
UP/DOWN state from data/ and routes:

- DOWN → Agent Technician (excursion management: MTP ticket, error, troubleshooting)
- UP   → Agent Yield (continue sustaining: yield vs goal)

Both hand off to a shared Escalation step that opens a ticket when required.
All evidence comes from files under data/.
"""

from .graph import build_graph

__all__ = ["build_graph"]
__version__ = "0.2.0"
