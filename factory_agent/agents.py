"""The specialist agents.

Each specialist is a LangGraph **ReAct agent** built with the prebuilt
``create_react_agent`` helper: give it a model, a set of tools, and a system
prompt, and it runs the Reason-Act loop for you (think → call tool → observe →
repeat → answer). That keeps this kickstart focused on the *orchestration* — how
agents collaborate — rather than re-implementing tool-calling plumbing.

The Floor Supervisor is NOT here: it is a router, not a tool-using worker, so it
lives in ``graph.py`` where the routing logic belongs.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from .tools import TECHNICIAN_TOOLS, YIELD_TOOLS

TECHNICIAN_PROMPT = (
    "You are Agent Technician for a manufacturing line. Your job is to help with "
    "real troubleshooting using ONLY Project Data (status.csv, mtp.csv, and "
    "Technician manuals).\n"
    "Communication style:\n"
    "- Talk like a helpful teammate on shift, not like a robot.\n"
    "- Prefer plain language over rigid templates.\n"
    "- Ask one clarifying question only when needed; otherwise give direct help.\n"
    "- Start with get_project_data_status when data health is unknown.\n"
    "- For line/entity incidents, use get_line_status_snapshot and summarize_open_tickets first, "
    "then drill into get_entity_status / get_entity_ticket_summary.\n"
    "- For procedure/how-to issues, use search_technician_manuals and cite source file names.\n"
    "- Never invent data that is not in Project Data.\n"
    "- End with a concise next-action checklist (2-4 bullets) when appropriate."
)

YIELD_PROMPT = (
    "You are Agent Yield for manufacturing analytics. Your job is to explain yield "
    "patterns using ONLY Project Data yield records.\n"
    "Communication style:\n"
    "- Explain results naturally in plain language and include key numbers.\n"
    "- Start with get_yield_dataset_status when needed.\n"
    "- Use get_tool_yield_summary, list_yield_hotspots, get_lot_yield_summary, and "
    "summarize_recent_yield_vs_baseline for evidence.\n"
    "- If line/ticket context matters, use get_line_status_snapshot and summarize_open_tickets.\n"
    "- Do not invent values and do not reference non-Project-Data sources.\n"
    "- End with one clear recommendation the team can act on today."
)


def build_technician_agent(model):
    """The Technician ReAct agent."""
    return create_react_agent(
        model, TECHNICIAN_TOOLS, prompt=TECHNICIAN_PROMPT, name="agent_technician"
    )


def build_yield_agent(model):
    """The Yield ReAct agent."""
    return create_react_agent(
        model, YIELD_TOOLS, prompt=YIELD_PROMPT, name="agent_yield"
    )
