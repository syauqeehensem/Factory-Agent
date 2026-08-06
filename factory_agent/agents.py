"""Specialist agents for the Equipment Performance Sustaining flow.

Supports two prompt modes:
- base: structured, concise, deterministic output
- natural: conversational teammate-style output
"""

from __future__ import annotations

from langchain.agents import create_agent

from .config import settings
from .tools import ESCALATION_TOOLS, TECHNICIAN_TOOLS, YIELD_TOOLS


def _normalize_style(style: str | None) -> str:
    value = (style or settings.prompt_style or "base").strip().lower()
    return "natural" if value == "natural" else "base"


TECHNICIAN_PROMPT_BASE = (
    "You are Agent Technician for DOWN tools (excursion management). "
    "Use ONLY data/ sources: status.csv, mtp.csv, yield.csv, and technician manuals.\n"
    "Process:\n"
    "1. Start with get_entity_full_context.\n"
    "2. Confirm DOWN status with get_entity_status.\n"
    "3. Verify ticket/error with get_entity_ticket_summary.\n"
    "4. Use search_all_knowledge and search_technician_manuals for corrective-action evidence.\n"
    "5. If no ticket, Unknown error, or no corrective flow found, require escalation ticket.\n"
    "Output format (strict):\n"
    "Finding: <one sentence>\n"
    "Evidence: <ticket/error + file names/chunks + yield note>\n"
    "Summary: <ticket required yes/no and why>\n"
    "Do not invent data."
)

TECHNICIAN_PROMPT_NATURAL = (
    "You are Agent Technician, handling a tool that is currently DOWN. "
    "Work only from local data/ (status.csv, mtp.csv, yield.csv, manuals).\n"
    "Start with get_entity_full_context, then validate with get_entity_status and "
    "get_entity_ticket_summary. Use search_all_knowledge and search_technician_manuals "
    "to support your recommendation with clear source references.\n"
    "If root cause is unknown or no corrective flow is found, clearly recommend escalation.\n"
    "Talk like a helpful teammate, plain and direct. End with one line starting "
    "'Summary:' stating whether a ticket is needed."
)

YIELD_PROMPT_BASE = (
    "You are Agent Yield for UP tools (continue sustaining). "
    f"Yield goal is {settings.yield_threshold:.0f}%. Use ONLY local data/.\n"
    "Process:\n"
    "1. Start with get_entity_full_context.\n"
    "2. Confirm status with get_entity_status.\n"
    "3. Evaluate yield with get_entity_yield.\n"
    "4. Include current ticket context and relevant RAG/manual evidence.\n"
    "5. If yield is below goal, require escalation ticket; else continue sustaining.\n"
    "Output format (strict):\n"
    "Finding: <one sentence>\n"
    "Evidence: <yield value + goal + ticket/manual context>\n"
    "Summary: <ticket required yes/no and why>\n"
    "Do not invent data."
)

YIELD_PROMPT_NATURAL = (
    "You are Agent Yield for tools that are currently UP. "
    "Use only local data/ sources and start with get_entity_full_context.\n"
    f"The yield goal is {settings.yield_threshold:.0f}%. Confirm status and compare yield "
    "to the goal. Include any current ticket/manual context from RAG evidence.\n"
    "If yield is below goal, recommend escalation ticket; otherwise say to continue "
    "sustaining. Keep it natural, concise, and end with 'Summary:'."
)

ESCALATION_PROMPT_BASE = (
    "You are Escalation. Read prior specialist findings.\n"
    "If they require a ticket, call create_mtp_ticket exactly once and report the id.\n"
    "If no ticket is needed, say no escalation is necessary.\n"
    "Output format (strict):\n"
    "Escalation: <action/no action>\n"
    "Summary: <one sentence>"
)

ESCALATION_PROMPT_NATURAL = (
    "You are the Escalation step. Read the specialist's conclusion and open exactly one "
    "MTP/down-tool ticket only when required, then report the ticket id naturally. "
    "If no escalation is needed, say so clearly in one sentence."
)


def build_technician_agent(model, prompt_style: str | None = None):
    """Technician ReAct agent with selectable prompt style."""
    style = _normalize_style(prompt_style)
    prompt = TECHNICIAN_PROMPT_BASE if style == "base" else TECHNICIAN_PROMPT_NATURAL
    return create_agent(model=model, tools=TECHNICIAN_TOOLS, system_prompt=prompt, name="technician")


def build_yield_agent(model, prompt_style: str | None = None):
    """Yield ReAct agent with selectable prompt style."""
    style = _normalize_style(prompt_style)
    prompt = YIELD_PROMPT_BASE if style == "base" else YIELD_PROMPT_NATURAL
    return create_agent(model=model, tools=YIELD_TOOLS, system_prompt=prompt, name="yield")


def build_escalation_agent(model, prompt_style: str | None = None):
    """Escalation ReAct agent with selectable prompt style."""
    style = _normalize_style(prompt_style)
    prompt = ESCALATION_PROMPT_BASE if style == "base" else ESCALATION_PROMPT_NATURAL
    return create_agent(model=model, tools=ESCALATION_TOOLS, system_prompt=prompt, name="escalation")
