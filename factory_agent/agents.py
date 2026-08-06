"""The specialist agents for the Equipment Performance Sustaining flow.

Each is a LangGraph ReAct agent (``create_react_agent``): a model, a set of
tools, and a system prompt that encodes one branch of the status-driven logic.

- Agent Technician -> DOWN tools (excursion management): check the MTP ticket &
  error, then recommend an action from the troubleshooting docs.
- Agent Yield      -> UP tools (continue sustaining): check yield vs the goal.
- Escalation       -> shared action step: open an MTP/down-tool ticket when the
  specialist says one is required.
"""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from .config import settings
from .tools import ESCALATION_TOOLS, TECHNICIAN_TOOLS, YIELD_TOOLS

TECHNICIAN_PROMPT = (
    "You are Agent Technician, handling a tool that is currently DOWN (excursion "
    "management). Work only from data/ (status.csv, mtp.csv, technician manuals).\n"
    "Steps:\n"
    "1. Confirm the tool is DOWN with get_entity_status.\n"
    "2. Check its MTP ticket and error message with get_entity_ticket_summary.\n"
    "3. If a ticket with a real error exists, use search_technician_manuals with that "
    "error text to find a recommended action, and cite the source file. Say clearly "
    "whether a troubleshooting action flow was found.\n"
    "4. If there is NO ticket, or the error is 'Unknown', or no troubleshooting flow is "
    "found, say escalation is required for unknown root-cause downtime (an MTP ticket "
    "must be created).\n"
    "Talk like a helpful shift teammate, not a robot. Keep it to a few sentences and end "
    "with one line starting 'Summary:' stating the finding and whether a ticket is needed.\n"
    "Never invent data that is not in the files."
)

YIELD_PROMPT = (
    "You are Agent Yield, handling a tool that is currently UP (continue sustaining). "
    "Work only from data/ (status.csv, yield.csv).\n"
    f"The yield goal is {settings.yield_threshold:.0f}%.\n"
    "Steps:\n"
    "1. Confirm the tool is UP with get_entity_status.\n"
    "2. Check its yield with get_entity_yield (it returns the yield percent and a "
    "PASS/FAIL verdict versus the goal).\n"
    "3. If the yield FAILS (below the goal), say performance has not met the goal and a "
    "down-tool/MTP ticket is required.\n"
    "4. If the yield PASSES, say no action is needed — continue sustaining.\n"
    "Talk naturally and include the actual yield number. End with one line starting "
    "'Summary:' stating the yield and whether a ticket is needed.\n"
    "Never invent data that is not in the files."
)

ESCALATION_PROMPT = (
    "You are the Escalation step. Read the specialist's findings in the conversation.\n"
    "If they said a ticket is required (unknown root-cause downtime, no troubleshooting "
    "flow found, or yield below goal), call create_mtp_ticket(entity, reason, ticket_type) "
    "once to open a down-tool/MTP ticket, then report the ticket id in plain language and "
    "note whether a separate escalation ticket was necessary.\n"
    "If the specialist recommended a fix and no ticket is required, simply state that no "
    "escalation was necessary.\n"
    "Be concise (1-2 sentences) and natural. Do not create more than one ticket."
)


def build_technician_agent(model):
    """The Technician ReAct agent (DOWN path)."""
    return create_react_agent(
        model, TECHNICIAN_TOOLS, prompt=TECHNICIAN_PROMPT, name="technician"
    )


def build_yield_agent(model):
    """The Yield ReAct agent (UP path)."""
    return create_react_agent(model, YIELD_TOOLS, prompt=YIELD_PROMPT, name="yield")


def build_escalation_agent(model):
    """The Escalation ReAct agent (shared action step)."""
    return create_react_agent(
        model, ESCALATION_TOOLS, prompt=ESCALATION_PROMPT, name="escalation"
    )
