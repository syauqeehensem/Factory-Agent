"""Tools for the entity-driven TCB chatbot flow.

Data tools are read-only and scoped to files under ``data/`` (status.csv,
mtp.csv, yield.csv, technician PDFs). The single action tool,
``create_mtp_ticket``, is used by the Escalation step and records a simulated
ticket in memory so nothing external is mutated.
"""

from __future__ import annotations

from langchain_core.tools import tool

from .manual_data import MANUAL_INDEX
from .project_data import PROJECT_DATA
from .tickets import TICKET_STORE
from .yield_data import YIELD_DATASET


def _data_status_text() -> str:
    return (
        f"{PROJECT_DATA.health_report()} "
        f"{YIELD_DATASET.status_text()} "
        f"{MANUAL_INDEX.status_text()}"
    )


@tool
def get_data_status() -> str:
    """Report availability of status, ticket, yield, and technician-manual data. Read-only."""
    return _data_status_text()


@tool
def refresh_data() -> str:
    """Reload status.csv, mtp.csv, yield.csv, and technician manuals from data/. Read-only."""
    PROJECT_DATA.reload()
    YIELD_DATASET.reload()
    MANUAL_INDEX.reload()
    return _data_status_text()


@tool
def get_entity_status(entity: str) -> str:
    """Return the current UP/DOWN status for one entity code (e.g. TCB706). Read-only."""
    return PROJECT_DATA.entity_status(entity)


@tool
def get_line_status_snapshot(max_down: int = 8) -> str:
    """Summarize how many entities are UP vs DOWN from status.csv. Read-only."""
    return PROJECT_DATA.status_snapshot(max_down=max_down)


@tool
def get_entity_ticket_summary(entity: str, limit: int = 6) -> str:
    """List the MTP ticket(s) and error message(s) for one entity from mtp.csv. Read-only."""
    return PROJECT_DATA.entity_ticket_summary(entity, limit=limit)


def _best_manual_query_for_entity(entity: str) -> str:
    """Choose the best troubleshooting query using known ticket errors first."""
    key = entity.strip().upper()
    rows = [r for r in PROJECT_DATA.ticket_rows if r.entity == key]
    for row in rows:
        err = (row.error or "").strip()
        if err and err.lower() != "unknown":
            return err
    if rows:
        fallback = (rows[0].error or "").strip()
        if fallback:
            return fallback
    return f"{key} troubleshooting"


@tool
def get_entity_full_context(entity: str, manual_top_k: int = 2) -> str:
    """Return integrated status+ticket+yield+manual evidence for one entity. Read-only."""
    key = entity.strip().upper()
    if not key:
        return "Please provide an entity code, e.g. TCB706 or TSX509."

    top_k = max(1, min(int(manual_top_k), 4))
    manual_query = _best_manual_query_for_entity(key)

    status_text = PROJECT_DATA.entity_status(key)
    ticket_text = PROJECT_DATA.entity_ticket_summary(key, limit=4)
    yield_text = YIELD_DATASET.entity_yield_text(key)
    manual_text = MANUAL_INDEX.search(manual_query, top_k=top_k)

    return (
        f"Integrated context for {key}:\n"
        f"- Status: {status_text}\n"
        f"- Tickets: {ticket_text}\n"
        f"- Yield: {yield_text}\n"
        f"- Manual evidence (query: {manual_query}): {manual_text}"
    )


@tool
def search_technician_manuals(question: str, top_k: int = 3) -> str:
    """Retrieve troubleshooting snippets from technician manuals for an error/symptom. Read-only."""
    k = max(1, min(int(top_k), 8))
    return MANUAL_INDEX.search(question, top_k=k)


@tool
def list_technician_documents() -> str:
    """List the indexed technician manual files under data/. Read-only."""
    MANUAL_INDEX.ensure_loaded()
    if MANUAL_INDEX.load_error:
        return f"Technician manuals unavailable: {MANUAL_INDEX.load_error}"
    files = sorted({chunk.source.split("#", 1)[0] for chunk in MANUAL_INDEX.chunks})
    if not files:
        return "No technician documents are indexed."
    return "Indexed technician docs:\n" + "\n".join(f"- {name}" for name in files)


@tool
def get_entity_yield(entity: str) -> str:
    """Return an entity's yield percent and whether it passes the yield goal. Read-only."""
    return YIELD_DATASET.entity_yield_text(entity)


@tool
def list_yield_below_goal(limit: int = 10) -> str:
    """List entities whose yield is below the goal threshold, lowest first. Read-only."""
    return YIELD_DATASET.worst_entities(limit=limit)


@tool
def create_mtp_ticket(entity: str, reason: str, ticket_type: str = "down-tool") -> str:
    """Create a simulated MTP/down-tool ticket for an entity and return its id. Action."""
    ticket = TICKET_STORE.create(entity=entity, ticket_type=ticket_type, reason=reason)
    return (
        f"Created {ticket.ticket_type} ticket {ticket.ticket_id} for {ticket.entity} "
        f"({ticket.created_at}). Reason: {ticket.reason or 'n/a'}."
    )


# Tool sets bound to each node in the graph.
TECHNICIAN_TOOLS = [
    get_data_status,
    get_entity_full_context,
    get_entity_status,
    get_entity_ticket_summary,
    get_entity_yield,
    search_technician_manuals,
    list_technician_documents,
]

YIELD_TOOLS = [
    get_data_status,
    get_entity_full_context,
    get_entity_status,
    get_entity_ticket_summary,
    get_entity_yield,
    search_technician_manuals,
    list_yield_below_goal,
]

ESCALATION_TOOLS = [
    get_entity_full_context,
    get_entity_status,
    get_entity_ticket_summary,
    get_entity_yield,
    create_mtp_ticket,
]
