"""Project-Data-only tools for the TCB chatbot agents.

All tools in this module are read-only and scoped to files under Project Data.
No mock machines, inventory, or external systems are used.
"""

from __future__ import annotations

from langchain_core.tools import tool

from .manual_data import MANUAL_INDEX
from .project_data import PROJECT_DATA
from .yield_data import YIELD_DATASET

@tool
def get_yield_dataset_status() -> str:
    """Report whether the project yield CSV is available and summarize its coverage. Read-only."""
    return YIELD_DATASET.status_text()


@tool
def get_tool_yield_summary(entity: str) -> str:
    """Summarize yield behavior for one tool/entity code (e.g. TSX501). Read-only."""
    return YIELD_DATASET.tool_summary(entity)


@tool
def get_lot_yield_summary(lot: str) -> str:
    """Return yield details for a specific lot id from the project CSV. Read-only."""
    return YIELD_DATASET.lot_summary(lot)


@tool
def list_yield_hotspots(max_avg_yield: float = 0.01, min_lots: int = 5, limit: int = 10) -> str:
    """List tools/entities with elevated average yield in the project CSV. Read-only."""
    return YIELD_DATASET.hotspot_table(
        max_avg_yield=max_avg_yield, min_lots=min_lots, limit=limit
    )


@tool
def summarize_recent_yield_vs_baseline(hours: int = 24, top_n: int = 3) -> str:
    """Compare recent-window yield against full-timeline baseline and quantify improvement room. Read-only."""
    return YIELD_DATASET.recent_vs_baseline_summary(hours=hours, top_n=top_n)


@tool
def refresh_project_data() -> str:
    """Reload status.csv/mtp.csv/yield/manual files from Project Data. Read-only."""
    PROJECT_DATA.reload()
    YIELD_DATASET.reload()
    MANUAL_INDEX.reload()
    return (
        f"{PROJECT_DATA.health_report()} "
        f"{YIELD_DATASET.status_text()} "
        f"{MANUAL_INDEX.status_text()}"
    )


@tool
def get_project_data_status() -> str:
    """Report Project Data availability for status, ticket, yield, and technician manuals."""
    return (
        f"{PROJECT_DATA.health_report()} "
        f"{YIELD_DATASET.status_text()} "
        f"{MANUAL_INDEX.status_text()}"
    )


@tool
def get_technician_manual_status() -> str:
    """Report whether technician manuals are indexed and searchable."""
    return MANUAL_INDEX.status_text()


@tool
def get_line_status_snapshot(max_down: int = 8) -> str:
    """Summarize current line/entity UP/DOWN status from Project Data status.csv. Read-only."""
    return PROJECT_DATA.status_snapshot(max_down=max_down)


@tool
def get_entity_status(entity: str) -> str:
    """Return the latest status for one entity code (e.g. TSX509 or TCB702). Read-only."""
    return PROJECT_DATA.entity_status(entity)


@tool
def summarize_open_tickets(top_n: int = 5) -> str:
    """Summarize open ticket patterns from Project Data mtp.csv. Read-only."""
    return PROJECT_DATA.ticket_snapshot(top_n=top_n)


@tool
def get_entity_ticket_summary(entity: str, limit: int = 6) -> str:
    """List recent tickets for one entity from Project Data mtp.csv. Read-only."""
    return PROJECT_DATA.entity_ticket_summary(entity, limit=limit)


@tool
def list_technician_documents() -> str:
    """List indexed technician source files from Project Data/Technician. Read-only."""
    MANUAL_INDEX.ensure_loaded()
    if MANUAL_INDEX.load_error:
        return f"Technician manuals unavailable: {MANUAL_INDEX.load_error}"
    files = sorted({chunk.source.split("#", 1)[0] for chunk in MANUAL_INDEX.chunks})
    if not files:
        return "No technician documents are indexed."
    return "Indexed technician docs:\n" + "\n".join(f"- {name}" for name in files)


@tool
def search_technician_manuals(question: str, top_k: int = 3) -> str:
    """Retrieve top manual snippets relevant to a technician troubleshooting question."""
    k = max(1, min(int(top_k), 8))
    return MANUAL_INDEX.search(question, top_k=k)


# Tool sets bound to each specialist.
TECHNICIAN_TOOLS = [
    refresh_project_data,
    get_project_data_status,
    get_technician_manual_status,
    get_line_status_snapshot,
    get_entity_status,
    summarize_open_tickets,
    get_entity_ticket_summary,
    list_technician_documents,
    search_technician_manuals,
]

YIELD_TOOLS = [
    refresh_project_data,
    get_project_data_status,
    get_line_status_snapshot,
    get_entity_status,
    summarize_open_tickets,
    get_entity_ticket_summary,
    get_yield_dataset_status,
    get_tool_yield_summary,
    get_lot_yield_summary,
    list_yield_hotspots,
    summarize_recent_yield_vs_baseline,
]
