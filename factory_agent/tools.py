"""The tools agents use to sense and act on the factory.

Each function decorated with ``@tool`` becomes callable by an LLM agent: the
docstring and type hints are what the model reads to decide when and how to call
it, so they are written for the model as much as for you. Keep them crisp.

Tools return short strings — an LLM-friendly "receipt" the agent can reason about
and relay. Action tools go through the security layer for auditing and guardrails.
"""

from __future__ import annotations

from langchain_core.tools import tool

from . import security
from .mock_factory import WORLD
from .project_data import PROJECT_DATA
from .yield_data import YIELD_DATASET

# ===========================================================================
# Read-only tools (safe — they never change factory state)
# ===========================================================================


@tool
def list_machines() -> str:
    """List all machines on the floor with their current status. Read-only."""
    return "\n".join(
        f"{m.machine_id} ({m.name}): {m.status}"
        + (f" | yield entity: {m.yield_entity}" if m.yield_entity else "")
        for m in WORLD.machines.values()
    )


@tool
def read_sensor(machine_id: str) -> str:
    """Read the latest vibration, temperature, and runtime for a machine.

    Use this to triage an alert. Read-only. ``machine_id`` e.g. 'CNC-01'.
    """
    m = WORLD.get_machine(machine_id)
    if not m:
        return f"Unknown machine '{machine_id}'. Known machines: {list(WORLD.machines)}"
    r = m.reading
    yield_hint = f" Yield entity mapping: {m.yield_entity}." if m.yield_entity else ""
    return (
        f"{m.machine_id} ({m.name}): vibration {r.vibration_mm_s} mm/s "
        f"(limit {m.vibration_limit}), temperature {r.temperature_c} C "
        f"(limit {m.temp_limit}), runtime {r.runtime_hours} h. Status: {m.status}."
        f"{yield_hint}"
    )


@tool
def get_maintenance_history(machine_id: str) -> str:
    """Return recent maintenance notes for a machine to inform a diagnosis. Read-only."""
    # Tiny canned history; in production this reads your CMMS.
    history = {
        "CNC-01": "Spindle bearing (BRG-204) last replaced 1,800 runtime-hours ago; "
                  "vendor MTBF ~1,700 h. Prior high-vibration events preceded bearing wear.",
        "CNC-02": "No open issues. Routine PM completed recently.",
        "CONV-01": "Idler roller replaced 2 weeks ago; running within limits.",
    }
    return history.get(machine_id.upper(), "No maintenance history on file for this machine.")


@tool
def check_parts_inventory(part_number: str) -> str:
    """Check stock, unit cost, and reorder level for a part. Read-only.

    ``part_number`` e.g. 'BRG-204'.
    """
    p = WORLD.get_part(part_number)
    if not p:
        return f"Unknown part '{part_number}'. Known parts: {list(WORLD.parts)}"
    flag = " — BELOW reorder level" if p.on_hand <= p.reorder_level else ""
    return (
        f"{p.part_number} ({p.description}): {p.on_hand} on hand, "
        f"unit cost ${p.unit_cost:.2f}, reorder level {p.reorder_level}{flag}."
    )


@tool
def list_open_work_orders() -> str:
    """List currently open work orders (so agents don't duplicate them). Read-only."""
    open_wos = [w for w in WORLD.work_orders if w["status"] == "open"]
    if not open_wos:
        return "No open work orders."
    return "\n".join(
        f"{w['id']} {w['machine_id']} [{w['priority']}] {w['issue']} "
        f"(tech: {w['technician'] or 'unassigned'})"
        for w in open_wos
    )


@tool
def list_machine_yield_mappings() -> str:
    """List machine IDs and their mapped Yield CSV entity/tool codes. Read-only."""
    lines = []
    for m in WORLD.machines.values():
        mapped = m.yield_entity if m.yield_entity else "(unmapped)"
        lines.append(f"{m.machine_id} -> {mapped}")
    return "\n".join(lines)


@tool
def get_yield_dataset_status() -> str:
    """Report whether the project yield CSV is available and summarize its coverage. Read-only."""
    return YIELD_DATASET.status_text()


@tool
def get_tool_yield_summary(entity: str) -> str:
    """Summarize yield behavior for one tool/entity code (e.g. TSX501). Read-only."""
    return YIELD_DATASET.tool_summary(entity)


@tool
def get_machine_yield_summary(machine_id: str) -> str:
    """Summarize yield behavior for the entity mapped from a machine id. Read-only."""
    m = WORLD.get_machine(machine_id)
    if not m:
        return f"Unknown machine '{machine_id}'. Known machines: {list(WORLD.machines)}"
    if not m.yield_entity:
        return (
            f"Machine {m.machine_id} has no yield entity mapping. "
            "Use list_machine_yield_mappings to inspect mappings."
        )
    return f"{m.machine_id} mapped to {m.yield_entity}. {YIELD_DATASET.tool_summary(m.yield_entity)}"


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
    """Reload status.csv/mtp.csv/yield CSV from Project Data after files are updated. Read-only."""
    PROJECT_DATA.reload()
    YIELD_DATASET.reload()
    return (
        f"{PROJECT_DATA.health_report()} "
        f"{YIELD_DATASET.status_text()}"
    )


@tool
def get_project_data_status() -> str:
    """Report Project Data availability for status, ticket, and yield sources. Read-only."""
    return (
        f"{PROJECT_DATA.health_report()} "
        f"{YIELD_DATASET.status_text()}"
    )


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


# ===========================================================================
# Action tools (change factory state — audited, guarded)
# ===========================================================================


@tool
def create_work_order(machine_id: str, issue: str, priority: str = "medium") -> str:
    """Raise a maintenance work order in the CMMS. ACTION (writes state).

    ``priority`` is one of 'low', 'medium', 'high'. Returns the new work-order id.
    """
    if not WORLD.get_machine(machine_id):
        return f"Cannot create work order: unknown machine '{machine_id}'."
    priority = priority.lower() if priority.lower() in {"low", "medium", "high"} else "medium"
    wo = WORLD.add_work_order(machine_id, issue, priority)
    security.audit("maintenance_scheduler", "create_work_order",
                   f"{wo['id']} {wo['machine_id']} [{priority}] {issue}")
    return f"Created work order {wo['id']} for {wo['machine_id']} ({priority}): {issue}."


@tool
def schedule_maintenance(work_order_id: str, technician: str, when: str) -> str:
    """Assign a technician and time to an existing work order. ACTION (writes state).

    ``when`` is a free-text slot like 'today 16:00' or 'tomorrow morning'.
    """
    wo = WORLD.find_work_order(work_order_id)
    if not wo:
        return f"Cannot schedule: work order '{work_order_id}' not found."
    wo["technician"] = technician
    wo["scheduled_for"] = when
    wo["status"] = "scheduled"
    security.audit("maintenance_scheduler", "schedule_maintenance",
                   f"{wo['id']} -> {technician} @ {when}")
    return f"Scheduled {wo['id']} with {technician} for {when}."


@tool
def order_parts(part_number: str, quantity: int) -> str:
    """Raise a purchase order for a part. ACTION (writes state, spends money).

    Orders above the auto-approval limit are blocked pending Floor Supervisor
    approval — report that back rather than retrying.
    """
    p = WORLD.get_part(part_number)
    if not p:
        return f"Cannot order: unknown part '{part_number}'."
    if quantity <= 0:
        return "Quantity must be a positive integer."
    total = p.unit_cost * quantity
    allowed, reason = security.purchase_allowed(total)
    if not allowed:
        security.audit("parts_procurement", "order_parts",
                       f"{quantity}x {p.part_number} (${total:.2f}) — {reason}",
                       status="blocked")
        return (
            f"BLOCKED: ordering {quantity}x {p.part_number} costs ${total:.2f}. {reason}. "
            "Ask the Floor Supervisor to approve before proceeding."
        )
    po = WORLD.add_purchase_order(part_number, quantity, total)
    security.audit("parts_procurement", "order_parts",
                   f"{po['id']} {quantity}x {p.part_number} ${total:.2f} ({reason})")
    return (
        f"Ordered {quantity}x {p.part_number} — {po['id']}, ${total:.2f}, "
        f"ETA {po['eta_days']} days."
    )


# Convenient groupings the agents bind. Read-only tools are shared; action tools
# are scoped to the agent allowed to use them.
MAINTENANCE_TOOLS = [
    list_machines, read_sensor, get_maintenance_history, list_open_work_orders,
    refresh_project_data,
    get_project_data_status,
    get_line_status_snapshot,
    get_entity_status,
    summarize_open_tickets,
    get_entity_ticket_summary,
    list_machine_yield_mappings,
    get_yield_dataset_status, get_tool_yield_summary, get_machine_yield_summary,
    get_lot_yield_summary, list_yield_hotspots, summarize_recent_yield_vs_baseline,
    create_work_order, schedule_maintenance,
]

YIELD_TOOLS = [
    refresh_project_data,
    get_project_data_status,
    get_line_status_snapshot,
    get_entity_status,
    summarize_open_tickets,
    get_entity_ticket_summary,
    list_machines,
    list_machine_yield_mappings,
    get_yield_dataset_status,
    get_tool_yield_summary,
    get_machine_yield_summary,
    get_lot_yield_summary,
    list_yield_hotspots,
    summarize_recent_yield_vs_baseline,
]
PROCUREMENT_TOOLS = [
    check_parts_inventory, list_open_work_orders, order_parts,
]
