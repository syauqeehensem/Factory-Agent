"""Secure tool-calling layer: audit trail + guardrails.

Giving an LLM the ability to *act* on factory systems (raise work orders, spend
money) demands guardrails. This module centralizes them so the protections are in
one auditable place rather than scattered through the tools:

* **Audit log** — every action tool records who/what/when/result. In production
  this would be append-only and shipped to your SIEM; here it's an in-memory list.
* **Read vs. action classification** — read-only tools are always safe; action
  tools change state and are logged. The lists below document the boundary.
* **Spend guardrail** — purchases above the auto-approval limit are blocked and
  must be approved by the Floor Supervisor, demonstrating human-/policy-in-the-loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import settings

# Documentation of the trust boundary. Read-only tools cannot change state;
# action tools can and are always audited.
READ_ONLY_TOOLS = {
    "read_sensor", "get_machine_status", "list_machines",
    "list_machine_yield_mappings",
    "get_maintenance_history", "check_parts_inventory", "list_open_work_orders",
    "get_yield_dataset_status", "get_tool_yield_summary", "get_lot_yield_summary",
    "get_machine_yield_summary", "list_yield_hotspots", "summarize_recent_yield_vs_baseline",
}
ACTION_TOOLS = {"create_work_order", "schedule_maintenance", "order_parts"}


@dataclass
class AuditEntry:
    actor: str       # which agent performed the action
    action: str      # tool name
    details: str     # short human-readable description
    status: str      # "ok" | "blocked" | "error"
    ts: float = field(default_factory=time.time)

    def __str__(self) -> str:
        clock = time.strftime("%H:%M:%S", time.localtime(self.ts))
        flag = "" if self.status == "ok" else f"  <{self.status.upper()}>"
        return f"[{clock}] {self.actor}: {self.action} — {self.details}{flag}"


# In-memory audit trail shared across the run.
AUDIT_LOG: list[AuditEntry] = []


def audit(actor: str, action: str, details: str, status: str = "ok") -> AuditEntry:
    """Record an action and return the entry (also used as the tool's receipt)."""
    entry = AuditEntry(actor=actor, action=action, details=details, status=status)
    AUDIT_LOG.append(entry)
    return entry


def reset_audit() -> None:
    AUDIT_LOG.clear()


def purchase_allowed(total_cost: float) -> tuple[bool, str]:
    """Apply the spend guardrail. Returns (allowed, reason)."""
    limit = settings.auto_approve_limit
    if total_cost <= limit:
        return True, f"within ${limit:.0f} auto-approval limit"
    return (
        False,
        f"${total_cost:.2f} exceeds the ${limit:.0f} auto-approval limit; "
        "requires Floor Supervisor approval",
    )
