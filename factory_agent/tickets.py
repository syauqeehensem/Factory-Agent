"""In-session ticket store for the escalation step.

The Escalation agent creates MTP / down-tool tickets when a specialist flags
that action is required (unknown root-cause downtime, no troubleshooting flow
found, or yield below goal). Tickets are simulated in memory for this workspace
so nothing external is mutated; each gets a deterministic, readable id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    entity: str
    ticket_type: str
    reason: str
    created_at: str


@dataclass
class TicketStore:
    """Tracks tickets created during the current session."""

    tickets: list[Ticket] = field(default_factory=list)
    _seq: int = 0

    def create(self, entity: str, ticket_type: str, reason: str) -> Ticket:
        self._seq += 1
        entity_key = entity.strip().upper() or "UNKNOWN"
        ticket = Ticket(
            ticket_id=f"MTP-{self._seq:04d}",
            entity=entity_key,
            ticket_type=ticket_type.strip() or "MTP",
            reason=reason.strip(),
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        self.tickets.append(ticket)
        return ticket

    def for_entity(self, entity: str) -> list[Ticket]:
        key = entity.strip().upper()
        return [t for t in self.tickets if t.entity == key]

    def reset(self) -> None:
        self.tickets.clear()
        self._seq = 0


TICKET_STORE = TicketStore()
