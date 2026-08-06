"""Project Data adapters for Factory Agent.

Reads status/ticket CSV files and exposes lightweight deterministic summaries
that can be used by tools and agents without extra dependencies.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .config import settings


@dataclass(frozen=True)
class StatusRow:
    entity: str
    status: str


@dataclass(frozen=True)
class TicketRow:
    entity: str
    ticket: str
    error: str


class ProjectDataStore:
    """In-memory view of status.csv and mtp.csv."""

    def __init__(self, status_csv_path: str | Path, mtp_csv_path: str | Path) -> None:
        self.status_csv_path = Path(status_csv_path)
        self.mtp_csv_path = Path(mtp_csv_path)
        self.status_rows: list[StatusRow] = []
        self.ticket_rows: list[TicketRow] = []
        self.status_error: str | None = None
        self.ticket_error: str | None = None
        self.reload()

    def reload(self) -> None:
        self.status_rows = []
        self.ticket_rows = []
        self.status_error = None
        self.ticket_error = None
        self._load_status()
        self._load_tickets()

    def health_report(self) -> str:
        status_part = (
            f"status.csv loaded ({len(self.status_rows)} rows)"
            if self.status_error is None
            else f"status.csv unavailable: {self.status_error}"
        )
        ticket_part = (
            f"mtp.csv loaded ({len(self.ticket_rows)} rows)"
            if self.ticket_error is None
            else f"mtp.csv unavailable: {self.ticket_error}"
        )
        return f"Project Data: {status_part}; {ticket_part}."

    def status_snapshot(self, max_down: int = 8) -> str:
        if self.status_error:
            return f"Line status unavailable: {self.status_error}"
        if not self.status_rows:
            return "Line status dataset is empty."

        total = len(self.status_rows)
        up_count = sum(1 for r in self.status_rows if r.status == "UP")
        down_entities = [r.entity for r in self.status_rows if r.status == "DOWN"]
        down_count = len(down_entities)
        availability = (up_count / total * 100.0) if total else 0.0

        preview = ", ".join(down_entities[:max_down]) if down_entities else "none"
        if down_count > max_down:
            preview += f", ... (+{down_count - max_down} more)"

        return (
            f"Line status snapshot: {up_count}/{total} entities UP "
            f"({availability:.1f}% availability), {down_count} DOWN. "
            f"DOWN entities: {preview}."
        )

    def entity_status(self, entity: str) -> str:
        key = entity.strip().upper()
        if not key:
            return "Please provide an entity id, e.g. TSX509 or TCB702."
        if self.status_error:
            return f"Line status unavailable: {self.status_error}"

        row = next((r for r in self.status_rows if r.entity == key), None)
        if row is None:
            known = ", ".join(sorted({r.entity for r in self.status_rows})[:20])
            return f"Entity {key} not found in status.csv. Known examples: {known}."
        return f"Entity {row.entity} is currently {row.status}."

    def ticket_snapshot(self, top_n: int = 5) -> str:
        if self.ticket_error:
            return f"Ticket dataset unavailable: {self.ticket_error}"
        if not self.ticket_rows:
            return "Ticket dataset is empty."

        error_counts = Counter((r.error or "Unknown") for r in self.ticket_rows)
        entity_counts = Counter(r.entity for r in self.ticket_rows if r.entity)

        top_errors = ", ".join(
            f"{name} ({count})" for name, count in error_counts.most_common(top_n)
        )
        top_entities = ", ".join(
            f"{name} ({count})" for name, count in entity_counts.most_common(top_n)
        )

        return (
            f"Ticket snapshot: {len(self.ticket_rows)} open rows. "
            f"Top error modes: {top_errors or 'none'}. "
            f"Most affected entities: {top_entities or 'none'}."
        )

    def entity_ticket_summary(self, entity: str, limit: int = 6) -> str:
        key = entity.strip().upper()
        if not key:
            return "Please provide an entity id, e.g. TSX509 or TCB702."
        if self.ticket_error:
            return f"Ticket dataset unavailable: {self.ticket_error}"

        rows = [r for r in self.ticket_rows if r.entity == key]
        if not rows:
            return f"No ticket rows found for entity {key}."

        lines = [f"Tickets for {key}: {len(rows)} row(s)."]
        for row in rows[:limit]:
            lines.append(f"- ticket {row.ticket}: {row.error or 'Unknown'}")
        if len(rows) > limit:
            lines.append(f"- ... (+{len(rows) - limit} more)")
        return "\n".join(lines)

    def _load_status(self) -> None:
        if not self.status_csv_path.exists():
            self.status_error = f"file not found at {self.status_csv_path}"
            return
        try:
            with self.status_csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    entity = (row.get("entity") or "").strip().upper()
                    status = (row.get("status") or "").strip().upper()
                    if not entity:
                        continue
                    if not status:
                        status = "UNKNOWN"
                    self.status_rows.append(StatusRow(entity=entity, status=status))
        except Exception as exc:  # noqa: BLE001
            self.status_error = f"unable to parse CSV: {exc}"

    def _load_tickets(self) -> None:
        if not self.mtp_csv_path.exists():
            self.ticket_error = f"file not found at {self.mtp_csv_path}"
            return
        try:
            with self.mtp_csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    entity = (row.get("entity") or "").strip().upper()
                    ticket = (row.get("ticket") or "").strip()
                    error = (row.get("error") or "").strip()
                    if not entity:
                        continue
                    self.ticket_rows.append(TicketRow(entity=entity, ticket=ticket, error=error))
        except Exception as exc:  # noqa: BLE001
            self.ticket_error = f"unable to parse CSV: {exc}"


PROJECT_DATA = ProjectDataStore(
    status_csv_path=settings.status_csv_path,
    mtp_csv_path=settings.mtp_csv_path,
)
