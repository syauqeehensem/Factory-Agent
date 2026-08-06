"""Yield-data integration (simple per-entity schema).

Loads ``data/yield.csv`` (columns: ``entity,yield``) and exposes lightweight
helpers the Yield agent uses to compare a tool's yield against the configured
goal (``settings.yield_threshold``). Standard library only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .config import settings


@dataclass(frozen=True)
class YieldRecord:
    entity: str
    yield_value: float


class YieldDataset:
    """In-memory view of per-entity yield records."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.records: list[YieldRecord] = []
        self.load_error: str | None = None
        self.reload()

    def reload(self) -> None:
        self.records = []
        self.load_error = None

        if not self.csv_path.exists():
            self.load_error = f"File not found at {self.csv_path}"
            return

        try:
            with self.csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    entity = (row.get("entity") or "").strip().upper()
                    if not entity:
                        continue
                    self.records.append(
                        YieldRecord(
                            entity=entity,
                            yield_value=_parse_float((row.get("yield") or "").strip()),
                        )
                    )
        except Exception as exc:  # noqa: BLE001 - surface file parsing errors in tools
            self.load_error = f"Unable to parse CSV: {exc}"

    @property
    def is_ready(self) -> bool:
        return self.load_error is None and bool(self.records)

    def status_text(self) -> str:
        if self.load_error:
            return f"Yield dataset unavailable: {self.load_error}"
        if not self.records:
            return "Yield dataset is empty."
        avg_yield = sum(r.yield_value for r in self.records) / len(self.records)
        entities = len({r.entity for r in self.records})
        return (
            f"Yield dataset loaded ({len(self.records)} rows, {entities} entities, "
            f"average yield {avg_yield:.1f}%)."
        )

    def entity_yield(self, entity: str) -> float | None:
        """Return the yield percent for an entity, or None if unknown."""
        key = entity.strip().upper()
        row = next((r for r in self.records if r.entity == key), None)
        return row.yield_value if row else None

    def entity_yield_text(self, entity: str, threshold: float | None = None) -> str:
        """Human-readable yield verdict for an entity vs the goal threshold."""
        if self.load_error:
            return f"Yield dataset unavailable: {self.load_error}"
        key = entity.strip().upper()
        value = self.entity_yield(key)
        if value is None:
            known = ", ".join(sorted({r.entity for r in self.records})[:20])
            return f"No yield record for {key}. Known entities: {known}."
        goal = settings.yield_threshold if threshold is None else threshold
        if value >= goal:
            return (
                f"{key}: yield {value:.1f}% vs goal {goal:.0f}% -> PASS "
                f"(at or above goal; continue sustaining, no action needed)."
            )
        return (
            f"{key}: yield {value:.1f}% vs goal {goal:.0f}% -> FAIL "
            f"(below goal; performance not met, a down-tool/MTP ticket is required)."
        )

    def worst_entities(self, threshold: float | None = None, limit: int = 10) -> str:
        """List entities below the goal threshold, lowest yield first."""
        if self.load_error:
            return f"Yield dataset unavailable: {self.load_error}"
        goal = settings.yield_threshold if threshold is None else threshold
        below = sorted(
            ((r.entity, r.yield_value) for r in self.records if r.yield_value < goal),
            key=lambda x: x[1],
        )[: max(1, limit)]
        if not below:
            return f"No entities are below the {goal:.0f}% yield goal."
        lines = [f"Entities below the {goal:.0f}% yield goal (lowest first):"]
        lines.extend(f"- {entity}: {value:.1f}%" for entity, value in below)
        return "\n".join(lines)


def _parse_float(value: str) -> float:
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


YIELD_DATASET = YieldDataset(settings.yield_csv_path)
