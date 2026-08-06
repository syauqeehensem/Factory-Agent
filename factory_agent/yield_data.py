"""Yield-data integration for the Factory Agent workspace.

This module loads the CSV in Project Data/Yield and exposes lightweight
analytics helpers that tools can call. It is intentionally dependency-free
(standard library only) so it runs in the same environment as the rest of the
starter kit.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .config import settings


@dataclass(frozen=True)
class YieldRecord:
    lot: str
    operation: str
    out_date: datetime | None
    oldqty1: int
    prodgroup3: str
    entity: str
    codeqty: int
    yield_value: float


class YieldDataset:
    """In-memory view of yield CSV records plus common aggregations."""

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
                    self.records.append(
                        YieldRecord(
                            lot=(row.get("lot") or "").strip(),
                            operation=(row.get("operation") or "").strip(),
                            out_date=_parse_dt((row.get("out_date") or "").strip()),
                            oldqty1=_parse_int((row.get("oldqty1") or "").strip()),
                            prodgroup3=(row.get("prodgroup3") or "").strip(),
                            entity=(row.get("entity") or "").strip().upper(),
                            codeqty=_parse_int((row.get("codeqty") or "").strip()),
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

        dates = [r.out_date for r in self.records if r.out_date is not None]
        date_min = min(dates).strftime("%Y-%m-%d %H:%M") if dates else "n/a"
        date_max = max(dates).strftime("%Y-%m-%d %H:%M") if dates else "n/a"
        entities = {r.entity for r in self.records if r.entity}
        avg_yield = sum(r.yield_value for r in self.records) / len(self.records)

        return (
            f"Yield dataset loaded. Rows: {len(self.records)}, tools/entities: {len(entities)}, "
            f"date range: {date_min} -> {date_max}, average yield: {avg_yield:.6f}."
        )

    def tool_summary(self, entity: str) -> str:
        if self.load_error:
            return f"Yield dataset unavailable: {self.load_error}"

        key = entity.strip().upper()
        rows = [r for r in self.records if r.entity == key]
        if not rows:
            known = sorted({r.entity for r in self.records if r.entity})[:20]
            return f"No yield rows found for '{key}'. Known entities include: {known}"

        total_qty = sum(r.oldqty1 for r in rows)
        total_code = sum(r.codeqty for r in rows)
        avg_yield = sum(r.yield_value for r in rows) / len(rows)

        dates = [r.out_date for r in rows if r.out_date is not None]
        date_min = min(dates).strftime("%Y-%m-%d %H:%M") if dates else "n/a"
        date_max = max(dates).strftime("%Y-%m-%d %H:%M") if dates else "n/a"

        computed = (total_code / total_qty) if total_qty else 0.0
        return (
            f"{key}: lots={len(rows)}, summed oldqty1={total_qty}, summed codeqty={total_code}, "
            f"avg(yield column)={avg_yield:.6f}, computed codeqty/oldqty1={computed:.6f}, "
            f"date range={date_min} -> {date_max}."
        )

    def lot_summary(self, lot: str) -> str:
        if self.load_error:
            return f"Yield dataset unavailable: {self.load_error}"

        key = lot.strip().upper()
        row = next((r for r in self.records if r.lot.upper() == key), None)
        if row is None:
            return f"No yield row found for lot '{key}'."

        out_dt = row.out_date.strftime("%Y-%m-%d %H:%M") if row.out_date else "n/a"
        computed = (row.codeqty / row.oldqty1) if row.oldqty1 else 0.0
        return (
            f"Lot {row.lot}: operation={row.operation}, entity={row.entity}, out_date={out_dt}, "
            f"oldqty1={row.oldqty1}, codeqty={row.codeqty}, "
            f"yield column={row.yield_value:.6f}, computed codeqty/oldqty1={computed:.6f}."
        )

    def hotspot_table(self, max_avg_yield: float = 0.01, min_lots: int = 5, limit: int = 10) -> str:
        if self.load_error:
            return f"Yield dataset unavailable: {self.load_error}"

        if limit < 1:
            limit = 1
        if limit > 50:
            limit = 50

        buckets: dict[str, dict[str, float]] = defaultdict(
            lambda: {"lots": 0.0, "sum_yield": 0.0, "sum_qty": 0.0, "sum_code": 0.0}
        )
        for r in self.records:
            if not r.entity:
                continue
            b = buckets[r.entity]
            b["lots"] += 1
            b["sum_yield"] += r.yield_value
            b["sum_qty"] += r.oldqty1
            b["sum_code"] += r.codeqty

        rows: list[tuple[str, float, float, int, int]] = []
        for entity, b in buckets.items():
            lots = int(b["lots"])
            if lots < min_lots:
                continue
            avg_yield = b["sum_yield"] / lots
            computed = (b["sum_code"] / b["sum_qty"]) if b["sum_qty"] else 0.0
            if avg_yield >= max_avg_yield:
                rows.append((entity, avg_yield, computed, lots, int(b["sum_qty"])))

        rows.sort(key=lambda x: (x[1], x[3]), reverse=True)
        rows = rows[:limit]

        if not rows:
            return (
                f"No entities found with avg yield >= {max_avg_yield:.6f} "
                f"and at least {min_lots} lots."
            )

        lines = [
            f"Yield hotspots (avg yield >= {max_avg_yield:.6f}, min lots {min_lots}):"
        ]
        for entity, avg_y, comp_y, lots, total_qty in rows:
            lines.append(
                f"- {entity}: lots={lots}, avg(yield)={avg_y:.6f}, "
                f"computed={comp_y:.6f}, total oldqty1={total_qty}"
            )
        return "\n".join(lines)

    def recent_vs_baseline_summary(self, hours: int = 24, top_n: int = 3) -> str:
        """Compare a recent time window against the full timeline baseline."""
        if self.load_error:
            return f"Yield dataset unavailable: {self.load_error}"
        if not self.records:
            return "Yield dataset is empty."

        if hours < 1:
            hours = 1
        if hours > 24 * 30:
            hours = 24 * 30

        if top_n < 1:
            top_n = 1
        if top_n > 10:
            top_n = 10

        dated_rows = [r for r in self.records if r.out_date is not None]
        if not dated_rows:
            return (
                "Yield dataset has no parseable out_date values; "
                "cannot build a timeline-based baseline comparison."
            )

        end_dt = max(r.out_date for r in dated_rows)
        start_dt = end_dt - timedelta(hours=hours)
        recent_rows = [r for r in dated_rows if r.out_date is not None and r.out_date >= start_dt]
        if not recent_rows:
            return (
                f"No rows found in the last {hours} hours of the dataset timeline "
                f"({start_dt.strftime('%Y-%m-%d %H:%M')} -> {end_dt.strftime('%Y-%m-%d %H:%M')})."
            )

        def _agg(rows: list[YieldRecord]) -> dict[str, float]:
            lots = float(len(rows))
            sum_qty = float(sum(r.oldqty1 for r in rows))
            sum_code = float(sum(r.codeqty for r in rows))
            avg_yield = (sum(r.yield_value for r in rows) / lots) if lots else 0.0
            computed = (sum_code / sum_qty) if sum_qty else 0.0
            return {
                "lots": lots,
                "sum_qty": sum_qty,
                "sum_code": sum_code,
                "avg_yield": avg_yield,
                "computed": computed,
            }

        def _pct_change(current: float, baseline: float) -> str:
            if baseline == 0:
                return "n/a"
            return f"{((current - baseline) / baseline) * 100:+.2f}%"

        baseline = _agg(dated_rows)
        recent = _agg(recent_rows)

        baseline_avg = baseline["avg_yield"]
        recent_avg = recent["avg_yield"]
        baseline_comp = baseline["computed"]
        recent_comp = recent["computed"]

        avg_delta = recent_avg - baseline_avg
        comp_delta = recent_comp - baseline_comp

        dates = [r.out_date for r in dated_rows if r.out_date is not None]
        timeline_min = min(dates).strftime("%Y-%m-%d %H:%M")
        timeline_max = max(dates).strftime("%Y-%m-%d %H:%M")

        lines = [
            f"Timeline baseline comparison (window: last {hours}h in dataset timeline).",
            f"Timeline range: {timeline_min} -> {timeline_max}.",
            f"Rows compared: recent={int(recent['lots'])}, baseline={int(baseline['lots'])}.",
            (
                f"avg(yield): recent={recent_avg:.6f}, baseline={baseline_avg:.6f}, "
                f"delta={avg_delta:+.6f} ({_pct_change(recent_avg, baseline_avg)})."
            ),
            (
                f"computed codeqty/oldqty1: recent={recent_comp:.6f}, baseline={baseline_comp:.6f}, "
                f"delta={comp_delta:+.6f} ({_pct_change(recent_comp, baseline_comp)})."
            ),
        ]

        if recent_avg > baseline_avg:
            gap = recent_avg - baseline_avg
            avoidable_codeqty = int(round(gap * recent["sum_qty"]))
            lines.append(
                "Improvement room vs baseline: "
                f"reduce avg(yield) by {gap:.6f} to match baseline; "
                f"estimated avoidable codeqty over recent volume: {avoidable_codeqty}."
            )
        else:
            lines.append(
                "Performance note: recent avg(yield) is at or better than the baseline."
            )

        sorted_yields = sorted(r.yield_value for r in dated_rows)
        p25 = sorted_yields[int(0.25 * (len(sorted_yields) - 1))]
        if recent_avg > p25:
            gap_to_p25 = recent_avg - p25
            potential_pct = (gap_to_p25 / recent_avg * 100) if recent_avg else 0.0
            avoidable_to_p25 = int(round(gap_to_p25 * recent["sum_qty"]))
            lines.append(
                f"Improvement room vs best quartile (P25={p25:.6f}): "
                f"reduce by {gap_to_p25:.6f} ({potential_pct:.2f}% of recent avg); "
                f"estimated avoidable codeqty: {avoidable_to_p25}."
            )
        else:
            lines.append(
                f"Recent avg(yield) is already at or better than best-quartile target (P25={p25:.6f})."
            )

        all_by_entity: dict[str, dict[str, float]] = defaultdict(
            lambda: {"lots": 0.0, "sum_yield": 0.0}
        )
        recent_by_entity: dict[str, dict[str, float]] = defaultdict(
            lambda: {"lots": 0.0, "sum_yield": 0.0}
        )

        for row in dated_rows:
            if not row.entity:
                continue
            b = all_by_entity[row.entity]
            b["lots"] += 1
            b["sum_yield"] += row.yield_value

        for row in recent_rows:
            if not row.entity:
                continue
            b = recent_by_entity[row.entity]
            b["lots"] += 1
            b["sum_yield"] += row.yield_value

        opportunities: list[tuple[float, str, int, float, float]] = []
        for entity, recent_bucket in recent_by_entity.items():
            recent_lots = int(recent_bucket["lots"])
            if recent_lots < 2:
                continue
            all_bucket = all_by_entity.get(entity)
            if not all_bucket or all_bucket["lots"] <= 0:
                continue
            recent_entity_avg = recent_bucket["sum_yield"] / recent_bucket["lots"]
            baseline_entity_avg = all_bucket["sum_yield"] / all_bucket["lots"]
            gap = recent_entity_avg - baseline_entity_avg
            if gap > 0:
                opportunities.append(
                    (gap, entity, recent_lots, recent_entity_avg, baseline_entity_avg)
                )

        opportunities.sort(key=lambda x: x[0], reverse=True)
        if opportunities:
            lines.append(
                f"Top {min(top_n, len(opportunities))} entity opportunities "
                "(recent window worse than own baseline):"
            )
            for gap, entity, lots, recent_entity_avg, baseline_entity_avg in opportunities[:top_n]:
                pct = ((gap / baseline_entity_avg) * 100) if baseline_entity_avg else 0.0
                lines.append(
                    f"- {entity}: recent avg={recent_entity_avg:.6f}, "
                    f"baseline avg={baseline_entity_avg:.6f}, "
                    f"delta={gap:+.6f} ({pct:+.2f}%), recent lots={lots}"
                )
        else:
            lines.append(
                "Entity opportunity scan: none of the entities with at least 2 recent lots "
                "is worse than its own baseline."
            )

        undated = len(self.records) - len(dated_rows)
        if undated:
            lines.append(f"Note: ignored {undated} rows without parseable out_date.")

        return "\n".join(lines)


def _parse_int(value: str) -> int:
    if not value:
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def _parse_float(value: str) -> float:
    if not value:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


YIELD_DATASET = YieldDataset(settings.yield_csv_path)
