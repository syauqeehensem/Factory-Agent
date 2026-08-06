"""A simulated factory — sensors, inventory, work orders, purchase orders.

This stands in for the real "factory software systems" (CMMS, MES, ERP, sensor
historian) so the kit runs end-to-end with no integrations. The agents only ever
touch it through the tools in ``tools.py``; swapping this for real API calls is
the natural next step toward production.

Everything is in-memory and deliberately tiny. ``WORLD`` is a module-level
singleton the tools share; ``reset_world()`` restores the starting scenario
(handy for tests and re-running the demo).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field


@dataclass
class SensorReading:
    vibration_mm_s: float  # ISO 10816 style velocity; higher = rougher running
    temperature_c: float
    runtime_hours: int


@dataclass
class Machine:
    machine_id: str
    name: str
    reading: SensorReading
    vibration_limit: float  # above this = maintenance alert
    temp_limit: float
    # Optional mapping to Yield CSV entity/tool code (e.g. TSX501).
    yield_entity: str | None = None

    @property
    def status(self) -> str:
        if self.reading.vibration_mm_s > self.vibration_limit:
            return "ALERT: high vibration"
        if self.reading.temperature_c > self.temp_limit:
            return "ALERT: over-temperature"
        return "normal"


@dataclass
class Part:
    part_number: str
    description: str
    on_hand: int
    unit_cost: float
    reorder_level: int


@dataclass
class FactoryWorld:
    machines: dict[str, Machine] = field(default_factory=dict)
    parts: dict[str, Part] = field(default_factory=dict)
    work_orders: list[dict] = field(default_factory=list)
    purchase_orders: list[dict] = field(default_factory=list)
    _wo_seq: itertools.count = field(default_factory=lambda: itertools.count(1001))
    _po_seq: itertools.count = field(default_factory=lambda: itertools.count(5001))

    # -- read helpers -------------------------------------------------------
    def get_machine(self, machine_id: str) -> Machine | None:
        return self.machines.get(machine_id.upper())

    def get_part(self, part_number: str) -> Part | None:
        return self.parts.get(part_number.upper())

    # -- write helpers (called by ACTION tools) -----------------------------
    def add_work_order(self, machine_id: str, issue: str, priority: str) -> dict:
        wo = {
            "id": f"WO-{next(self._wo_seq)}",
            "machine_id": machine_id.upper(),
            "issue": issue,
            "priority": priority,
            "status": "open",
            "technician": None,
            "scheduled_for": None,
        }
        self.work_orders.append(wo)
        return wo

    def find_work_order(self, wo_id: str) -> dict | None:
        return next((w for w in self.work_orders if w["id"] == wo_id.upper()), None)

    def add_purchase_order(self, part_number: str, quantity: int, total_cost: float) -> dict:
        po = {
            "id": f"PO-{next(self._po_seq)}",
            "part_number": part_number.upper(),
            "quantity": quantity,
            "total_cost": round(total_cost, 2),
            "status": "ordered",
            "eta_days": 3,
        }
        self.purchase_orders.append(po)
        return po

    def summary(self) -> str:
        """A human-readable snapshot for the demo's closing report."""
        lines = ["Work orders:"]
        lines += [
            f"  {w['id']} {w['machine_id']} [{w['priority']}] {w['issue']} "
            f"(status: {w['status']}"
            + (f", tech: {w['technician']}" if w["technician"] else "")
            + (f", when: {w['scheduled_for']}" if w["scheduled_for"] else "")
            + ")"
            for w in self.work_orders
        ] or ["  (none)"]
        lines.append("Purchase orders:")
        lines += [
            f"  {p['id']} {p['quantity']}x {p['part_number']} ${p['total_cost']} "
            f"(status: {p['status']}, ETA {p['eta_days']}d)"
            for p in self.purchase_orders
        ] or ["  (none)"]
        return "\n".join(lines)


def _starting_world() -> FactoryWorld:
    """The default scenario: CNC-01 is running rough (high vibration)."""
    w = FactoryWorld()
    w.machines = {
        "CNC-01": Machine(
            "CNC-01", "Vertical Mill VMX-500",
            SensorReading(vibration_mm_s=7.8, temperature_c=72.0, runtime_hours=1850),
            vibration_limit=4.5, temp_limit=80.0,
        ),
        "CNC-02": Machine(
            "CNC-02", "Vertical Mill VMX-500",
            SensorReading(vibration_mm_s=2.1, temperature_c=58.0, runtime_hours=900),
            vibration_limit=4.5, temp_limit=80.0,
        ),
        "CONV-01": Machine(
            "CONV-01", "Belt Conveyor BX-200",
            SensorReading(vibration_mm_s=3.2, temperature_c=49.0, runtime_hours=4200),
            vibration_limit=6.0, temp_limit=70.0,
        ),
    }
    w.parts = {
        "BRG-204": Part("BRG-204", "Spindle bearing (VMX-500)", on_hand=0, unit_cost=320.0, reorder_level=2),
        "BELT-11": Part("BELT-11", "Conveyor drive belt (BX-200)", on_hand=3, unit_cost=85.0, reorder_level=2),
        "FLT-120": Part("FLT-120", "Coolant filter (VMX-500)", on_hand=6, unit_cost=25.0, reorder_level=4),
    }
    return w


# The shared singleton the tools operate on.
WORLD = _starting_world()


def reset_world() -> None:
    """Restore the starting scenario (used by the demo and tests)."""
    global WORLD
    fresh = _starting_world()
    WORLD.machines = fresh.machines
    WORLD.parts = fresh.parts
    WORLD.work_orders = []
    WORLD.purchase_orders = []
    WORLD._wo_seq = itertools.count(1001)
    WORLD._po_seq = itertools.count(5001)
