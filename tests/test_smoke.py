"""Smoke tests — verify tools, guardrails, and graph wiring WITHOUT an API key.

The graph compiles offline because constructing ChatOpenAI never hits the network
(only ``.invoke()`` does). We pass a dummy-key model so we can assert the graph is
wired correctly. Running the agents end-to-end needs a real key and is exercised
by ``run_demo.py`` instead.

Run with::

    pytest            # if pytest is installed
    python tests/test_smoke.py   # also runs standalone
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factory_agent import build_graph
from factory_agent.config import settings
from factory_agent.mock_factory import WORLD, reset_world
from factory_agent.security import AUDIT_LOG, reset_audit
from factory_agent.tools import (
    create_work_order,
    get_entity_status,
    get_entity_ticket_summary,
    get_machine_yield_summary,
    get_line_status_snapshot,
    get_project_data_status,
    get_tool_yield_summary,
    get_yield_dataset_status,
    list_machine_yield_mappings,
    list_yield_hotspots,
    order_parts,
    read_sensor,
    schedule_maintenance,
    summarize_recent_yield_vs_baseline,
)


def setup():  # called by the standalone runner before each test
    reset_world()
    reset_audit()


def test_read_sensor_flags_alert():
    setup()
    out = read_sensor.invoke({"machine_id": "CNC-01"})
    assert "ALERT" in out and "7.8" in out


def test_read_sensor_unknown_machine():
    setup()
    out = read_sensor.invoke({"machine_id": "NOPE-99"})
    assert "Unknown machine" in out


def test_create_work_order_writes_and_audits():
    setup()
    out = create_work_order.invoke(
        {"machine_id": "CNC-01", "issue": "spindle bearing wear", "priority": "high"}
    )
    assert "WO-" in out
    assert len(WORLD.work_orders) == 1
    assert WORLD.work_orders[0]["priority"] == "high"
    assert any(e.action == "create_work_order" for e in AUDIT_LOG)


def test_schedule_maintenance_updates_work_order():
    setup()
    create_work_order.invoke({"machine_id": "CNC-01", "issue": "bearing", "priority": "high"})
    wo_id = WORLD.work_orders[0]["id"]
    out = schedule_maintenance.invoke(
        {"work_order_id": wo_id, "technician": "R. Okafor", "when": "today 16:00"}
    )
    assert "Scheduled" in out
    assert WORLD.work_orders[0]["status"] == "scheduled"
    assert WORLD.work_orders[0]["technician"] == "R. Okafor"


def test_order_parts_within_limit_succeeds():
    setup()
    # BRG-204 is $320; 2 units = $640, under the default $1000 limit.
    out = order_parts.invoke({"part_number": "BRG-204", "quantity": 2})
    assert "Ordered" in out and "PO-" in out
    assert len(WORLD.purchase_orders) == 1


def test_order_parts_over_limit_is_blocked():
    setup()
    original = settings.auto_approve_limit
    settings.auto_approve_limit = 500.0  # force the guardrail to trigger
    try:
        out = order_parts.invoke({"part_number": "BRG-204", "quantity": 2})  # $640 > $500
        assert "BLOCKED" in out
        assert len(WORLD.purchase_orders) == 0  # nothing was actually ordered
        assert any(e.status == "blocked" for e in AUDIT_LOG)
    finally:
        settings.auto_approve_limit = original


def test_graph_compiles_with_expected_nodes():
    setup()
    from langchain_openai import ChatOpenAI

    dummy = ChatOpenAI(model="gpt-4o-mini", api_key="sk-dummy-compile-only")
    graph = build_graph(dummy)  # no network — construction only
    nodes = set(graph.get_graph().nodes)
    assert {
        "supervisor", "maintenance_scheduler", "yield_specialist", "parts_procurement"
    }.issubset(nodes)


def test_yield_dataset_status_reports_loaded_or_unavailable():
    setup()
    out = get_yield_dataset_status.invoke({})
    assert "Yield dataset" in out


def test_project_data_status_reports_loaded_or_unavailable():
    setup()
    out = get_project_data_status.invoke({})
    assert "Project Data:" in out


def test_line_status_snapshot_returns_summary_text():
    setup()
    out = get_line_status_snapshot.invoke({"max_down": 5})
    assert "Line status snapshot" in out


def test_entity_status_for_known_entity():
    setup()
    out = get_entity_status.invoke({"entity": "TSX509"})
    assert "Entity TSX509" in out


def test_entity_ticket_summary_for_known_entity():
    setup()
    out = get_entity_ticket_summary.invoke({"entity": "TSX509", "limit": 3})
    assert "Tickets for TSX509" in out


def test_tool_yield_summary_for_known_entity():
    setup()
    out = get_tool_yield_summary.invoke({"entity": "TSX501"})
    assert "TSX501" in out and "lots=" in out


def test_machine_yield_mapping_summary_for_known_machine():
    setup()
    out = get_machine_yield_summary.invoke({"machine_id": "CNC-01"})
    assert "CNC-01" in out and "no yield entity mapping" in out


def test_machine_yield_mapping_list_contains_defaults():
    setup()
    out = list_machine_yield_mappings.invoke({})
    assert "CNC-01 -> (unmapped)" in out and "CNC-02 -> (unmapped)" in out


def test_yield_hotspot_listing_returns_text():
    setup()
    out = list_yield_hotspots.invoke({"max_avg_yield": 0.005, "min_lots": 3, "limit": 5})
    assert out


def test_recent_yield_vs_baseline_returns_text():
    setup()
    out = summarize_recent_yield_vs_baseline.invoke({"hours": 24, "top_n": 3})
    assert out


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"ERROR {name}: {exc!r}")
    print(f"\n{'All tests passed.' if not failures else f'{failures} test(s) failed.'}")
    sys.exit(1 if failures else 0)
