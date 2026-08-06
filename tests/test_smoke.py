"""Smoke tests for the Project-Data-only two-agent pipeline.

These tests avoid live model calls and validate tool/data wiring plus graph
structure only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factory_agent import build_graph
from factory_agent.tools import (
    get_entity_status,
    get_entity_ticket_summary,
    get_line_status_snapshot,
    get_project_data_status,
    get_technician_manual_status,
    get_tool_yield_summary,
    get_yield_dataset_status,
    list_technician_documents,
    list_yield_hotspots,
    search_technician_manuals,
    summarize_recent_yield_vs_baseline,
    summarize_open_tickets,
)


def setup():
    # Keep this hook for standalone execution symmetry.
    return None


def test_graph_compiles_with_expected_nodes():
    setup()
    from langchain_openai import ChatOpenAI

    dummy = ChatOpenAI(model="gpt-4o-mini", api_key="sk-dummy-compile-only")
    graph = build_graph(dummy)  # no network — construction only
    nodes = set(graph.get_graph().nodes)
    assert {"supervisor", "agent_technician", "agent_yield"}.issubset(nodes)


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
    assert "Line status" in out or "unavailable" in out


def test_entity_status_for_known_entity():
    setup()
    out = get_entity_status.invoke({"entity": "TSX509"})
    assert "Entity TSX509" in out


def test_entity_ticket_summary_for_known_entity():
    setup()
    out = get_entity_ticket_summary.invoke({"entity": "TSX509", "limit": 3})
    assert "Tickets for TSX509" in out


def test_open_ticket_summary_returns_text():
    setup()
    out = summarize_open_tickets.invoke({"top_n": 5})
    assert "Ticket" in out


def test_tool_yield_summary_for_known_entity():
    setup()
    out = get_tool_yield_summary.invoke({"entity": "TSX501"})
    assert "TSX501" in out


def test_technician_manual_status_reports_state():
    setup()
    out = get_technician_manual_status.invoke({})
    assert "Technician manual index" in out


def test_list_technician_documents_returns_text():
    setup()
    out = list_technician_documents.invoke({})
    assert out


def test_search_technician_manuals_returns_text():
    setup()
    out = search_technician_manuals.invoke(
        {"question": "vision error on TSX", "top_k": 2}
    )
    assert out


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
