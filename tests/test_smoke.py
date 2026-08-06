"""Smoke tests for the entity-driven Equipment Performance Sustaining pipeline.

These tests avoid live model calls and validate tool/data wiring, the
deterministic status router, and graph structure only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factory_agent import build_graph
from factory_agent.graph import _extract_entity
from factory_agent.project_data import PROJECT_DATA
from factory_agent.tools import (
    create_mtp_ticket,
    get_data_status,
    get_entity_full_context,
    get_entity_status,
    get_entity_ticket_summary,
    get_entity_yield,
    get_line_status_snapshot,
    get_rag_status,
    list_technician_documents,
    list_yield_below_goal,
    search_all_knowledge,
    search_technician_manuals,
)


def test_graph_compiles_with_expected_nodes():
    from langchain_openai import ChatOpenAI

    dummy = ChatOpenAI(model="gpt-4o-mini", api_key="sk-dummy-compile-only")
    graph = build_graph(dummy, prompt_style="base")  # no network — construction only
    nodes = set(graph.get_graph().nodes)
    assert {"status_check", "technician", "yield", "escalation"}.issubset(nodes)

    graph2 = build_graph(dummy, prompt_style="natural")
    nodes2 = set(graph2.get_graph().nodes)
    assert {"status_check", "technician", "yield", "escalation"}.issubset(nodes2)


def test_extract_entity_from_text():
    assert _extract_entity("status of tcb706 please") == "TCB706"
    assert _extract_entity("TSX509") == "TSX509"
    assert _extract_entity("no code here") == ""


def test_status_check_values_up_and_down():
    assert PROJECT_DATA.entity_status_value("TCB706") == "UP"
    assert PROJECT_DATA.entity_status_value("TCB702") == "DOWN"


def test_rag_status_reports_loaded_or_lazy():
    out = get_rag_status.invoke({})
    assert "RAG index" in out


def test_search_all_knowledge_returns_chunks():
    out = search_all_knowledge.invoke(
        {"query": "Optics Table PR Vision Error", "entity": "TSX509", "top_k": 4}
    )
    assert "Top integrated RAG chunks" in out or "RAG index unavailable" in out


def test_search_all_knowledge_contains_csv_sources_for_entity():
    out = search_all_knowledge.invoke(
        {"query": "TSX509 status ticket yield", "entity": "TSX509", "top_k": 6}
    )
    assert "status.csv" in out
    assert "yield.csv" in out


def test_data_status_reports_sources():
    out = get_data_status.invoke({})
    assert "Data:" in out
    assert "Yield dataset" in out
    assert "Technician manual" in out
    assert "RAG index" in out


def test_line_status_snapshot_returns_summary_text():
    out = get_line_status_snapshot.invoke({"max_down": 5})
    assert "Line status" in out or "unavailable" in out


def test_entity_status_for_known_entity():
    out = get_entity_status.invoke({"entity": "TCB706"})
    assert "Entity TCB706" in out


def test_entity_ticket_summary_for_known_entity():
    out = get_entity_ticket_summary.invoke({"entity": "TCB702", "limit": 3})
    assert "Tickets for TCB702" in out or "No ticket" in out


def test_entity_full_context_includes_all_sections():
    out = get_entity_full_context.invoke({"entity": "TCB702", "manual_top_k": 1})
    assert "Integrated context for TCB702" in out
    assert "Status:" in out
    assert "Tickets:" in out
    assert "Yield:" in out
    assert "RAG evidence" in out


def test_entity_yield_below_goal_is_fail():
    out = get_entity_yield.invoke({"entity": "TCB706"})
    assert "FAIL" in out
    assert "34.2%" in out


def test_entity_yield_at_goal_is_pass():
    out = get_entity_yield.invoke({"entity": "TCB003"})
    assert "PASS" in out


def test_list_yield_below_goal_flags_low_tool():
    out = list_yield_below_goal.invoke({"limit": 10})
    assert "TCB706" in out


def test_list_technician_documents_returns_text():
    out = list_technician_documents.invoke({})
    assert out


def test_search_technician_manuals_returns_text():
    out = search_technician_manuals.invoke(
        {"question": "Optics Table PR Vision Error", "top_k": 2}
    )
    assert out


def test_create_mtp_ticket_returns_id():
    out = create_mtp_ticket.invoke({"entity": "TCB706", "reason": "yield below goal"})
    assert "MTP-" in out
    assert "TCB706" in out


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
