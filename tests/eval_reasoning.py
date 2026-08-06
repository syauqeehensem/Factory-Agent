"""Reasoning-quality benchmark for the TCB chatbot.

Runs batch prompts against deterministic local logic (and optionally live graph)
and writes scored reports to ``tests/reports/``.

Usage examples:

    python tests/eval_reasoning.py --count 100
    python tests/eval_reasoning.py --count 100 --style natural --attempt-live
    python tests/eval_reasoning.py --count 40 --seed 7 --out-prefix week1

Scored dimensions:
- route accuracy (technician vs yield)
- escalation correctness (ticket/no-ticket expectation)
- grounding coverage (status/ticket/yield/rag evidence present)
- latency / fallback rate
"""

from __future__ import annotations

import argparse
import math
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factory_agent import build_graph
from factory_agent.config import settings
from factory_agent.graph import _extract_entity
from factory_agent.llm import LLMNotConfigured, build_chat_model
from factory_agent.project_data import PROJECT_DATA
from factory_agent.tools import get_entity_full_context
from factory_agent.yield_data import YIELD_DATASET

ENTITY_PROMPT_TEMPLATES = [
    "{entity}",
    "status for {entity}",
    "please check {entity}",
    "can you analyze {entity} and decide action",
    "{entity} needs support, what should we do",
    "review {entity} and give recommendation",
    "for {entity}, should we escalate",
]

NOT_FOUND_PROMPTS = [
    "TCB999",
    "status for ZZZ000",
    "please check AAA111",
    "review TTT123",
]


@dataclass(frozen=True)
class EvalCase:
    prompt: str
    entity: str
    expected_route: str
    expected_escalation: bool
    expected_status: str
    expected_known: bool


@dataclass
class EvalResult:
    idx: int
    prompt: str
    entity: str
    expected_route: str
    expected_escalation: bool
    expected_known: bool
    actual_route: str
    actual_escalation: bool
    route_ok: bool
    escalation_ok: bool
    grounded_status: bool
    grounded_tickets: bool
    grounded_yield: bool
    grounded_rag: bool
    grounded_score: float
    latency_ms: int
    fallback_used: bool
    mode: str
    notes: str


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def _entity_meta(entity: str) -> tuple[bool, str, bool, str]:
    status = PROJECT_DATA.entity_status_value(entity)
    if status is None:
        return False, "FINISH", False, "UNKNOWN"

    route = "yield" if status == "UP" else "technician"
    if route == "technician":
        return True, route, True, status

    # UP route escalates when yield is below goal.
    y = YIELD_DATASET.entity_yield(entity)
    needs_escalation = (y is not None) and (y < settings.yield_threshold)
    return True, route, bool(needs_escalation), status


def _build_cases(count: int, seed: int, include_unknown_ratio: float) -> list[EvalCase]:
    rng = random.Random(seed)
    known_entities = sorted({row.entity for row in PROJECT_DATA.status_rows})
    if not known_entities:
        raise RuntimeError("No known entities found in status.csv; cannot build eval set.")

    cases: list[EvalCase] = []
    unknown_count = int(round(count * max(0.0, min(include_unknown_ratio, 0.4))))
    known_count = max(0, count - unknown_count)

    for _ in range(known_count):
        entity = rng.choice(known_entities)
        known, route, escalate, status = _entity_meta(entity)
        template = rng.choice(ENTITY_PROMPT_TEMPLATES)
        prompt = template.format(entity=entity)
        cases.append(
            EvalCase(
                prompt=prompt,
                entity=entity,
                expected_route=route,
                expected_escalation=escalate,
                expected_status=status,
                expected_known=known,
            )
        )

    for _ in range(unknown_count):
        prompt = rng.choice(NOT_FOUND_PROMPTS)
        entity = _extract_entity(prompt) or "UNKNOWN"
        cases.append(
            EvalCase(
                prompt=prompt,
                entity=entity,
                expected_route="FINISH",
                expected_escalation=False,
                expected_status="UNKNOWN",
                expected_known=False,
            )
        )

    rng.shuffle(cases)
    return cases[:count]


def _deterministic_answer(entity: str, reason: str = "eval") -> str:
    context = get_entity_full_context.invoke({"entity": entity, "manual_top_k": 2})
    return (
        f"Mode: deterministic local-data fallback ({reason})\n\n"
        f"{context}"
    )


def _guess_route_from_text(text: str, known: bool) -> str:
    lowered = (text or "").lower()
    if not known:
        if "not found" in lowered or "couldn't find" in lowered or "known entities" in lowered:
            return "FINISH"
    if "mode: deterministic" in lowered:
        # Deterministic answers include integrated context; infer from entity status phrase.
        if "currently up" in lowered:
            return "yield"
        if "currently down" in lowered:
            return "technician"
    if "agent technician" in lowered or "down" in lowered:
        return "technician"
    if "agent yield" in lowered or "continue sustaining" in lowered:
        return "yield"
    return "unknown"


def _guess_escalation_from_text(text: str) -> bool:
    lowered = (text or "").lower()
    positive_markers = [
        "ticket is required",
        "require escalation",
        "create",
        "mtp-",
        "escalation: action",
        "below goal",
    ]
    negative_markers = [
        "no escalation",
        "no action needed",
        "continue sustaining",
        "ticket required no",
    ]
    if any(marker in lowered for marker in negative_markers):
        return False
    return any(marker in lowered for marker in positive_markers)


def _grounding_flags(text: str) -> tuple[bool, bool, bool, bool]:
    lowered = (text or "").lower()
    has_status = "status:" in lowered or "currently up" in lowered or "currently down" in lowered
    has_tickets = "tickets:" in lowered or "ticket" in lowered
    has_yield = "yield:" in lowered or "yield " in lowered
    has_rag = "rag evidence" in lowered or "status.csv" in lowered or "mtp.csv" in lowered
    return has_status, has_tickets, has_yield, has_rag


def _run_live_case(
    graph,
    prompt: str,
    style: str,
    timeout_seconds: float,
    idx: int,
) -> tuple[str, bool, int, str]:
    started = time.perf_counter()
    thread_id = f"eval-{idx}"
    inputs = {"messages": [HumanMessage(content=prompt)], "next": ""}
    run_config = {
        "recursion_limit": settings.recursion_limit,
        "configurable": {"thread_id": thread_id},
    }
    steps: list[str] = []

    try:
        iterator = graph.stream(inputs, config=run_config)
        for step in iterator:
            if (time.perf_counter() - started) > timeout_seconds:
                return "", True, int((time.perf_counter() - started) * 1000), "timeout"
            for _node, update in step.items():
                msgs = update.get("messages") if isinstance(update, dict) else None
                if msgs and msgs[-1].content.strip():
                    steps.append(msgs[-1].content.strip())
    except Exception as exc:  # noqa: BLE001
        return "", True, int((time.perf_counter() - started) * 1000), f"error: {exc}"

    latency = int((time.perf_counter() - started) * 1000)
    reply = "\n\n".join(steps).strip()
    if not reply:
        reply = "I didn't produce a response for that."
    return reply, False, latency, "ok"


def _evaluate_case(
    idx: int,
    case: EvalCase,
    style: str,
    graph,
    attempt_live: bool,
    timeout_seconds: float,
) -> EvalResult:
    notes = "deterministic"
    if attempt_live and graph is not None and style == "natural":
        reply, fallback, latency_ms, state = _run_live_case(
            graph=graph,
            prompt=case.prompt,
            style=style,
            timeout_seconds=timeout_seconds,
            idx=idx,
        )
        if fallback or not reply:
            reply = _deterministic_answer(case.entity, reason="eval-fallback")
            notes = state
        else:
            notes = "live"
    else:
        started = time.perf_counter()
        reply = _deterministic_answer(case.entity, reason="eval")
        latency_ms = int((time.perf_counter() - started) * 1000)
        fallback = True

    actual_route = _guess_route_from_text(reply, case.expected_known)
    actual_escalation = _guess_escalation_from_text(reply)
    route_ok = actual_route == case.expected_route
    escalation_ok = actual_escalation == case.expected_escalation

    g_status, g_tickets, g_yield, g_rag = _grounding_flags(reply)
    grounding_score = sum([g_status, g_tickets, g_yield, g_rag]) / 4.0

    return EvalResult(
        idx=idx,
        prompt=case.prompt,
        entity=case.entity,
        expected_route=case.expected_route,
        expected_escalation=case.expected_escalation,
        expected_known=case.expected_known,
        actual_route=actual_route,
        actual_escalation=actual_escalation,
        route_ok=route_ok,
        escalation_ok=escalation_ok,
        grounded_status=g_status,
        grounded_tickets=g_tickets,
        grounded_yield=g_yield,
        grounded_rag=g_rag,
        grounded_score=grounding_score,
        latency_ms=latency_ms,
        fallback_used=fallback,
        mode=style,
        notes=notes,
    )


def _score(results: list[EvalResult]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    route_hits = sum(1 for r in results if r.route_ok)
    escalation_hits = sum(1 for r in results if r.escalation_ok)
    grounded_avg = mean(r.grounded_score for r in results)
    latency_avg = mean(r.latency_ms for r in results)
    latencies = sorted(r.latency_ms for r in results)
    p95_index = min(total - 1, max(0, math.ceil(0.95 * total) - 1))
    fallback_rate = sum(1 for r in results if r.fallback_used) / total

    return {
        "count": total,
        "route_accuracy": round(route_hits / total, 4),
        "escalation_accuracy": round(escalation_hits / total, 4),
        "grounding_score_avg": round(grounded_avg, 4),
        "latency_ms_avg": round(latency_avg, 1),
        "latency_ms_p95": latencies[p95_index],
        "fallback_rate": round(fallback_rate, 4),
    }


def _top_failures(results: list[EvalResult], max_items: int = 10) -> list[dict[str, Any]]:
    failures = [r for r in results if (not r.route_ok) or (not r.escalation_ok) or (r.grounded_score < 0.75)]
    failures.sort(
        key=lambda r: (
            r.route_ok,
            r.escalation_ok,
            r.grounded_score,
            -r.latency_ms,
        )
    )
    rows: list[dict[str, Any]] = []
    for r in failures[:max_items]:
        rows.append(
            {
                "idx": r.idx,
                "prompt": r.prompt,
                "entity": r.entity,
                "expected_route": r.expected_route,
                "actual_route": r.actual_route,
                "expected_escalation": r.expected_escalation,
                "actual_escalation": r.actual_escalation,
                "grounding_score": round(r.grounded_score, 3),
                "latency_ms": r.latency_ms,
                "fallback_used": r.fallback_used,
                "notes": r.notes,
            }
        )
    return rows


def _write_reports(
    out_prefix: str,
    meta: dict[str, Any],
    score: dict[str, Any],
    results: list[EvalResult],
    failures: list[dict[str, Any]],
) -> tuple[Path, Path]:
    out_dir = ROOT / "tests" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{out_prefix}_{stamp}"
    json_path = out_dir / f"{base}.json"
    md_path = out_dir / f"{base}.md"

    payload = {
        "meta": meta,
        "score": score,
        "failures": failures,
        "results": [r.__dict__ for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# TCB Chatbot Reasoning Evaluation")
    lines.append("")
    lines.append(f"- timestamp: {meta['timestamp']}")
    lines.append(f"- style: {meta['style']}")
    lines.append(f"- attempt_live: {meta['attempt_live']}")
    lines.append(f"- count: {meta['count']}")
    lines.append(f"- seed: {meta['seed']}")
    lines.append(f"- timeout_seconds: {meta['timeout_seconds']}")
    lines.append("")
    lines.append("## Score")
    lines.append("")
    for key, value in score.items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("## Top Failures")
    lines.append("")
    if not failures:
        lines.append("No major failures found.")
    else:
        lines.append("| idx | entity | prompt | route (exp->act) | esc (exp->act) | grounding | latency_ms | fallback | notes |")
        lines.append("|---|---|---|---|---|---:|---:|---|---|")
        for row in failures:
            prompt = str(row["prompt"]).replace("|", "\\|")
            lines.append(
                "| {idx} | {entity} | {prompt} | {er}->{ar} | {ee}->{ae} | {gs} | {lat} | {fb} | {notes} |".format(
                    idx=row["idx"],
                    entity=row["entity"],
                    prompt=prompt,
                    er=row["expected_route"],
                    ar=row["actual_route"],
                    ee=row["expected_escalation"],
                    ae=row["actual_escalation"],
                    gs=row["grounding_score"],
                    lat=row["latency_ms"],
                    fb=row["fallback_used"],
                    notes=row["notes"],
                )
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate reasoning quality over batch prompts.")
    parser.add_argument("--count", type=int, default=100, help="Number of prompts to run.")
    parser.add_argument("--style", choices=["base", "natural"], default="natural")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=max(6.0, settings.ui_soft_timeout_seconds))
    parser.add_argument("--unknown-ratio", type=float, default=0.08, help="Fraction of unknown-entity prompts.")
    parser.add_argument("--attempt-live", action="store_true", help="Attempt live model graph calls in natural mode.")
    parser.add_argument("--out-prefix", default="reasoning_eval")
    args = parser.parse_args()

    PROJECT_DATA.reload()
    YIELD_DATASET.reload()

    graph = None
    if args.attempt_live and args.style == "natural":
        try:
            graph = build_graph(build_chat_model(), checkpointer=MemorySaver(), prompt_style=args.style)
        except (LLMNotConfigured, Exception) as exc:  # noqa: BLE001
            _safe_print(f"[warn] Live graph unavailable; falling back to deterministic eval only: {exc}")
            graph = None

    cases = _build_cases(
        count=max(1, args.count),
        seed=args.seed,
        include_unknown_ratio=args.unknown_ratio,
    )

    results: list[EvalResult] = []
    started = time.perf_counter()
    for idx, case in enumerate(cases, start=1):
        result = _evaluate_case(
            idx=idx,
            case=case,
            style=args.style,
            graph=graph,
            attempt_live=bool(args.attempt_live),
            timeout_seconds=max(1.0, args.timeout_seconds),
        )
        results.append(result)
        if idx % 10 == 0 or idx == len(cases):
            _safe_print(f"progress: {idx}/{len(cases)}")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    score = _score(results)
    failures = _top_failures(results, max_items=12)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "style": args.style,
        "attempt_live": bool(args.attempt_live and graph is not None),
        "count": len(results),
        "seed": args.seed,
        "timeout_seconds": float(args.timeout_seconds),
        "elapsed_ms": elapsed_ms,
    }

    json_path, md_path = _write_reports(
        out_prefix=args.out_prefix,
        meta=meta,
        score=score,
        results=results,
        failures=failures,
    )

    _safe_print("\n=== SCORE ===")
    for key, value in score.items():
        _safe_print(f"{key}: {value}")
    _safe_print(f"\nReport JSON: {json_path}")
    _safe_print(f"Report MD  : {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
