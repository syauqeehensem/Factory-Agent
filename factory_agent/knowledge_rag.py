"""Unified RAG-style retrieval across CSVs and manuals.

This index combines chunks from:
- status.csv (entity,status)
- mtp.csv (entity,ticket,error)
- yield.csv (entity,yield)
- technician manuals (PDF/XLSX/TXT/MD chunks from ManualIndex)

The retriever is deterministic and dependency-free: keyword overlap with
entity-aware boosts. This keeps it stable offline while still providing a
single evidence surface for all local data sources.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .config import settings
from .manual_data import MANUAL_INDEX
from .project_data import PROJECT_DATA
from .yield_data import YIELD_DATASET

_ENTITY_RE = re.compile(r"[A-Za-z]{2,4}\d{2,4}")


@dataclass(frozen=True)
class RagChunk:
    source: str
    kind: str
    text: str
    entity: str = ""


class KnowledgeRAG:
    """In-memory unified chunk index for CSV + manual evidence."""

    def __init__(self) -> None:
        self.chunks: list[RagChunk] = []
        self.load_error: str | None = None
        self._loaded = False
        self._version = 0
        self.query_cache_size = max(0, int(settings.rag_query_cache_size))
        self._query_cache: dict[tuple[int, str, str, int], str] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def version(self) -> int:
        return self._version

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.reload()

    def reload(self) -> None:
        self._version += 1
        self._loaded = True
        self.chunks = []
        self.load_error = None
        self._query_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

        self._load_csv_chunks()
        self._load_manual_chunks()

        if not self.chunks:
            self.load_error = "No RAG chunks available from CSV/manual sources."

    def status_text(self) -> str:
        if not self._loaded:
            return "RAG index not loaded yet. It will load on first RAG search."
        if self.load_error:
            return f"RAG index unavailable: {self.load_error}"

        counts = Counter(c.kind for c in self.chunks)
        return (
            "RAG index ready: "
            f"status={counts.get('status', 0)}, "
            f"tickets={counts.get('ticket', 0)}, "
            f"yield={counts.get('yield', 0)}, "
            f"manual={counts.get('manual', 0)} chunk(s). "
            f"Query cache: {len(self._query_cache)} item(s), "
            f"hits={self._cache_hits}, misses={self._cache_misses}."
        )

    def search(self, query: str, entity: str = "", top_k: int = 6) -> str:
        self.ensure_loaded()
        if self.load_error:
            return f"RAG index unavailable: {self.load_error}"
        if not self.chunks:
            return "RAG index is empty."

        key = entity.strip().upper()
        k = max(1, min(int(top_k), 12))
        normalized_query = self._normalize_query(query)
        cache_key = (self._version, normalized_query, key, k)
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1

        terms = self._terms(f"{query} {key}".strip())

        selected: list[tuple[int, RagChunk]] = []
        selected_keys: set[str] = set()

        if key:
            self._add_entity_csv_context(selected, selected_keys, key)
            self._add_ticket_driven_manual(selected, selected_keys, key)

        scored: list[tuple[int, RagChunk]] = []
        for chunk in self.chunks:
            ckey = self._chunk_key(chunk)
            if ckey in selected_keys:
                continue
            score = self._score_chunk(chunk, terms, key)
            if score > 0:
                scored.append((score, chunk))

        selected.sort(key=lambda x: x[0], reverse=True)
        scored.sort(key=lambda x: x[0], reverse=True)
        results = (selected + scored)[:k]

        if not results:
            result = (
                "No strong RAG matches found. "
                "Try including an entity code, error phrase, or ticket id."
            )
            self._remember_result(cache_key, result)
            return result

        rows = ["Top integrated RAG chunks:"]
        for rank, (score, chunk) in enumerate(results, start=1):
            rows.append(f"{rank}. [{chunk.source}] ({chunk.kind}) score={score}")
            rows.append(f"   {self._trim(chunk.text, 260)}")
        result = "\n".join(rows)
        self._remember_result(cache_key, result)
        return result

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join((query or "").strip().lower().split())

    def _remember_result(self, key: tuple[int, str, str, int], value: str) -> None:
        if self.query_cache_size <= 0:
            return
        if key in self._query_cache:
            self._query_cache.pop(key, None)
        elif len(self._query_cache) >= self.query_cache_size:
            oldest = next(iter(self._query_cache), None)
            if oldest is not None:
                self._query_cache.pop(oldest, None)
        self._query_cache[key] = value

    def _load_csv_chunks(self) -> None:
        for row in PROJECT_DATA.status_rows:
            self.chunks.append(
                RagChunk(
                    source=f"status.csv#{row.entity}",
                    kind="status",
                    entity=row.entity,
                    text=f"status.csv row: entity {row.entity} is {row.status}.",
                )
            )

        for row in PROJECT_DATA.ticket_rows:
            error = row.error or "Unknown"
            self.chunks.append(
                RagChunk(
                    source=f"mtp.csv#{row.ticket or row.entity}",
                    kind="ticket",
                    entity=row.entity,
                    text=(
                        "mtp.csv row: "
                        f"entity {row.entity}, ticket {row.ticket or 'n/a'}, error {error}."
                    ),
                )
            )

        for row in YIELD_DATASET.records:
            self.chunks.append(
                RagChunk(
                    source=f"yield.csv#{row.entity}",
                    kind="yield",
                    entity=row.entity,
                    text=f"yield.csv row: entity {row.entity}, yield {row.yield_value:.1f}%.",
                )
            )

    def _load_manual_chunks(self) -> None:
        MANUAL_INDEX.ensure_loaded()
        if MANUAL_INDEX.load_error:
            return

        limit = max(0, settings.rag_manual_max_chunks)
        manual_chunks = MANUAL_INDEX.chunks[:limit] if limit else MANUAL_INDEX.chunks
        for chunk in manual_chunks:
            entity = self._extract_entity(chunk.text)
            source_name = chunk.source.split("#", 1)[0]
            self.chunks.append(
                RagChunk(
                    source=f"manual:{chunk.source}",
                    kind="manual",
                    entity=entity,
                    text=f"{source_name}: {chunk.text}",
                )
            )

    def _add_entity_csv_context(
        self,
        selected: list[tuple[int, RagChunk]],
        selected_keys: set[str],
        entity: str,
    ) -> None:
        # Ensure at least one row from each CSV contributes when available.
        boosts = {"status": 200, "ticket": 190, "yield": 180}
        for kind in ("status", "ticket", "yield"):
            matches = [c for c in self.chunks if c.kind == kind and c.entity == entity]
            if not matches:
                continue
            chunk = matches[0]
            ckey = self._chunk_key(chunk)
            if ckey in selected_keys:
                continue
            selected.append((boosts[kind], chunk))
            selected_keys.add(ckey)

    def _add_ticket_driven_manual(
        self,
        selected: list[tuple[int, RagChunk]],
        selected_keys: set[str],
        entity: str,
    ) -> None:
        errors = [
            (r.error or "").strip()
            for r in PROJECT_DATA.ticket_rows
            if r.entity == entity and (r.error or "").strip() and (r.error or "").strip().lower() != "unknown"
        ]
        if not errors:
            return

        terms = self._terms(" ".join(errors))
        best_score = 0
        best_chunk: RagChunk | None = None
        for chunk in self.chunks:
            if chunk.kind != "manual":
                continue
            score = self._score_chunk(chunk, terms, entity)
            if score > best_score:
                best_score = score
                best_chunk = chunk

        if best_chunk is None or best_score <= 0:
            return

        ckey = self._chunk_key(best_chunk)
        if ckey in selected_keys:
            return
        selected.append((170 + best_score, best_chunk))
        selected_keys.add(ckey)

    @staticmethod
    def _chunk_key(chunk: RagChunk) -> str:
        return f"{chunk.source}|{chunk.kind}|{chunk.entity}|{hash(chunk.text)}"

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {t for t in re.findall(r"[A-Za-z0-9_\-/\.]{2,}", text.lower())}

    @classmethod
    def _score_chunk(cls, chunk: RagChunk, terms: set[str], entity: str) -> int:
        hay = cls._terms(chunk.text)
        overlap = sum(1 for term in terms if term in hay)

        score = overlap
        if entity and chunk.entity == entity:
            score += 6
        if entity and entity.lower() in chunk.text.lower():
            score += 3
        if chunk.kind in {"status", "ticket", "yield"}:
            score += 2
        elif chunk.kind == "manual":
            score += 1
        return score

    @staticmethod
    def _extract_entity(text: str) -> str:
        match = _ENTITY_RE.search(text or "")
        return match.group(0).upper() if match else ""

    @staticmethod
    def _trim(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        clipped = text[: max_chars - 3].rstrip()
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        return f"{clipped}..."


KNOWLEDGE_RAG = KnowledgeRAG()
