"""Technician manual indexing for Project Data-only retrieval.

Builds a lightweight local index from files under Project Data/Technician and
supports keyword-based snippet retrieval for the Technician agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import settings


@dataclass(frozen=True)
class ManualChunk:
    source: str
    text: str


class ManualIndex:
    """Small in-memory chunk index with keyword-overlap ranking."""

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.chunks: list[ManualChunk] = []
        self.load_error: str | None = None
        self._loaded = False

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.reload()

    def reload(self) -> None:
        self._loaded = True
        self.chunks = []
        self.load_error = None

        if not self.root_dir.exists():
            self.load_error = f"folder not found at {self.root_dir}"
            return

        # Only index technician docs; the data CSVs (status/mtp/yield) are not manuals.
        files = sorted(
            p
            for p in self.root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".pdf", ".xlsx", ".txt", ".md"}
        )
        if not files:
            self.load_error = f"no supported files found under {self.root_dir}"
            return

        for path in files:
            text = self._read_file(path)
            if not text:
                continue
            for idx, chunk in enumerate(self._chunk_text(text), start=1):
                source = f"{path.name}#chunk{idx}"
                self.chunks.append(ManualChunk(source=source, text=chunk))

        if not self.chunks:
            self.load_error = (
                "supported files were found but no readable content was extracted; "
                "install optional parsers pypdf/openpyxl for PDF/XLSX support"
            )

    @property
    def is_ready(self) -> bool:
        return self.load_error is None and bool(self.chunks)

    def status_text(self) -> str:
        self.ensure_loaded()
        if self.load_error:
            return f"Technician manual index unavailable: {self.load_error}"
        files = {c.source.split("#", 1)[0] for c in self.chunks}
        return (
            f"Technician manual index ready: {len(files)} file(s), "
            f"{len(self.chunks)} chunks."
        )

    def search(self, query: str, top_k: int = 3) -> str:
        self.ensure_loaded()
        if self.load_error:
            return f"Technician manual index unavailable: {self.load_error}"
        if not self.chunks:
            return "Technician manual index is empty."

        terms = self._terms(query)
        if not terms:
            return "Please include a clearer manual question (error code, tool id, or symptom)."

        scored: list[tuple[int, ManualChunk]] = []
        for chunk in self.chunks:
            score = self._score_terms(chunk.text, terms)
            if score > 0:
                scored.append((score, chunk))

        if not scored:
            return (
                "I could not find a close manual match for that question. "
                "Try adding the exact symptom, station, or error keyword."
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        rows = ["Top technician manual snippets:"]
        for rank, (score, chunk) in enumerate(scored[: max(1, top_k)], start=1):
            rows.append(
                f"{rank}. [{chunk.source}] score={score}\n"
                f"   {self._trim(chunk.text, 260)}"
            )
        return "\n".join(rows)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {t for t in re.findall(r"[A-Za-z0-9_\-/\.]{2,}", text.lower())}

    @classmethod
    def _score_terms(cls, text: str, terms: set[str]) -> int:
        hay = cls._terms(text)
        return sum(2 if term in hay and any(ch.isdigit() for ch in term) else 1 for term in terms if term in hay)

    @staticmethod
    def _chunk_text(text: str, size: int = 1200, overlap: int = 120) -> list[str]:
        normalized = " ".join(text.split())
        if not normalized:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(len(normalized), start + size)
            snippet = normalized[start:end].strip()
            if snippet:
                chunks.append(snippet)
            if end >= len(normalized):
                break
            start = max(0, end - overlap)
        return chunks

    @staticmethod
    def _trim(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        clipped = text[: max_chars - 3].rstrip()
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        return f"{clipped}..."

    def _read_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".csv"}:
            return self._read_text(path)
        if suffix == ".pdf":
            return self._read_pdf(path)
        if suffix == ".xlsx":
            return self._read_xlsx(path)
        return ""

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader
        except Exception:
            return ""
        try:
            reader = PdfReader(str(path))
            out: list[str] = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                if txt:
                    out.append(txt)
            return "\n".join(out)
        except Exception:
            return ""

    @staticmethod
    def _read_xlsx(path: Path) -> str:
        try:
            from openpyxl import load_workbook
        except Exception:
            return ""

        try:
            wb = load_workbook(filename=str(path), read_only=True, data_only=True)
            out: list[str] = []
            for ws in wb.worksheets:
                out.append(f"Sheet: {ws.title}")
                for row in ws.iter_rows(values_only=True):
                    vals = [str(v).strip() for v in row if v is not None and str(v).strip()]
                    if vals:
                        out.append(" | ".join(vals))
            wb.close()
            return "\n".join(out)
        except Exception:
            return ""


MANUAL_INDEX = ManualIndex(settings.technician_docs_dir)
