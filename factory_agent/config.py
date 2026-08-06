"""Central configuration.

Values are read from environment variables (optionally via a local ``.env``
file) so you never hard-code secrets. Copy ``.env.example`` to ``.env`` and
fill in your OpenAI key, or reuse the same key as the other AIMP kits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Real data lives under data/ (status.csv, mtp.csv, yield.csv, technician PDFs).
DATA_ROOT = PROJECT_ROOT / "data"


def _load_dotenv() -> None:
    """Load key=value pairs from a ``.env`` file if present (zero-dependency)."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value:
            # A real value in .env wins, even over a stale variable already in
            # the environment (a common "I edited .env but nothing changed" trap).
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


_load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    """Parse common truthy/falsey env values with a safe default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    """Runtime settings, populated from the environment with sane defaults."""

    # --- LLM ---------------------------------------------------------------
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "gpt-4o-mini"))
    llm_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
    )
    llm_max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "0")))
    # Low temperature = predictable orchestration decisions.
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1")))

    # --- Orchestration -----------------------------------------------------
    # Safety net for cyclic graphs: max super-steps before LangGraph stops.
    recursion_limit: int = field(default_factory=lambda: int(os.getenv("RECURSION_LIMIT", "18")))

    # --- Secure tool-calling guardrail ------------------------------------
    # Purchase orders at or below this auto-approve; above needs supervisor sign-off.
    auto_approve_limit: float = field(
        default_factory=lambda: float(os.getenv("AUTO_APPROVE_LIMIT", "1000"))
    )

    # --- Project data ------------------------------------------------------
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", str(DATA_ROOT)))
    status_csv_path: str = field(
        default_factory=lambda: os.getenv("STATUS_CSV_PATH", str(DATA_ROOT / "status.csv"))
    )
    mtp_csv_path: str = field(
        default_factory=lambda: os.getenv("MTP_CSV_PATH", str(DATA_ROOT / "mtp.csv"))
    )
    technician_docs_dir: str = field(
        default_factory=lambda: os.getenv("TECHNICIAN_DOCS_DIR", str(DATA_ROOT))
    )
    yield_csv_path: str = field(
        default_factory=lambda: os.getenv("YIELD_CSV_PATH", str(DATA_ROOT / "yield.csv"))
    )
    manual_top_k: int = field(default_factory=lambda: int(os.getenv("MANUAL_TOP_K", "3")))
    manual_max_pdf_pages: int = field(
        default_factory=lambda: int(os.getenv("MANUAL_MAX_PDF_PAGES", "60"))
    )
    manual_max_chunks_per_file: int = field(
        default_factory=lambda: int(os.getenv("MANUAL_MAX_CHUNKS_PER_FILE", "180"))
    )
    manual_status_lazy: bool = field(
        default_factory=lambda: _env_bool("MANUAL_STATUS_LAZY", True)
    )
    # Yield goal: tools at/above this percent are healthy; below triggers escalation.
    yield_threshold: float = field(
        default_factory=lambda: float(os.getenv("YIELD_THRESHOLD", "50"))
    )

    # --- UI branding -------------------------------------------------------
    app_title: str = field(default_factory=lambda: os.getenv("APP_TITLE", "TCB Chatbot"))
    foundry_logo_path: str = field(
        default_factory=lambda: os.getenv(
            "FOUNDRY_LOGO_PATH", str(PROJECT_ROOT / "assets" / "foundry-logo.svg")
        )
    )

    @property
    def llm_enabled(self) -> bool:
        """True when an OpenAI key is configured (required to RUN the agents)."""
        return bool(self.openai_api_key)


settings = Settings()
