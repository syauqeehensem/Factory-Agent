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
PROJECT_DATA_ROOT = PROJECT_ROOT / "Project Data"


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
    project_data_dir: str = field(
        default_factory=lambda: os.getenv("PROJECT_DATA_DIR", str(PROJECT_DATA_ROOT))
    )
    status_csv_path: str = field(
        default_factory=lambda: os.getenv(
            "STATUS_CSV_PATH", str(PROJECT_DATA_ROOT / "status.csv")
        )
    )
    mtp_csv_path: str = field(
        default_factory=lambda: os.getenv("MTP_CSV_PATH", str(PROJECT_DATA_ROOT / "mtp.csv"))
    )
    technician_docs_dir: str = field(
        default_factory=lambda: os.getenv(
            "TECHNICIAN_DOCS_DIR", str(PROJECT_DATA_ROOT / "Technician")
        )
    )
    # Optional path to yield data CSV used by the analytics tools.
    yield_csv_path: str = field(
        default_factory=lambda: os.getenv(
            "YIELD_CSV_PATH",
            str(PROJECT_DATA_ROOT / "Yield" / "Yield data by tools.csv"),
        )
    )
    manual_top_k: int = field(default_factory=lambda: int(os.getenv("MANUAL_TOP_K", "3")))

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
