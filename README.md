# TCB Chatbot (Factory Agent)

Project-Data-only multi-agent chatbot for manufacturing support.

This version is intentionally focused on two specialists only:

- Agent Technician: line status, ticket analysis, and technician-manual troubleshooting
- Agent Yield: yield trends, hotspots, lot checks, and baseline comparisons

A supervisor node routes requests between these two agents and finishes when the
question is sufficiently answered.

## Data scope

The pipeline is constrained to files under `Project Data` only:

- `Project Data/status.csv`
- `Project Data/mtp.csv`
- `Project Data/Yield/Yield data by tools.csv`
- `Project Data/Technician/*` (PDF/XLSX/TXT/MD/CSV)

No mock-machine or non-Project-Data sources are used for normal responses.

## Quick start

```bash
pip install -r requirements.txt
copy .env.example .env
# add OPENAI_API_KEY in .env

streamlit run chat_ui.py
```

Minimal UI goals:

- clean chat-only interface
- no settings sidebar
- one "New Chat" button
- natural conversational answers

## Optional commands

```bash
python cli.py
python run_demo.py --status
python run_demo.py --graph
python tests/test_smoke.py
```

## Environment variables

Core values in `.env.example`:

- `OPENAI_API_KEY`
- `CHAT_MODEL`
- `LLM_TIMEOUT_SECONDS`
- `LLM_MAX_RETRIES`
- `RECURSION_LIMIT`
- `PROJECT_DATA_DIR`
- `STATUS_CSV_PATH`
- `MTP_CSV_PATH`
- `YIELD_CSV_PATH`
- `TECHNICIAN_DOCS_DIR`
- `MANUAL_TOP_K`
- `APP_TITLE`
- `FOUNDRY_LOGO_PATH`

## Privacy note

`Project Data/` is git-ignored by default and should stay local if it contains
PnC/proprietary material.

## Key files

- `factory_agent/graph.py` - supervisor + 2-agent routing graph
- `factory_agent/agents.py` - Agent Technician and Agent Yield prompts
- `factory_agent/tools.py` - Project-Data-only tools
- `factory_agent/project_data.py` - status/ticket data access
- `factory_agent/yield_data.py` - yield analytics
- `factory_agent/manual_data.py` - technician manual indexing/search
- `chat_ui.py` - minimal Streamlit chat interface
