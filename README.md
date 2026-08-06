# TCB Chatbot (Factory Agent)

Entity-first manufacturing support chatbot with deterministic routing and local-data RAG.

## Core flow

- `status_check` extracts an entity code (`TCB706`, `TSX509`, ...)
- DOWN routes to `Agent Technician`
- UP routes to `Agent Yield`
- both route to shared `Escalation`

## Local data scope

The chatbot uses only files under `data/`:

- `data/status.csv`
- `data/mtp.csv`
- `data/yield.csv`
- `data/*.pdf`, `data/*.xlsx`, `data/*.txt`, `data/*.md` for manuals

No external factory system writes are performed. Ticket creation is simulated in memory.

## Smoothness features

- Base style (`/style base`) is deterministic and does not wait on model latency.
- Natural style (`/style natural`) has a soft UI timeout and falls back to local data.
- Runtime profiles (`/profile fast`, `/profile rich`) let you switch between
	speed-first and richer model-first behavior in one command.
- Agent trace visibility can be toggled live (`/trace on|off|toggle`).
- Session response cache speeds repeated prompts.
- RAG and manual search caches reduce repeated retrieval cost.
- Quick action buttons in UI for common flows, profile switching, and trace toggling.
- Runtime commands for status and reload.

## Quick start

```bash
pip install -r requirements.txt
copy .env.example .env
# add OPENAI_API_KEY in .env

python -m streamlit run chat_ui.py
```

## Slash commands (UI)

- `/style base`
- `/style natural`
- `/profile fast`
- `/profile rich`
- `/trace on|off|toggle`
- `/status`
- `/rag`
- `/reload`
- `/help`

## Optional commands

```bash
python cli.py
python run_demo.py --status
python run_demo.py --graph
python -m pytest tests/test_smoke.py -q
python tests/eval_reasoning.py --count 100
python tests/eval_reasoning.py --count 100 --style natural --attempt-live
```

## Reasoning evaluation

Use `tests/eval_reasoning.py` to benchmark reasoning quality over batch prompts.

- Builds up to 100 prompts from known entities plus a small unknown-entity slice.
- Scores route accuracy, escalation accuracy, grounding coverage, latency, and fallback rate.
- Writes report artifacts to `tests/reports/` as JSON + Markdown.

Example:

```bash
python tests/eval_reasoning.py --count 100 --style natural --attempt-live --out-prefix weekly
```

## Important environment variables

- `OPENAI_API_KEY`
- `CHAT_MODEL`
- `LLM_TIMEOUT_SECONDS`
- `UI_SOFT_TIMEOUT_SECONDS`
- `UI_RESPONSE_CACHE_SIZE`
- `RAG_QUERY_CACHE_SIZE`
- `MANUAL_SEARCH_CACHE_SIZE`
- `ENTITY_CONTEXT_CACHE_SIZE`
- `STATUS_CSV_PATH`
- `MTP_CSV_PATH`
- `YIELD_CSV_PATH`
- `TECHNICIAN_DOCS_DIR`

## Privacy note

`data/` is git-ignored by default and should remain local when it contains confidential production information.

## Key files

- `factory_agent/graph.py` - deterministic status router + specialist handoffs
- `factory_agent/agents.py` - specialist prompts (base and natural)
- `factory_agent/tools.py` - integrated read/action tools and entity context
- `factory_agent/knowledge_rag.py` - unified CSV/manual retrieval and query cache
- `factory_agent/manual_data.py` - manual indexing and search cache
- `factory_agent/project_data.py` - status and ticket CSV adapters
- `factory_agent/yield_data.py` - per-entity yield checks
- `chat_ui.py` - Streamlit frontend with soft-timeout fallback and quick actions
