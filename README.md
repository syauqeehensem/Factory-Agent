# Equipment Performance Sustaining Chatbot (TCB Factory Agent)

This document is written for both non-coders and developers.

## What this app does (plain language)

This app is a factory support chatbot.
It helps users quickly answer questions like:

- Which tools are down right now?
- Which entity has low yield?
- Should this entity be escalated?

The app reads local project data files and gives structured answers.
It can also keep short conversation memory so follow-up questions make sense.

## Who this is for

- Operations users who need status and escalation answers
- Supervisors who want quick summary checks
- Project team members demonstrating a working AI prototype

## What data it uses

The chatbot reads only local files in the `data/` folder:

- `data/status.csv`
- `data/mtp.csv`
- `data/yield.csv`
- `data/*.pdf`, `data/*.xlsx`, `data/*.txt`, `data/*.md` (manuals and references)

No external factory system updates are performed.
Ticket creation is simulated inside the app memory.

## How the chatbot answers questions

```mermaid
flowchart TD
  start([__start__]) --> check_status[check_status]

  %% Parallel assessment
  check_status --> technician_agent[technician_agent]
  check_status --> quality_agent[quality_agent]

  technician_agent --> read_ticket["read_ticket (MTP)"]
  quality_agent --> yield_check["yield_check (yield data)"]

  read_ticket --> retrieve_cases[retrieve_cases]
  yield_check --> retrieve_cases

  retrieve_cases --> recommend_steps[recommend_steps]
  recommend_steps --> technician_update[technician_update]

  %% Optional iteration
  recommend_steps --> reanalyze[reanalyze]
  reanalyze --> recommend_steps

  %% Technician update either learns/stores or reanalyzes
  technician_update -.-> learn[learn]
  technician_update -.-> reanalyze
  learn --> store[store]
  store --> end_node([__end__])
```

## Quick start for non-coders

1. Open this project folder in VS Code.
2. Open Terminal in VS Code.
3. Run the commands below one by one.

```bash
pip install -r requirements.txt
copy .env.example .env
```

4. Open `.env` and add your API key value for `OPENAI_API_KEY`.
5. Start the app:

```bash
python -m streamlit run chat_ui.py
```

6. Open the shown local URL in your browser (usually `http://localhost:8501`).

## Example questions you can try

- `what tools is down`
- `which entity has the lowest yield`
- `tell me about TCB706`
- `should TCB706 be escalated`
- `what tools are underperforming`

## What has been delivered

- Working Streamlit chatbot prototype
- Company-branded GUI customization
- Local-data-based status, yield, and escalation logic
- Fallback behavior when model is unavailable
- Conversation memory for better follow-up handling
- Public GitHub repository with source code

## Privacy note

`data/` is git-ignored by default and should stay local when it contains confidential production information.

## Developer appendix

### Optional commands

```bash
python cli.py
python run_demo.py --status
python run_demo.py --graph
python -m pytest tests/test_smoke.py -q
python tests/eval_reasoning.py --count 100
python tests/eval_reasoning.py --count 100 --style natural --attempt-live
```

### Reasoning evaluation

Use `tests/eval_reasoning.py` to benchmark reasoning quality over batch prompts.

- Builds up to 100 prompts from known entities plus a small unknown-entity slice.
- Scores route accuracy, escalation accuracy, grounding coverage, latency, and fallback rate.
- Writes report artifacts to `tests/reports/` as JSON and Markdown.

Example:

```bash
python tests/eval_reasoning.py --count 100 --style natural --attempt-live --out-prefix weekly
```

### Important environment variables

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
- `QUESTION_MEMORY_PATH`
- `QUESTION_MEMORY_ENABLED`
- `QUESTION_MEMORY_MAX_ITEMS`
- `KNOWLEDGE_CHUNKS_PATH`
- `KNOWLEDGE_ENABLED`
- `KNOWLEDGE_TOP_K`

### Key files

- `factory_agent/graph.py` - status routing and specialist handoffs
- `factory_agent/agents.py` - specialist prompts
- `factory_agent/tools.py` - integrated read/action tools and entity context
- `factory_agent/knowledge_rag.py` - CSV/manual retrieval and query cache
- `factory_agent/manual_data.py` - manual indexing and search cache
- `factory_agent/pdf_knowledge.py` - PDF knowledge index and citation answers
- `factory_agent/project_data.py` - status and ticket CSV adapters
- `factory_agent/yield_data.py` - per-entity yield checks
- `chat_ui.py` - Streamlit frontend
- `tools/build_pdf_knowledge.py` - build JSONL knowledge chunks from PDF files
