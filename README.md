# 🤖 Factory Agent / TCB Chatbot — Multi-Agent Production Orchestration Kickstart

A small, readable **LangGraph** project where specialized agents collaborate to
automate a factory workflow — predictive-maintenance triage — through a *stateful,
cyclic* graph with **secure, audited tool-calling**.

Four agents in this workspace (one supervisor + three specialists):

- **🧑‍✈️ Floor Supervisor** — the orchestrator. Decides which specialist acts next, and when the job is done.
- **🔧 Maintenance Scheduler** — triages sensor alerts, raises and schedules work orders.
- **📈 Yield Specialist** — analyzes lot/tool yield history and flags systematic patterns.
- **📦 Parts Procurement Agent** — checks inventory and orders parts (within a spend guardrail).

It is built to be **both a learning tool and a project seed**: the whole thing is a
few hundred commented lines, and every part is an obvious place to extend.

```
            ┌───────────────────────────────────────────────┐
            ▼                                               │ (cycle)
  START → Floor Supervisor ──route──► Maintenance Scheduler ─┤
              │     ▲       │          (read sensors,        │
              │     │       │           create/schedule WOs) │
              │     │       ├────────► Yield Specialist ─────┤
              │     │       │         (analyze yield trends) │
              │     │       └────────► Parts Procurement ────┘
              │     │                  (check stock, order parts)
              │  "FINISH"
              ▼
             END
```

Run `python run_demo.py --graph` to print this graph as Mermaid you can paste into
[mermaid.live](https://mermaid.live).

---

## What you'll learn (the LangGraph mental model)

| Concept | Where it lives | One-line idea |
|--------|----------------|---------------|
| **State + reducers** | [`state.py`](factory_agent/state.py) | A typed dict that flows through nodes; `messages` uses the `add_messages` reducer to *append*. |
| **Nodes** | [`graph.py`](factory_agent/graph.py) | Plain functions: `state -> partial state update`. |
| **Conditional edges** | [`graph.py`](factory_agent/graph.py) | The supervisor's typed decision picks the next node. |
| **Cycles** | [`graph.py`](factory_agent/graph.py) | Workers route back to the supervisor — a loop, not a DAG. |
| **Tool-calling agents** | [`agents.py`](factory_agent/agents.py) | `create_react_agent` runs the reason→act→observe loop. |
| **Secure tools** | [`tools.py`](factory_agent/tools.py) + [`security.py`](factory_agent/security.py) | Read vs. action tools, an audit trail, and a spend guardrail. |
| **Recursion limit** | [`config.py`](factory_agent/config.py) | The safety net that stops a runaway cycle. |

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env                 # Windows: copy .env.example .env
# edit .env and paste your key:  OPENAI_API_KEY=sk-...   (reuse your previous key)

python run_demo.py                   # run the predictive-maintenance scenario
```

You'll watch the Floor Supervisor delegate, each agent act, and then a closing
report of work orders, purchase orders, and the secure audit trail.

Other entry points:

```bash
python run_demo.py --list            # just show the simulated factory floor
python run_demo.py --graph           # print the graph structure (no API key needed)
python run_demo.py --ask "CONV-01 is running noisy, please investigate"
python cli.py                        # interactive: hand the team situations one by one
python chat_popup.py                 # desktop popup chatbot (no Streamlit)
streamlit run chat_ui.py             # popup chatbot UI in your browser
python tests/test_smoke.py           # offline tests (tools, guardrails, graph wiring)
```

The Streamlit app now uses TCB branding and renders an Intel Foundry logo
from `assets/foundry-logo.svg` (override via `FOUNDRY_LOGO_PATH` in `.env`).

## Integrating project Yield CSV

`Project Data/` is treated as local-only (PnC/proprietary) and is git-ignored by default.
Keep those files on your machine and do not commit them.

This workspace now includes first-pass integration with the Yield dataset at
`Project Data/Yield/Yield data by tools.csv`.

Configure path (optional) in `.env`:

```bash
YIELD_CSV_PATH=Project Data/Yield/Yield data by tools.csv
```

The Maintenance Scheduler can use these read-only tools during triage:

- `list_machine_yield_mappings` — machine id -> yield entity mapping (if provided)
- `get_yield_dataset_status` — row count, date range, and average yield
- `get_machine_yield_summary(machine_id)` — joined context when mapping exists
- `get_tool_yield_summary(entity)` — summary for a tool code, e.g. `TSX501`
- `get_lot_yield_summary(lot)` — details for one lot id
- `list_yield_hotspots(max_avg_yield, min_lots, limit)` — high-yield entities
- `summarize_recent_yield_vs_baseline(hours=24, top_n=3)` — compares recent window
  vs full timeline baseline and quantifies improvement room

The workflow now includes a dedicated `yield_specialist` agent node, and these
tools are integrated into `MAINTENANCE_TOOLS` and `YIELD_TOOLS` for use from
`cli.py`, `chat_popup.py`, and `chat_ui.py` via normal natural-language prompts.

Conversation memory is enabled in `cli.py`, `chat_popup.py`, and `chat_ui.py`
using LangGraph checkpointer threads. Follow-up questions now retain context
within the current session until you reset.

By default, machines start as `(unmapped)` to avoid false correlations between
machine IDs and yield entity/tool codes. Add mappings only when your domain data
confirms they are equivalent.

## Integrating status.csv and mtp.csv

This workspace now includes deterministic tools for line status and open-ticket
analysis from:

- `Project Data/status.csv`
- `Project Data/mtp.csv`

Configure paths in `.env` if needed:

```bash
PROJECT_DATA_DIR=Project Data
STATUS_CSV_PATH=Project Data/status.csv
MTP_CSV_PATH=Project Data/mtp.csv
```

New tools available to agents:

- `get_project_data_status`
- `get_line_status_snapshot`
- `get_entity_status(entity)`
- `summarize_open_tickets(top_n=5)`
- `get_entity_ticket_summary(entity, limit=6)`
- `refresh_project_data`

> No API key? The **tests and `--graph`/`--list` run offline** — only the live agent
> run needs a key, because the agents use the LLM to decide what to do.

---

## The demo, step by step

The default scenario: *CNC-01 trips a high-vibration alert.* A typical run:

1. **Floor Supervisor → Maintenance Scheduler.** "Diagnose CNC-01 and act."
2. **Maintenance Scheduler** calls `read_sensor` + `get_maintenance_history`,
   concludes the spindle bearing is worn, calls `create_work_order` (high priority)
   and `schedule_maintenance`. Reports the part likely needed (BRG-204).
3. **Floor Supervisor → Parts Procurement.** "Make sure that part is available."
4. **Yield Specialist** checks project yield context (machine/tool mapping + trend)
  to indicate whether the issue appears isolated or systemic.
5. **Parts Procurement** calls `check_parts_inventory` (BRG-204 is out of stock),
  then `order_parts` — a $640 PO, under the auto-approval limit, so it goes through.
6. **Floor Supervisor → FINISH.** Work order scheduled, yield context captured, part on order.

Every action lands in the **audit trail**, e.g.:

```
maintenance_scheduler: create_work_order — WO-1001 CNC-01 [high] ...
maintenance_scheduler: schedule_maintenance — WO-1001 -> Tech A @ tomorrow morning
parts_procurement:     order_parts — PO-5001 2x BRG-204 $640.00 (within $1000 limit)
```

### See the security guardrail fire

Set `AUTO_APPROVE_LIMIT=500` in `.env` and re-run. The bearing order now exceeds the
limit, so the action tool **blocks it and audits it as `<BLOCKED>`**, and the agent
reports that Floor-Supervisor approval is required — instead of silently spending.

---

## Secure tool-calling

Letting an LLM *act* on factory systems needs guardrails, kept in one auditable place:

- **Read-only vs. action tools** — reads (sensors, inventory) are always safe; actions
  (work orders, purchases) change state and are **always logged**. The boundary is
  declared in `security.py` (`READ_ONLY_TOOLS` / `ACTION_TOOLS`).
- **Audit trail** — every action records actor, tool, details, and result.
- **Spend guardrail** — purchases above `AUTO_APPROVE_LIMIT` are blocked pending
  supervisor approval (policy-in-the-loop).
- **Scoped tools** — each agent is bound only to the tools its role needs (the
  Maintenance Scheduler cannot spend money; Procurement cannot edit work orders).

---

## Project structure

```
AIMP_factory_agent/
├── run_demo.py                 # end-to-end scenario (start here)
├── cli.py                      # interactive console
├── requirements.txt
├── .env.example                # copy to .env, add your key
│
├── factory_agent/
│   ├── state.py                # FactoryState (messages + routing) — the graph state
│   ├── graph.py                # ★ supervisor + workers + cyclic edges (the orchestration)
│   ├── agents.py               # the specialist ReAct agents + their prompts
│   ├── tools.py                # secure tools: sensors, work orders, parts
│   ├── security.py             # audit trail + spend guardrail (trust boundary)
│   ├── mock_factory.py         # simulated sensors / inventory / CMMS (swap for real systems)
│   ├── llm.py                  # ChatOpenAI builder
│   └── config.py               # settings from env / .env
│
└── tests/test_smoke.py         # offline checks (no API key needed)
```

---

## Extend it (your capstone starts here)

- **Add an agent** — e.g. a *Quality Inspector* or *Shift Logger*. Write its tools,
  build a ReAct agent in `agents.py`, add a node + edge in `graph.py`, and list it in
  the supervisor prompt. The cyclic pattern scales to more specialists.
- **Add human-in-the-loop approval** — when a purchase is blocked, use LangGraph's
  `interrupt()` to pause and ask a human (or the Floor Supervisor) to approve, then
  resume. This closes the loop the guardrail opens.
- **Add memory / persistence** — pass a `checkpointer` (e.g. `MemorySaver`) to
  `build_graph` so a run can be paused, resumed, and inspected by `thread_id`.
- **Connect real systems** — replace `mock_factory.py` calls with your CMMS/ERP/sensor
  historian APIs. The tools are the only thing that changes.
- **Swap the model** — point `llm.py` at Azure OpenAI or a local OpenAI-compatible
  endpoint; the graph is unchanged.
- **Predictive triage** — feed real sensor trends and have the Maintenance Scheduler
  call a small model/heuristic to predict failure before raising the work order.

> The factory, sensor values, parts, and histories here are **fictional sample data**
> for demonstration. Replace them with your real systems before relying on any action.
