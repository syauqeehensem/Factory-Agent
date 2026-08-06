"""Desktop popup chatbot for the Factory Agent workspace (no Streamlit).

Run:
    python chat_popup.py

Behavior:
- If OPENAI_API_KEY is configured, the full multi-agent graph is used.
- If no key is configured, the app falls back to fixed/configured local responses.
"""

from __future__ import annotations

import queue
import re
import threading
import tkinter as tk
from tkinter import ttk
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from factory_agent import build_graph
from factory_agent.config import settings
from factory_agent.llm import LLMNotConfigured, build_chat_model
from factory_agent.manual_data import MANUAL_INDEX
from factory_agent.project_data import PROJECT_DATA
from factory_agent.yield_data import YIELD_DATASET

CHECKPOINTER = MemorySaver()


class FactoryChatPopup:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TCB Chatbot Desktop")
        self.root.geometry("980x680")
        self.root.minsize(760, 520)

        self.graph = None
        self.graph_error = ""
        self.thread_id = str(uuid4())
        self.result_queue: queue.Queue[str] = queue.Queue()
        self.busy = False

        self._build_ui()
        self._reset_session(clear_chat=True)
        self._append_assistant(
            "Ask about line status, tickets, manuals, or yield. Commands: /status /reset /quit"
        )
        self.root.after(120, self._poll_worker_results)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        header = ttk.Label(
            frame,
            text="TCB Chatbot",
            font=("Segoe UI", 15, "bold"),
        )
        header.pack(anchor="w")

        self.status_var = tk.StringVar(value="")
        status = ttk.Label(frame, textvariable=self.status_var)
        status.pack(anchor="w", pady=(2, 8))

        self.chat_box = tk.Text(frame, wrap="word", state="disabled", height=26)
        self.chat_box.pack(fill="both", expand=True)
        self.chat_box.tag_configure("user", foreground="#0f172a")
        self.chat_box.tag_configure("assistant", foreground="#7c2d12")
        self.chat_box.tag_configure("system", foreground="#334155")

        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=(10, 0))

        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(controls, textvariable=self.input_var)
        self.input_entry.pack(side="left", fill="x", expand=True)
        self.input_entry.bind("<Return>", self._on_send)

        self.send_btn = ttk.Button(controls, text="Send", command=self._on_send)
        self.send_btn.pack(side="left", padx=(8, 0))

        action_row = ttk.Frame(frame)
        action_row.pack(fill="x", pady=(8, 0))
        ttk.Button(action_row, text="Data Status", command=self._show_status).pack(side="left")
        ttk.Button(action_row, text="Reset", command=self._reset_clicked).pack(side="left", padx=(6, 0))
        ttk.Button(action_row, text="Quit", command=self.root.destroy).pack(side="right")

        self.input_entry.focus_set()

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _append(self, role: str, text: str, tag: str) -> None:
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"{role}: {text}\n\n", tag)
        self.chat_box.configure(state="disabled")
        self.chat_box.see("end")

    def _append_user(self, text: str) -> None:
        self._append("You", text, "user")

    def _append_assistant(self, text: str) -> None:
        self._append("TCB Chatbot", text, "assistant")

    def _append_system(self, text: str) -> None:
        self._append("System", text, "system")

    def _set_busy(self, value: bool) -> None:
        self.busy = value
        state = "disabled" if value else "normal"
        self.send_btn.configure(state=state)
        self.input_entry.configure(state=state)

    def _build_graph(self):
        if not settings.llm_enabled:
            return None, (
                "No OPENAI_API_KEY in .env. Running in offline fixed-output mode."
            )
        try:
            return build_graph(build_chat_model(), checkpointer=CHECKPOINTER), ""
        except LLMNotConfigured as exc:
            return None, str(exc)
        except Exception as exc:  # noqa: BLE001 - startup diagnostics in UI
            return None, f"Agent startup error: {exc}"

    def _reset_session(self, clear_chat: bool) -> None:
        PROJECT_DATA.reload()
        YIELD_DATASET.reload()
        MANUAL_INDEX.reload()
        self.thread_id = str(uuid4())
        self.graph, self.graph_error = self._build_graph()
        if clear_chat:
            self.chat_box.configure(state="normal")
            self.chat_box.delete("1.0", "end")
            self.chat_box.configure(state="disabled")

        if self.graph is None:
            self._set_status("Mode: Offline fixed-output")
            if self.graph_error:
                self._append_system(self.graph_error)
        else:
            self._set_status(f"Mode: LLM online ({settings.chat_model})")

    def _show_status(self) -> None:
        self._append_assistant(
            "\n".join(
                [
                    PROJECT_DATA.health_report(),
                    YIELD_DATASET.status_text(),
                    MANUAL_INDEX.status_text(),
                ]
            )
        )

    def _reset_clicked(self) -> None:
        if self.busy:
            return
        self._reset_session(clear_chat=True)
        self._append_system("Conversation and data cache reset.")
        self._append_assistant(
            "Ask about line status, tickets, manuals, or yield. Commands: /status /reset /quit"
        )

    def _offline_fixed_reply(self, question: str) -> str:
        return (
            "Offline fixed-output mode (no API key).\n"
            "I can still show configured Project Data status with /status, and you can use /reset.\n\n"
            "If OPENAI_API_KEY is present in .env, this same app switches to live agent mode."
        )

    def _run_graph_turn(self, question: str) -> str:
        inputs = {"messages": [HumanMessage(content=question)], "next": ""}
        run_config = {
            "recursion_limit": settings.recursion_limit,
            "configurable": {"thread_id": self.thread_id},
        }
        lines: list[str] = []

        try:
            for step in self.graph.stream(inputs, config=run_config):
                for node, update in step.items():
                    msgs = update.get("messages") if isinstance(update, dict) else None
                    if msgs:
                        lines.append(f"{node}: {msgs[-1].content.strip()}")
        except Exception as exc:  # noqa: BLE001 - keep app responsive on failures
            hint = ""
            text = str(exc)
            if "401" in text or "api_key" in text.lower() or "authentication" in text.lower():
                hint = "\n\nCheck OPENAI_API_KEY in .env (invalid or expired keys return 401)."
            return f"[error] {exc}{hint}"

        if not lines:
            return "(no response generated)"
        return "\n\n".join(lines)

    def _worker(self, question: str) -> None:
        if self.graph is None:
            answer = self._offline_fixed_reply(question)
        else:
            answer = self._run_graph_turn(question)
        self.result_queue.put(answer)

    def _poll_worker_results(self) -> None:
        try:
            while True:
                answer = self.result_queue.get_nowait()
                self._append_assistant(answer)
                self._set_busy(False)
                if self.graph is None:
                    self._set_status("Mode: Offline fixed-output")
                else:
                    self._set_status(f"Mode: LLM online ({settings.chat_model})")
                self.input_entry.focus_set()
        except queue.Empty:
            pass
        finally:
            self.root.after(120, self._poll_worker_results)

    def _on_send(self, _event=None):
        if self.busy:
            return "break"

        question = self.input_var.get().strip()
        if not question:
            return "break"

        self.input_var.set("")
        self._append_user(question)

        cmd = question.lower()
        if cmd in {"/quit", "/exit", "/q"}:
            self.root.destroy()
            return "break"
        if cmd == "/status":
            self._show_status()
            return "break"
        if cmd == "/reset":
            self._reset_clicked()
            return "break"

        self._set_busy(True)
        self._set_status("Thinking...")
        thread = threading.Thread(target=self._worker, args=(question,), daemon=True)
        thread.start()
        return "break"


def main() -> None:
    root = tk.Tk()
    FactoryChatPopup(root)
    root.mainloop()


if __name__ == "__main__":
    main()
