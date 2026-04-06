"""
agent_notebook.py - Markdown notebook for the agent.

Manages:
  .agent-notes/STATE.md              - current sequence state, updated robustly using regex.
  .agent-notes/YYYY-MM-DD_<slug>.md  - dated session notes & specific cache logs.
  .agent-notes/SESSION_REPORT.md     - auto-generated session summary at exit.

Combines the regex features of Implementation A and the reporting of Implementation B.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from config import NOTEBOOK_DIR, STATE_ILE, SESSION_REPORT_ILE

def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:40]

class AgentNotebook:
    def __init__(self, workspace: str | Path):
        self._base = Path(workspace).resolve()
        self._notes_dir = self._base / NOTEBOOK_DIR
        self._notes_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._notes_dir / STATE_ILE
        self._report_path = self._notes_dir / SESSION_REPORT_ILE
        
        # Session metrics
        self._session_start = time.time()
        self.metrics = {
            "evictions": 0,
            "decisions": 0,
            "files_hit": {}
        }
        self._session_decisions: List[str] = []
        self._session_log: List[str] = []

        self._ensure_state()


    def _ensure_state(self):
        if not self._state_path.exists():
            self._state_path.write_text(
                "# STATE\n\n"
                "## Objective\n_(none)_\n\n"
                "## Plan\n_(none)_\n\n"
                "## Decisions\n_(none)_\n\n"
                "## Scratchpad\n_(none)_\n",
                encoding="utf-8",
            )

    def read_state(self) -> str:
        return self._state_path.read_text(encoding="utf-8")

    def write_state(self, content: str) -> str:
        self._state_path.write_text(content, encoding="utf-8")
        return "STATE.md atualizado."

    def update_state_section(self, section: str, content: str) -> str:
        """Atualiza uma seção específica do STATE.md pelo nome do heading."""
        text = self.read_state()
        pattern = rf"(## {re.escape(section)}\n)(.*?)(?=\n## |\Z)"
        replacement = rf"\g<1>{content}\n"
        new_text, n = re.subn(pattern, replacement, text, flags=re.DOTALL)
        if n == 0:
            #Seção não existe - cria no final
            new_text = text.rstrip() + f"\n\n## {section}\n{content}\n"
        self._state_path.write_text(new_text, encoding="utf-8")
        return f"Seção '{section}' atualizada."


    def add_note(self, title: str, content: str, **kwargs) -> str:
        """Cria uma nota de sessão Markdown datada."""
        now = datetime.now()
        slug = _slugify(title)
        filename = f"{now.strftime('%Y-%m-%d')}_{slug}.md"
        note_path = self._notes_dir / filename

        lines = [
            f"# {title}",
            f"_Created: {now.strftime('%Y-%m-%d %H:%M')}_",
            "",
            "## Content",
            content,
            ""
        ]
        
        with note_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        return f"Nota salva: {filename}"

    def append_note(self, content: str, slug: str = "session") -> Path:
        """Append content to today's session note file (Legacy API)."""
        now = datetime.now()
        filename = f"{now.strftime('%Y-%m-%d')}_{_slugify(slug)}.md"
        note_path = self._notes_dir / filename
        with note_path.open("a", encoding="utf-8") as fh:
            if note_path.stat().st_size == 0:
                fh.write(f"#Notas da Sessão - {now.strftime('%Y-%m-%d')}\n\n")
            fh.write(content)
            if not content.endswith("\n"):
                fh.write("\n")
        return note_path


    def log_eviction(self, item_id: str, reason: str, summary: str = "") -> None:
        self.metrics["evictions"] += 1
        ts = time.strftime("%H:%M:%S", time.gmtime())
        line = f"- [{ts}] EVICT `{item_id}` reason={reason}"
        if summary:
            line += f" - {summary}"
        self._session_log.append(line)
        self.add_note("Cache Eviction Log", line)

    def log_decision(self, decision: str) -> None:
        self.metrics["decisions"] += 1
        self._session_decisions.append(decision)
        ts = time.strftime("%H:%M:%S", time.gmtime())
        self.add_note("Agent Decisions", f"- [{ts}] {decision}")

    def record_file_access(self, path: str) -> None:
        self.metrics["files_hit"][path] = self.metrics["files_hit"].get(path, 0) + 1

    @property
    def _session_stats(self):
        """Legacy compatibility."""
        return self.metrics["files_hit"]


    def write_session_report(self, step_count: int = 0) -> Path:
        """Gera SESSION_REPORT.md no final da sessão (via atexit)."""
        elapsed = time.time() - self._session_start
        lines: List[str] = [
            "# SESSION REPORT\n\n",
            f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n",
            f"**Duration**: {elapsed:.1f}s\n",
            f"**Steps**: {step_count}\n",
            f"**Evictions**: {self.metrics['evictions']}\n",
            f"**Decisions Logged**: {self.metrics['decisions']}\n\n",
        ]

        if self.metrics["files_hit"]:
            sorted_files = sorted(
                self.metrics["files_hit"].items(), key=lambda x: x[1], reverse=True
            )
            lines.append("## Most-accessed files\n\n")
            for p, count in sorted_files[:10]:
                lines.append(f"- `{p}` ({count}×)\n")
            lines.append("\n")

        if self._session_decisions:
            lines.append("## Key decisions\n\n")
            for d in self._session_decisions[-10:]:
                lines.append(f"- {d}\n")
            lines.append("\n")

        self._report_path.write_text("".join(lines), encoding="utf-8")
        return self._report_path

    def read_session_report(self) -> str:
        if self._report_path.exists():
            return self._report_path.read_text(encoding="utf-8")
        return ""
