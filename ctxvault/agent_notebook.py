"""
agent_notebook.py — Markdown notebook for the agent.

Manages:
  .agent-notes/STATE.md          — current objective, constraints, plan, scratchpad
  .agent-notes/YYYY-MM-DD_<slug>.md — dated session notes
  .agent-notes/SESSION_REPORT.md — auto-generated session summary
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from ctxvault.config import NOTEBOOK_DIR, SESSION_REPORT_FILE, STATE_FILE


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:40]


class AgentNotebook:
    """
    Write/read Markdown notes and manage STATE.md.

    Parameters
    ----------
    base_dir : str | Path
        Root directory of the vault (notebook lives in base_dir/NOTEBOOK_DIR).
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)
        self._notes_dir = self._base / NOTEBOOK_DIR
        self._notes_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self._notes_dir / STATE_FILE
        self._report_path = self._notes_dir / SESSION_REPORT_FILE
        self._session_start = time.time()
        self._session_stats: Dict[str, int] = {}  # path → access count
        self._session_decisions: List[str] = []
        self._session_log: List[str] = []

        # Ensure STATE.md exists
        if not self._state_path.exists():
            self._state_path.write_text(
                "# STATE\n\n## Objective\n_none_\n\n"
                "## Constraints\n_none_\n\n"
                "## Plan\n_none_\n\n"
                "## Scratchpad\n",
                encoding="utf-8",
            )

    # ------------------------------------------------------------------
    # STATE.md
    # ------------------------------------------------------------------

    def read_state(self) -> str:
        return self._state_path.read_text(encoding="utf-8")

    def write_state(self, content: str) -> None:
        self._state_path.write_text(content, encoding="utf-8")

    def update_state_section(self, section: str, value: str) -> None:
        """Update a named section (## Objective, ## Plan, etc.)."""
        text = self.read_state()
        pattern = re.compile(
            rf"(## {re.escape(section)}\n)(.*?)(?=\n## |\Z)",
            re.DOTALL,
        )
        replacement = f"## {section}\n{value.rstrip()}\n\n"
        if pattern.search(text):
            text = pattern.sub(replacement, text)
        else:
            text = text.rstrip() + f"\n\n## {section}\n{value.rstrip()}\n"
        self.write_state(text)

    # ------------------------------------------------------------------
    # Session notes
    # ------------------------------------------------------------------

    def append_note(self, content: str, slug: str = "session") -> Path:
        """Append content to today's session note file."""
        date_str = time.strftime("%Y-%m-%d", time.gmtime())
        fname = f"{date_str}_{_slugify(slug)}.md"
        note_path = self._notes_dir / fname
        with note_path.open("a", encoding="utf-8") as fh:
            if note_path.stat().st_size == 0:
                fh.write(f"# Session notes — {date_str}\n\n")
            fh.write(content)
            if not content.endswith("\n"):
                fh.write("\n")
        return note_path

    def log_eviction(self, item_id: str, reason: str, summary: str = "") -> None:
        ts = time.strftime("%H:%M:%S", time.gmtime())
        line = f"- [{ts}] EVICT `{item_id}` reason={reason}"
        if summary:
            line += f" — {summary}"
        self._session_log.append(line)
        self.append_note(line + "\n", slug="eviction-log")

    def log_decision(self, decision: str) -> None:
        self._session_decisions.append(decision)
        ts = time.strftime("%H:%M:%S", time.gmtime())
        self.append_note(f"- [{ts}] DECISION: {decision}\n", slug="decisions")

    def record_file_access(self, path: str) -> None:
        self._session_stats[path] = self._session_stats.get(path, 0) + 1

    # ------------------------------------------------------------------
    # Session report
    # ------------------------------------------------------------------

    def write_session_report(
        self,
        *,
        step_count: int = 0,
        extra: Optional[str] = None,
    ) -> Path:
        """Auto-generate SESSION_REPORT.md on session end."""
        elapsed = time.time() - self._session_start
        lines: List[str] = [
            "# SESSION REPORT\n\n",
            f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n",
            f"**Duration**: {elapsed:.1f}s\n",
            f"**Steps**: {step_count}\n\n",
        ]

        # Most-accessed files
        if self._session_stats:
            sorted_files = sorted(
                self._session_stats.items(), key=lambda x: x[1], reverse=True
            )
            lines.append("## Most-accessed files\n\n")
            for path, count in sorted_files[:10]:
                lines.append(f"- `{path}` ({count}×)\n")
            lines.append("\n")

        # Decisions
        if self._session_decisions:
            lines.append("## Key decisions\n\n")
            for d in self._session_decisions:
                lines.append(f"- {d}\n")
            lines.append("\n")

        # Eviction log
        if self._session_log:
            lines.append("## Eviction log\n\n")
            for entry in self._session_log:
                lines.append(entry + "\n")
            lines.append("\n")

        if extra:
            lines.append("## Notes\n\n")
            lines.append(extra + "\n")

        report = "".join(lines)
        self._report_path.write_text(report, encoding="utf-8")
        return self._report_path

    def read_session_report(self) -> str:
        if self._report_path.exists():
            return self._report_path.read_text(encoding="utf-8")
        return ""
