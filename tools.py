"""
tools.py - All 10 tool implementations for the agentfs agent.

Tools
-----
list_dir         - directory listing with optional depth/filter
search_text      - text search (literal or regex) across paths
read_file        - paginated file reading (start_line/end_line)
read_symbols     - extract functions/classes via regex (tree-sitter upgrade path)
write_file       - write a file (creates parents)
append_file      - append to agent notebook or any file
summarize_to_cache  - generate a short summary and store it
retrieve_candidates - BM25 + semantic + graph retrieval via RR
pin / unpin      - pin/unpin cache items
run_command      - (optional) sandboxed shell command

Each tool returns a compact dict with at least:
  {"ok": bool, "output": str, ...pagination fields...}
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config import RETRIEVAL_TOP_K


# ---------------------------------------------------------------------------
# Pagination helpers
# ---------------------------------------------------------------------------

_PAGE_SIZE = 100  # lines per page for read_file


def _paginate_lines(
    lines: List[str], start: int, end: Optional[int]
) -> tuple[List[str], bool]:
    """Return (page_lines, has_more)."""
    total = len(lines)
    end_idx = min(end + 1, total) if end is not None else min(start + _PAGE_SIZE, total)
    page = lines[start:end_idx]
    has_more = end_idx < total
    return page, has_more


# ---------------------------------------------------------------------------
# ToolSet
# ---------------------------------------------------------------------------

class ToolSet:
    """
    Container for all agent tools.  Instantiate once per session and pass
    ``retriever`` and ``context_manager`` if available.
    """

    def __init__(
        self,
        vault_root: str | Path,
        *,
        retriever: Optional[Any] = None,
        context_manager: Optional[Any] = None,
        notebook: Optional[Any] = None,
        allow_run_command: bool = False,
    ) -> None:
        self._root = Path(vault_root).resolve()
        self._retriever = retriever
        self._cm = context_manager
        self._notebook = notebook
        self._allow_run_command = allow_run_command
        self.current_step: int = 0
        # Re-read observability: track per path per step
        self._read_counts: Dict[str, List[int]] = {}  # path → [step_indices]

    # ------------------------------------------------------------------
    # list_dir
    # ------------------------------------------------------------------

    def list_dir(
        self,
        path: str = ".",
        depth: int = 2,
        filters: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """List directory contents up to ``depth`` levels deep."""
        target = (self._root / path).resolve()
        if not target.exists():
            return {"ok": False, "output": f"Path not found: {path}"}

        lines: List[str] = []
        self._tree(target, depth=depth, current_depth=0, filters=filters, lines=lines)
        return {
            "ok": True,
            "output": "\n".join(lines),
            "path": str(target.relative_to(self._root)),
        }

    def _tree(
        self,
        path: Path,
        *,
        depth: int,
        current_depth: int,
        filters: Optional[List[str]],
        lines: List[str],
    ) -> None:
        indent = "  " * current_depth
        if path.is_file():
            lines.append(f"{indent}{path.name}")
            return
        if path.is_dir():
            lines.append(f"{indent}{path.name}/")
            if current_depth >= depth:
                return
            try:
                children = sorted(path.iterdir())
            except PermissionError:
                return
            for child in children:
                if filters:
                    if not any(
                        re.search(f, child.name, re.IGNORECASE) for f in filters
                    ):
                        continue
                self._tree(
                    child,
                    depth=depth,
                    current_depth=current_depth + 1,
                    filters=filters,
                    lines=lines,
                )

    # ------------------------------------------------------------------
    # search_text
    # ------------------------------------------------------------------

    def search_text(
        self,
        query: str,
        paths: Optional[List[str]] = None,
        regex: bool = False,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        """Search for text in files under the given paths."""
        search_roots: List[Path] = []
        if paths:
            for p in paths:
                resolved = (self._root / p).resolve()
                if resolved.exists():
                    search_roots.append(resolved)
        if not search_roots:
            search_roots = [self._root]

        results: List[str] = []
        try:
            pattern = re.compile(query, re.IGNORECASE) if regex else None
        except re.error as exc:
            return {"ok": False, "output": f"Invalid regex: {exc}"}

        for root in search_roots:
            for fpath in root.rglob("*"):
                if not fpath.is_file():
                    continue
                if len(results) >= max_results:
                    break
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                rel = str(fpath.relative_to(self._root))
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if (
                        (pattern and pattern.search(line))
                        or (not pattern and query.lower() in line.lower())
                    ):
                        results.append(f"{rel}:{lineno}: {line.rstrip()}")
                        if len(results) >= max_results:
                            break

        truncated = len(results) == max_results
        return {
            "ok": True,
            "output": "\n".join(results) if results else "(no matches)",
            "count": len(results),
            "truncated": truncated,
        }

    # ------------------------------------------------------------------
    # read_file
    # ------------------------------------------------------------------

    def read_file(
        self,
        path: str,
        start_line: int = 0,
        end_line: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Read a file (or range of lines). Lines are 0-indexed."""
        fpath = (self._root / path).resolve()
        if not fpath.is_file():
            return {"ok": False, "output": f"File not found: {path}"}

        # Observability: re-read tracking
        self._track_read(str(fpath))

        try:
            lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return {"ok": False, "output": str(exc)}

        page, has_more = _paginate_lines(lines, start_line, end_line)
        if self._notebook:
            self._notebook.record_file_access(path)
        return {
            "ok": True,
            "output": "\n".join(page),
            "start_line": start_line,
            "end_line": start_line + len(page) - 1,
            "total_lines": len(lines),
            "has_more": has_more,
        }

    def _track_read(self, path: str) -> None:
        ts_list = self._read_counts.setdefault(path, [])
        ts_list.append(self.current_step)
        # Check for re-read warning: ≥ threshold reads across last 2 steps
        recent = [s for s in ts_list if self.current_step - s <= 2]
        if len(recent) >= 3:
            print(
                f"[WARN] read_file called {len(recent)}× on '{path}' "
                "in the last ≤ 2 steps - possible re-read loop"
            )

    # ------------------------------------------------------------------
    # read_symbols
    # ------------------------------------------------------------------

    def read_symbols(self, path: str) -> Dict[str, Any]:
        """Extract top-level symbols (functions, classes) via regex."""
        fpath = (self._root / path).resolve()
        if not fpath.is_file():
            return {"ok": False, "output": f"File not found: {path}"}

        try:
            lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return {"ok": False, "output": str(exc)}

        symbols: List[str] = []
        sym_re = re.compile(
            r"^(?P<kind>def |class |async def |function |"
            r"const |export function |pub fn |func )"
            r"(?P<name>\w+)"
        )
        for lineno, line in enumerate(lines, start=1):
            m = sym_re.match(line)
            if m:
                symbols.append(
                    f"L{lineno}: {m.group('kind').strip()} {m.group('name')}"
                )
        return {
            "ok": True,
            "output": "\n".join(symbols) if symbols else "(no symbols found)",
            "count": len(symbols),
        }

    # ------------------------------------------------------------------
    # write_file
    # ------------------------------------------------------------------

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to a file (creates parent directories)."""
        fpath = (self._root / path).resolve()
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_text(content, encoding="utf-8")
        return {"ok": True, "output": f"Written {len(content)} chars to {path}"}

    # ------------------------------------------------------------------
    # append_file
    # ------------------------------------------------------------------

    def append_file(self, path: str, content: str) -> Dict[str, Any]:
        """Append content to a file (creates if missing)."""
        fpath = (self._root / path).resolve()
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with fpath.open("a", encoding="utf-8") as fh:
            fh.write(content)
        return {"ok": True, "output": f"Appended {len(content)} chars to {path}"}

    # ------------------------------------------------------------------
    # summarize_to_cache
    # ------------------------------------------------------------------

    def summarize_to_cache(self, item_id: str) -> Dict[str, Any]:
        """
        Generate a short summary for ``item_id`` and store it in the context cache.

        Storage is conditional on a ``context_manager`` being provided at construction.
        ``item_id`` should be a path (optionally with ``:``) that can be read.
        """
        # Parse item_id → path[:chunk_index]
        parts = item_id.split(":")
        path = parts[0]
        fpath = (self._root / path).resolve()
        if not fpath.is_file():
            return {"ok": False, "output": f"File not found: {path}"}

        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"ok": False, "output": str(exc)}

        # Simple extractive summary: first non-empty lines up to 10
        lines = [l for l in text.splitlines() if l.strip()]
        summary_lines = lines[:10]
        summary = "\n".join(summary_lines)

        if self._cm:
            self._cm.add(
                item_id,
                summary,
                importance=0.6,
            )
        return {"ok": True, "output": summary, "item_id": item_id}

    # ------------------------------------------------------------------
    # retrieve_candidates
    # ------------------------------------------------------------------

    def retrieve_candidates(
        self,
        query: str,
        k: int = RETRIEVAL_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve top-k candidates via BM25 + semantic + graph (RR)."""
        if self._retriever is None:
            return {
                "ok": False,
                "output": "Retriever not initialised (no index available).",
            }
        results = self._retriever.retrieve(query, k=k, filters=filters)
        output_lines: List[str] = []
        for r in results:
            rel_path = r.get("path", "?")
            try:
                rel_path = str(
                    Path(rel_path).relative_to(self._root)
                )
            except ValueError:
                pass
            snippet = (r.get("content") or "")[:120].replace("\n", " ")
            output_lines.append(
                f"[{r['rrf_score']:.4f}] {rel_path}:{r['start_line']}  {snippet}"
            )
        return {
            "ok": True,
            "output": "\n".join(output_lines) if output_lines else "(no results)",
            "count": len(results),
            "results": results,
        }

    # ------------------------------------------------------------------
    # pin / unpin
    # ------------------------------------------------------------------

    def pin(self, item_id: str) -> Dict[str, Any]:
        if self._cm:
            self._cm.pin(item_id)
            return {"ok": True, "output": f"Pinned: {item_id}"}
        return {"ok": False, "output": "Context manager not available."}

    def unpin(self, item_id: str) -> Dict[str, Any]:
        if self._cm:
            self._cm.unpin(item_id)
            return {"ok": True, "output": f"Unpinned: {item_id}"}
        return {"ok": False, "output": "Context manager not available."}

    # ------------------------------------------------------------------
    # run_command (optional, sandboxed)
    # ------------------------------------------------------------------

    def run_command(
        self,
        cmd: str,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Run a shell command (only if allow_run_command=True)."""
        if not self._allow_run_command:
            return {
                "ok": False,
                "output": "run_command is disabled (allow_run_command=False).",
            }
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self._root),
            )
            out = (result.stdout or "") + (result.stderr or "")
            return {
                "ok": result.returncode == 0,
                "output": out[:4000],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "output": f"Command timed out after {timeout}s"}
        except Exception as exc:
            return {"ok": False, "output": str(exc)}

    # ------------------------------------------------------------------
    # Dispatch helper (for agent loop)
    # ------------------------------------------------------------------

    def dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by name with a kwargs dict."""
        mapping = {
            "list_dir": self.list_dir,
            "search_text": self.search_text,
            "read_file": self.read_file,
            "read_symbols": self.read_symbols,
            "write_file": self.write_file,
            "append_file": self.append_file,
            "summarize_to_cache": self.summarize_to_cache,
            "retrieve_candidates": self.retrieve_candidates,
            "pin": self.pin,
            "unpin": self.unpin,
            "run_command": self.run_command,
        }
        fn = mapping.get(name)
        if fn is None:
            return {"ok": False, "output": f"Unknown tool: {name}"}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"ok": False, "output": f"Tool call error: {exc}"}
