"""
agent.py — Main agent loop implementing ReAct (Reason → JSON tool call).

Architecture
------------
* System prompt always includes VAULT_MAP.md + STATE.md (working context).
* Each turn: model produces reasoning text + a JSON tool call block.
* JSON tool call format (plain text, no function_call API needed)::

    ```json
    {
      "tool": "read_file",
      "args": {"path": "ctxvault/config.py", "start_line": 0}
    }
    ```

* Tool result is fed back as the next user message.
* Context manager tracks token usage and runs eviction if needed.
* Re-read warning emitted if same file read ≥ REREAD_WARN_THRESHOLD times.
* Status line emitted every turn: [CTX: 68% | ZONE: normal | ...]

Offline-first
-------------
* Calls llama.cpp-compatible HTTP server (configurable via LLAMA_SERVER_URL).
* If server is unreachable, falls back to a simple echo/stub for testing.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

from ctxvault.agent_notebook import AgentNotebook
from ctxvault.config import (
    CONTEXT_BUDGET,
    LLAMA_COMPLETIONS_PATH,
    LLAMA_MAX_TOKENS,
    LLAMA_SERVER_URL,
    LLAMA_TEMPERATURE,
    MODEL_RESPONSE_TOKENS,
    NOTEBOOK_DIR,
    REREAD_WARN_THRESHOLD,
    SYSTEM_PROMPT_TOKENS,
    TOOLS_TOKENS,
    VAULT_MAP_TOKENS,
)
from ctxvault.context_manager import ContextManager
from ctxvault.tools import ToolSet

# Regex to extract JSON tool call from model output
_JSON_BLOCK_RE = re.compile(
    r"```json\s*(\{.*?\})\s*```",
    re.DOTALL,
)

_TOOL_DESCRIPTIONS = """
Available tools (call exactly one per turn as JSON inside ```json ... ```):

list_dir(path, depth, filters)          — list directory tree
search_text(query, paths, regex)        — grep files
read_file(path, start_line, end_line)   — read file (paginated)
read_symbols(path)                      — list functions/classes
write_file(path, content)               — write file
append_file(path, content)              — append to file
summarize_to_cache(item_id)             — summarise file into cache
retrieve_candidates(query, k, filters)  — hybrid BM25+semantic+graph search
pin(item_id) / unpin(item_id)           — pin/unpin cache item
run_command(cmd)                        — run shell command (if enabled)
""".strip()


def _build_system_prompt(vault_map: str, state: str) -> str:
    return (
        f"You are an AI coding agent.\n\n"
        f"## VAULT_MAP\n{vault_map}\n\n"
        f"## STATE\n{state}\n\n"
        f"## Tools\n{_TOOL_DESCRIPTIONS}\n\n"
        "Reason step by step, then emit exactly one JSON tool call per turn.\n"
        "Format: ```json\\n{\"tool\": \"...\", \"args\": {...}}\\n```"
    )


def _extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON tool call from model output."""
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Find any JSON object containing "tool" key by scanning for balanced braces
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:end + 1]
                    try:
                        obj = json.loads(candidate)
                        if "tool" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
    return None


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class Agent:
    """
    ReAct agent loop.

    Parameters
    ----------
    vault_root : str | Path
        Root of the vault to operate on.
    tools : ToolSet
        Pre-built tool set.
    context_manager : ContextManager
        Tracks token budget.
    notebook : AgentNotebook
        For session notes and STATE.md.
    graph_active : bool
        Whether the graph retrieval is currently active.
    """

    def __init__(
        self,
        vault_root: str | Path,
        *,
        tools: ToolSet,
        context_manager: ContextManager,
        notebook: AgentNotebook,
        graph_active: bool = True,
        max_steps: int = 50,
    ) -> None:
        self._root = Path(vault_root).resolve()
        self._tools = tools
        self._cm = context_manager
        self._nb = notebook
        self._graph_active = graph_active
        self._max_steps = max_steps
        self._history: List[Dict[str, str]] = []  # {role, content}
        self._step_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str) -> str:
        """Run the agent on a task until completion or max_steps."""
        self._nb.update_state_section("Objective", task)
        self._nb.log_decision(f"Starting task: {task}")

        # Register eviction callback
        self._cm._on_evict = self._handle_eviction

        result = ""
        for _step in range(self._max_steps):
            self._step_count += 1
            result = self._step(task)
            if result.startswith("DONE:") or result.startswith("FINAL:"):
                break
            # Check memory pressure
            evicted = self._cm.maybe_evict()
            for item_id in evicted:
                self._nb.log_eviction(item_id, "pressure", "evicted by context manager")
            # Critical zone: compress history
            if self._cm.zone() == "critical":
                kept, flushed = self._cm.compress_history(self._history)
                if flushed:
                    flush_text = "\n".join(
                        f"[{m['role']}]: {m['content'][:80]}" for m in flushed
                    )
                    self._nb.append_note(
                        f"## Flushed history ({len(flushed)} msgs)\n{flush_text}\n",
                        slug="history-flush",
                    )
                    self._history = kept

        self._nb.write_session_report(step_count=self._step_count)
        return result

    def _step(self, task: str) -> str:
        """One ReAct step: build prompt → call LLM → parse tool → execute."""
        vault_map = self._load_vault_map()
        state = self._nb.read_state()

        system_prompt = _build_system_prompt(vault_map, state)
        messages = self._history[-20:]  # keep last 20 turns in context

        # Emit status line
        status = self._cm.status_line(
            graph_active=self._graph_active,
            step_count=self._step_count,
        )
        print(status)

        # Add current task as user message if first step
        if self._step_count == 1:
            user_msg = f"Task: {task}"
        else:
            user_msg = self._history[-1]["content"] if self._history else f"Task: {task}"

        model_output = self._call_llm(system_prompt, messages, user_msg)

        # Parse tool call
        tool_call = _extract_tool_call(model_output)
        if tool_call is None:
            # No tool call — treat as final answer
            self._history.append({"role": "assistant", "content": model_output})
            return f"FINAL: {model_output}"

        tool_name = tool_call.get("tool", "")
        tool_args = tool_call.get("args", {})

        # Execute tool
        tool_result = self._tools.dispatch(tool_name, tool_args)
        result_text = tool_result.get("output", str(tool_result))

        # Track tokens
        self._cm.add(
            f"step:{self._step_count}:tool_result",
            result_text,
            importance=0.4,
            tokens=_approx_tokens(result_text),
        )

        self._history.append({"role": "assistant", "content": model_output})
        self._history.append(
            {
                "role": "user",
                "content": f"Tool `{tool_name}` result:\n{result_text}",
            }
        )
        return result_text

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        user_msg: str,
    ) -> str:
        """Call llama.cpp-compatible server. Fallback: stub."""
        prompt = self._build_prompt(system_prompt, messages, user_msg)
        url = LLAMA_SERVER_URL.rstrip("/") + LLAMA_COMPLETIONS_PATH
        payload = json.dumps(
            {
                "prompt": prompt,
                "n_predict": LLAMA_MAX_TOKENS,
                "temperature": LLAMA_TEMPERATURE,
                "stop": ["```\n\n"],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("content", body.get("text", ""))
        except (urllib.error.URLError, OSError):
            # Server not available — return a stub "no-op" response
            return (
                "I was unable to reach the LLM server. "
                "Please start llama.cpp server and retry.\n"
            )

    def _build_prompt(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        user_msg: str,
    ) -> str:
        """Build a simple ChatML-style prompt string."""
        parts = [f"<|system|>\n{system_prompt}\n</s>"]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}\n</s>")
        parts.append(f"<|user|>\n{user_msg}\n</s>")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_vault_map(self) -> str:
        map_path = self._root / "VAULT_MAP.md"
        if map_path.exists():
            return map_path.read_text(encoding="utf-8")[:VAULT_MAP_TOKENS * 4]
        return "(VAULT_MAP not yet generated)"

    def _handle_eviction(self, item_id: str, reason: str) -> None:
        self._nb.log_eviction(item_id, reason)


# ---------------------------------------------------------------------------
# Factory / entry point
# ---------------------------------------------------------------------------

def create_agent(
    vault_root: str | Path = ".",
    *,
    db_path: Optional[str] = None,
    allow_run_command: bool = False,
    max_steps: int = 50,
) -> Agent:
    """Create a fully wired agent for the given vault root."""
    from ctxvault.config import DB_PATH
    from ctxvault.retriever import Retriever
    from ctxvault.vault_indexer import VaultIndexer

    vault_root = Path(vault_root).resolve()
    _db = db_path or str(vault_root / "index.sqlite")

    indexer = VaultIndexer(vault_root, db_path=_db)
    indexer.index_all()

    retriever = Retriever(db_path=_db)
    cm = ContextManager(budget=CONTEXT_BUDGET)
    notebook = AgentNotebook(vault_root)
    tools = ToolSet(
        vault_root,
        retriever=retriever,
        context_manager=cm,
        notebook=notebook,
        allow_run_command=allow_run_command,
    )
    return Agent(
        vault_root,
        tools=tools,
        context_manager=cm,
        notebook=notebook,
        max_steps=max_steps,
    )


if __name__ == "__main__":
    import sys

    task = " ".join(sys.argv[1:]) or "Summarise the project."
    agent = create_agent(vault_root=".")
    print(agent.run(task))
