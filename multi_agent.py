"""
multi_agent.py - Planner / Executor split for agentfs.

Architecture
------------
* **Planner**: Receives the full task, produces a step-by-step plan.
  Each plan step includes: description, tool_subset, context_budget.
* **Executor**: Receives one step + compact history (~30t/step) + 3–4 tools.
  Calls the LLM and executes the assigned tools.
* Steps communicate via shared ContextManager and AgentNotebook.

Context budget per step (4096 tokens):
  system prompt: ~250  | tools: ~150 | VAULT_MAP: ~200
  history (N×30t)      | model response: ~500
"""

from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_notebook import AgentNotebook
from config import (
    CONTEXT_BUDGET,
    HISTORY_TOKENS_PER_STEP,
    LLAMA_COMPLETIONS_PATH,
    LLAMA_MAX_TOKENS,
    LLAMA_SERVER_URL,
    LLAMA_TEMPERATURE,
    MODEL_RESPONSE_TOKENS,
    SYSTEM_PROMPT_TOKENS,
    TOOLS_TOKENS,
    VAULT_MAP_TOKENS,
)
from context_manager import ContextManager
from tools import ToolSet


# ---------------------------------------------------------------------------
# Plan step dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    index: int
    description: str
    tool_subset: List[str] = field(default_factory=list)
    expected_output: str = ""
    done: bool = False
    result: str = ""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """You are a task planner for an AI coding agent.

Given a task, produce a JSON plan with steps. Each step has:
- description: str  - what to do
- tool_subset: list[str]  - which tools to use (from the available set)
- expected_output: str  - what you expect to learn/produce

Available tools:
list_dir, search_text, read_file, read_symbols, write_file, append_file,
summarize_to_cache, retrieve_candidates, pin, unpin, run_command

Respond ONLY with a JSON object like:
{
  "steps": [
    {"description": "...", "tool_subset": ["read_file"], "expected_output": "..."},
    ...
  ]
}
""".strip()


class Planner:
    """Calls the LLM to decompose a task into plan steps."""

    def __init__(self, vault_map: str = "") -> None:
        self._vault_map = vault_map

    def plan(self, task: str) -> List[PlanStep]:
        prompt = self._build_prompt(task)
        raw = self._call_llm(prompt)
        return self._parse_plan(raw)

    def _build_prompt(self, task: str) -> str:
        parts = [f"<|system|>\n{_PLANNER_SYSTEM}\n</s>"]
        if self._vault_map:
            parts.append(
                f"<|user|>\nVAULT_MAP:\n{self._vault_map[:800]}\n\nTask: {task}\n</s>"
            )
        else:
            parts.append(f"<|user|>\nTask: {task}\n</s>")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        url = LLAMA_SERVER_URL.rstrip("/") + LLAMA_COMPLETIONS_PATH
        payload = json.dumps(
            {
                "prompt": prompt,
                "n_predict": LLAMA_MAX_TOKENS,
                "temperature": 0.2,
                "stop": ["\n\n\n"],
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
            return ""

    def _parse_plan(self, raw: str) -> List[PlanStep]:
        """Extract steps from the LLM output."""
        # Try to find JSON object
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                steps_data = data.get("steps", [])
                steps: List[PlanStep] = []
                for i, s in enumerate(steps_data):
                    steps.append(
                        PlanStep(
                            index=i,
                            description=s.get("description", f"Step {i}"),
                            tool_subset=s.get("tool_subset", []),
                            expected_output=s.get("expected_output", ""),
                        )
                    )
                return steps
            except (json.JSONDecodeError, KeyError):
                pass
        # Fallback: single step
        return [PlanStep(index=0, description=raw or "Execute task", tool_subset=[])]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

_EXECUTOR_SYSTEM = """You are an executor agent. You receive one step from a plan and must execute it.

Use exactly one tool per turn via JSON:
```json
{{"tool": "<name>", "args": {{...}}}}
```

When the step is complete, respond with:
```json
{{"tool": "done", "args": {{"result": "<summary of what you found/did>"}}}}
```

Step: {step_description}
Expected output: {expected_output}
Available tools: {tool_subset}
""".strip()


class Executor:
    """Executes a single plan step via ReAct loop."""

    def __init__(
        self,
        tools: ToolSet,
        context_manager: ContextManager,
        notebook: AgentNotebook,
        *,
        max_turns: int = 10,
    ) -> None:
        self._tools = tools
        self._cm = context_manager
        self._nb = notebook
        self._max_turns = max_turns

    def execute(
        self,
        step: PlanStep,
        compact_history: str = "",
        vault_map: str = "",
    ) -> str:
        """Execute a plan step. Returns a result summary."""
        system = _EXECUTOR_SYSTEM.format(
            step_description=step.description,
            expected_output=step.expected_output,
            tool_subset=", ".join(step.tool_subset) or "all",
        )
        if vault_map:
            system += f"\n\nVAULT_MAP:\n{vault_map[:400]}"
        if compact_history:
            system += f"\n\nHistory:\n{compact_history}"

        history: List[Dict[str, str]] = []

        for turn in range(self._max_turns):
            prompt = self._build_prompt(system, history)
            raw = self._call_llm(prompt)

            tool_call = self._extract_tool_call(raw)
            if tool_call is None:
                step.result = raw
                step.done = True
                return raw

            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})

            if tool_name == "done":
                result = tool_args.get("result", raw)
                step.result = result
                step.done = True
                return result

            # Execute tool
            tool_result = self._tools.dispatch(tool_name, tool_args)
            result_text = tool_result.get("output", str(tool_result))

            history.append({"role": "assistant", "content": raw})
            history.append(
                {
                    "role": "user",
                    "content": f"Tool `{tool_name}` result:\n{result_text}",
                }
            )

            # Evict if needed
            self._cm.maybe_evict()

        step.done = True
        step.result = history[-1]["content"] if history else "(no result)"
        return step.result

    def _build_prompt(
        self,
        system: str,
        history: List[Dict[str, str]],
    ) -> str:
        parts = [f"<|system|>\n{system}\n</s>"]
        for msg in history[-6:]:  # last 3 turns
            parts.append(f"<|{msg['role']}|>\n{msg['content']}\n</s>")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def _call_llm(self, prompt: str) -> str:
        url = LLAMA_SERVER_URL.rstrip("/") + LLAMA_COMPLETIONS_PATH
        payload = json.dumps(
            {
                "prompt": prompt,
                "n_predict": LLAMA_MAX_TOKENS,
                "temperature": LLAMA_TEMPERATURE,
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
            return '```json\n{"tool": "done", "args": {"result": "LLM server unavailable"}}\n```'

    @staticmethod
    def _extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Scan for balanced-brace JSON object containing "tool"
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


# ---------------------------------------------------------------------------
# MultiAgent - orchestrates Planner + Executor
# ---------------------------------------------------------------------------

class MultiAgent:
    """
    Planner/Executor multi-agent system.

    Usage::

        ma = MultiAgent.create(vault_root=".")
        result = ma.run("Refactor the config module")
    """

    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        notebook: AgentNotebook,
        context_manager: ContextManager,
        *,
        vault_root: Path,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._nb = notebook
        self._cm = context_manager
        self._root = vault_root

    def run(self, task: str) -> str:
        self._nb.update_state_section("Objective", task)
        self._nb.log_decision(f"[MultiAgent] Starting task: {task}")

        vault_map = self._load_vault_map()
        self._planner._vault_map = vault_map

        plan = self._planner.plan(task)
        self._nb.log_decision(f"Plan: {len(plan)} steps")

        compact_history = ""
        results: List[str] = []

        for step in plan:
            self._nb.log_decision(f"Executing step {step.index}: {step.description}")
            result = self._executor.execute(
                step,
                compact_history=compact_history,
                vault_map=vault_map,
            )
            results.append(f"Step {step.index}: {result[:80]}")
            # Build compact history (30 tokens per step)
            compact_history += f"[{step.index}] {step.description[:60]}: {result[:60]}\n"
            # Trim compact history to budget
            lines = compact_history.splitlines()
            max_lines = max(1, (CONTEXT_BUDGET - SYSTEM_PROMPT_TOKENS - TOOLS_TOKENS
                                - VAULT_MAP_TOKENS - MODEL_RESPONSE_TOKENS)
                            // HISTORY_TOKENS_PER_STEP)
            if len(lines) > max_lines:
                compact_history = "\n".join(lines[-max_lines:]) + "\n"

        final = "\n".join(results)
        self._nb.write_session_report(step_count=len(plan))
        return final

    def _load_vault_map(self) -> str:
        map_path = self._root / "VAULT_MAP.md"
        if map_path.exists():
            return map_path.read_text(encoding="utf-8")[:VAULT_MAP_TOKENS * 4]
        return ""

    @classmethod
    def create(
        cls,
        vault_root: str | Path = ".",
        *,
        db_path: Optional[str] = None,
        allow_run_command: bool = False,
    ) -> "MultiAgent":
        vault_root = Path(vault_root).resolve()
        _db = db_path or str(vault_root / "index.sqlite")

        from retriever import Retriever
        from vault_indexer import VaultIndexer

        indexer = VaultIndexer(vault_root, db_path=_db)
        indexer.scan()

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
        planner = Planner()
        executor = Executor(tools, cm, notebook)
        return cls(planner, executor, notebook, cm, vault_root=vault_root)


if __name__ == "__main__":
    import sys

    task = " ".join(sys.argv[1:]) or "Summarise the project."
    ma = MultiAgent.create(vault_root=".")
    print(ma.run(task))
