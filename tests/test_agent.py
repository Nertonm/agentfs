"""Integration tests for agent + multi_agent (no LLM server required)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from ctxvault.agent import Agent, _extract_tool_call, create_agent
from ctxvault.agent_notebook import AgentNotebook
from ctxvault.context_manager import ContextManager
from ctxvault.multi_agent import MultiAgent, PlanStep, Planner, Executor
from ctxvault.tools import ToolSet


# ---------------------------------------------------------------------------
# _extract_tool_call
# ---------------------------------------------------------------------------

def test_extract_json_block():
    text = 'Reasoning here.\n```json\n{"tool": "read_file", "args": {"path": "x.py"}}\n```\n'
    call = _extract_tool_call(text)
    assert call is not None
    assert call["tool"] == "read_file"
    assert call["args"]["path"] == "x.py"


def test_extract_no_json():
    text = "No tool call here."
    assert _extract_tool_call(text) is None


def test_extract_inline_json():
    text = 'Do this: {"tool": "list_dir", "args": {"path": "."}}'
    call = _extract_tool_call(text)
    assert call is not None
    assert call["tool"] == "list_dir"


# ---------------------------------------------------------------------------
# Agent (mocked LLM)
# ---------------------------------------------------------------------------

@pytest.fixture
def agent_setup(tmp_path):
    """Create a wired agent with mocked LLM."""
    (tmp_path / "sample.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    cm = ContextManager(budget=4096)
    nb = AgentNotebook(tmp_path)
    tools = ToolSet(tmp_path, context_manager=cm, notebook=nb)
    agent = Agent(
        tmp_path,
        tools=tools,
        context_manager=cm,
        notebook=nb,
        max_steps=3,
    )
    return agent, tmp_path


def test_agent_no_tool_call_returns_final(agent_setup):
    agent, tmp_path = agent_setup
    # LLM returns no JSON tool call → FINAL
    with patch.object(agent, "_call_llm", return_value="All done."):
        result = agent.run("Describe the project.")
    assert "FINAL:" in result or "All done." in result


def test_agent_tool_call_executed(agent_setup):
    agent, tmp_path = agent_setup
    responses = [
        '```json\n{"tool": "list_dir", "args": {"path": "."}}\n```',
        "All done.",
    ]
    with patch.object(agent, "_call_llm", side_effect=responses):
        result = agent.run("List the files.")
    assert isinstance(result, str)


def test_agent_state_updated(agent_setup):
    agent, tmp_path = agent_setup
    with patch.object(agent, "_call_llm", return_value="Done."):
        agent.run("Test objective")
    state = agent._nb.read_state()
    assert "Test objective" in state


def test_agent_session_report_written(agent_setup):
    agent, tmp_path = agent_setup
    with patch.object(agent, "_call_llm", return_value="Done."):
        agent.run("Task")
    report_path = tmp_path / ".agent-notes" / "SESSION_REPORT.md"
    assert report_path.exists()


# ---------------------------------------------------------------------------
# PlanStep
# ---------------------------------------------------------------------------

def test_plan_step_defaults():
    step = PlanStep(index=0, description="Do something")
    assert not step.done
    assert step.result == ""
    assert step.tool_subset == []


# ---------------------------------------------------------------------------
# Planner (mocked)
# ---------------------------------------------------------------------------

def test_planner_parse_valid_json():
    planner = Planner()
    raw = json.dumps({
        "steps": [
            {"description": "Read file", "tool_subset": ["read_file"],
             "expected_output": "content"},
            {"description": "Write result", "tool_subset": ["write_file"],
             "expected_output": "file written"},
        ]
    })
    steps = planner._parse_plan(raw)
    assert len(steps) == 2
    assert steps[0].description == "Read file"
    assert "read_file" in steps[0].tool_subset


def test_planner_parse_invalid_json():
    planner = Planner()
    steps = planner._parse_plan("not json at all, just text")
    assert len(steps) == 1
    assert steps[0].index == 0


def test_planner_parse_empty():
    planner = Planner()
    steps = planner._parse_plan("")
    assert len(steps) == 1


# ---------------------------------------------------------------------------
# Executor (mocked LLM)
# ---------------------------------------------------------------------------

@pytest.fixture
def executor_setup(tmp_path):
    (tmp_path / "data.py").write_text("x = 1\n", encoding="utf-8")
    cm = ContextManager(budget=4096)
    nb = AgentNotebook(tmp_path)
    tools = ToolSet(tmp_path, context_manager=cm, notebook=nb)
    executor = Executor(tools, cm, nb, max_turns=3)
    return executor, tmp_path


def test_executor_done_immediately(executor_setup):
    executor, tmp_path = executor_setup
    step = PlanStep(index=0, description="Do task", tool_subset=["read_file"])
    with patch.object(executor, "_call_llm",
                      return_value='```json\n{"tool": "done", "args": {"result": "found it"}}\n```'):
        result = executor.execute(step)
    assert result == "found it"
    assert step.done


def test_executor_tool_then_done(executor_setup):
    executor, tmp_path = executor_setup
    step = PlanStep(index=0, description="List files", tool_subset=["list_dir"])
    responses = [
        '```json\n{"tool": "list_dir", "args": {"path": "."}}\n```',
        '```json\n{"tool": "done", "args": {"result": "listed"}}\n```',
    ]
    with patch.object(executor, "_call_llm", side_effect=responses):
        result = executor.execute(step)
    assert result == "listed"


def test_executor_no_tool_call(executor_setup):
    executor, tmp_path = executor_setup
    step = PlanStep(index=0, description="Explain")
    with patch.object(executor, "_call_llm", return_value="Here is the explanation."):
        result = executor.execute(step)
    assert isinstance(result, str)
    assert step.done


# ---------------------------------------------------------------------------
# MultiAgent (fully mocked)
# ---------------------------------------------------------------------------

def test_multiagent_run(tmp_path):
    (tmp_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
    cm = ContextManager(budget=4096)
    nb = AgentNotebook(tmp_path)
    tools = ToolSet(tmp_path, context_manager=cm, notebook=nb)

    planner = Planner()
    executor = Executor(tools, cm, nb, max_turns=2)

    # Stub planner to return one step
    planner._parse_plan = lambda _: [
        PlanStep(index=0, description="List files", tool_subset=["list_dir"])
    ]
    with patch.object(planner, "_call_llm",
                      return_value=json.dumps({"steps": [
                          {"description": "List files",
                           "tool_subset": ["list_dir"],
                           "expected_output": "file list"}
                      ]})):
        with patch.object(executor, "_call_llm",
                          return_value='```json\n{"tool": "done", "args": {"result": "done"}}\n```'):
            ma = MultiAgent(planner, executor, nb, cm, vault_root=tmp_path)
            result = ma.run("Show the vault structure")

    assert isinstance(result, str)
