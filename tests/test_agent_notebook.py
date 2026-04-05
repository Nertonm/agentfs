"""Tests for ctxvault.agent_notebook."""

import time
from pathlib import Path

import pytest

from ctxvault.agent_notebook import AgentNotebook, _slugify


def test_slugify():
    assert _slugify("Hello World") == "hello-world"
    assert _slugify("Session #1: foo!") == "session-1-foo"
    assert _slugify("a" * 50) == "a" * 40


def test_create_state(tmp_path):
    nb = AgentNotebook(tmp_path)
    state_path = tmp_path / ".agent-notes" / "STATE.md"
    assert state_path.exists()
    content = nb.read_state()
    assert "Objective" in content


def test_write_read_state(tmp_path):
    nb = AgentNotebook(tmp_path)
    nb.write_state("# STATE\n\n## Objective\nTest task\n")
    assert "Test task" in nb.read_state()


def test_update_state_section(tmp_path):
    nb = AgentNotebook(tmp_path)
    nb.update_state_section("Objective", "New objective text")
    content = nb.read_state()
    assert "New objective text" in content


def test_update_state_section_new(tmp_path):
    nb = AgentNotebook(tmp_path)
    nb.update_state_section("Custom", "custom value")
    content = nb.read_state()
    assert "custom value" in content


def test_append_note(tmp_path):
    nb = AgentNotebook(tmp_path)
    path = nb.append_note("line1\n", slug="test")
    assert path.exists()
    assert "line1" in path.read_text(encoding="utf-8")


def test_append_note_accumulates(tmp_path):
    nb = AgentNotebook(tmp_path)
    nb.append_note("first\n", slug="test")
    nb.append_note("second\n", slug="test")
    date_str = time.strftime("%Y-%m-%d", time.gmtime())
    note_path = tmp_path / ".agent-notes" / f"{date_str}_test.md"
    content = note_path.read_text(encoding="utf-8")
    assert "first" in content
    assert "second" in content


def test_log_eviction(tmp_path):
    nb = AgentNotebook(tmp_path)
    nb.log_eviction("file:foo.py:0", "advisory_lru", "old summary")
    assert len(nb._session_log) == 1
    assert "advisory_lru" in nb._session_log[0]


def test_log_decision(tmp_path):
    nb = AgentNotebook(tmp_path)
    nb.log_decision("decided to refactor")
    assert len(nb._session_decisions) == 1


def test_record_file_access(tmp_path):
    nb = AgentNotebook(tmp_path)
    nb.record_file_access("foo.py")
    nb.record_file_access("foo.py")
    assert nb._session_stats["foo.py"] == 2


def test_write_session_report(tmp_path):
    nb = AgentNotebook(tmp_path)
    nb.record_file_access("foo.py")
    nb.log_decision("some decision")
    report_path = nb.write_session_report(step_count=5)
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "SESSION REPORT" in content
    assert "foo.py" in content
    assert "some decision" in content
    assert "**Steps**: 5" in content


def test_read_session_report_empty(tmp_path):
    nb = AgentNotebook(tmp_path)
    result = nb.read_session_report()
    assert isinstance(result, str)
