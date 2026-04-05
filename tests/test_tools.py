"""Tests for ctxvault.tools."""

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from ctxvault.tools import ToolSet


@pytest.fixture
def vault(tmp_path):
    """Create a minimal vault with sample files."""
    (tmp_path / "hello.py").write_text(
        "def greet(name):\n    return f'Hello, {name}'\n\nclass Greeter:\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.md").write_text(
        "# Notes\nSome content here.\nSecond line.\n",
        encoding="utf-8",
    )
    subdir = tmp_path / "sub"
    subdir.mkdir()
    (subdir / "module.py").write_text(
        "def helper():\n    pass\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def tools(vault):
    return ToolSet(vault)


# ------------------------------------------------------------------
# list_dir
# ------------------------------------------------------------------

def test_list_dir_root(tools, vault):
    result = tools.list_dir(".")
    assert result["ok"]
    assert "hello.py" in result["output"]
    assert "notes.md" in result["output"]


def test_list_dir_depth_1(tools, vault):
    result = tools.list_dir(".", depth=1)
    assert result["ok"]


def test_list_dir_nonexistent(tools):
    result = tools.list_dir("nonexistent_xyz")
    assert not result["ok"]


def test_list_dir_filters(tools, vault):
    result = tools.list_dir(".", filters=[r"\.py$"])
    assert result["ok"]
    assert "hello.py" in result["output"]


# ------------------------------------------------------------------
# search_text
# ------------------------------------------------------------------

def test_search_text_literal(tools):
    result = tools.search_text("greet")
    assert result["ok"]
    assert "hello.py" in result["output"]


def test_search_text_regex(tools):
    result = tools.search_text(r"def \w+", regex=True)
    assert result["ok"]
    assert result["count"] > 0


def test_search_text_no_match(tools):
    result = tools.search_text("xyzzy_no_match_token")
    assert result["ok"]
    assert result["count"] == 0


def test_search_text_bad_regex(tools):
    result = tools.search_text("[invalid", regex=True)
    assert not result["ok"]


# ------------------------------------------------------------------
# read_file
# ------------------------------------------------------------------

def test_read_file_full(tools):
    result = tools.read_file("hello.py")
    assert result["ok"]
    assert "def greet" in result["output"]


def test_read_file_range(tools):
    result = tools.read_file("hello.py", start_line=0, end_line=0)
    assert result["ok"]
    assert "def greet" in result["output"]
    assert "class Greeter" not in result["output"]


def test_read_file_not_found(tools):
    result = tools.read_file("missing.py")
    assert not result["ok"]


def test_read_file_has_more(tools, vault):
    # Write a long file to trigger has_more
    long_file = vault / "long.py"
    long_file.write_text("\n".join(f"line{i}" for i in range(200)), encoding="utf-8")
    result = tools.read_file("long.py", start_line=0, end_line=50)
    assert result["ok"]
    assert result["has_more"] is True


# ------------------------------------------------------------------
# read_symbols
# ------------------------------------------------------------------

def test_read_symbols(tools):
    result = tools.read_symbols("hello.py")
    assert result["ok"]
    assert "greet" in result["output"]
    assert "Greeter" in result["output"]
    assert result["count"] == 2


def test_read_symbols_no_symbols(tools, vault):
    (vault / "plain.txt").write_text("just text\n", encoding="utf-8")
    result = tools.read_symbols("plain.txt")
    assert result["ok"]
    assert result["count"] == 0


# ------------------------------------------------------------------
# write_file / append_file
# ------------------------------------------------------------------

def test_write_file(tools, vault):
    result = tools.write_file("output.txt", "hello world")
    assert result["ok"]
    assert (vault / "output.txt").read_text(encoding="utf-8") == "hello world"


def test_write_file_creates_parents(tools, vault):
    result = tools.write_file("nested/dir/file.txt", "content")
    assert result["ok"]
    assert (vault / "nested" / "dir" / "file.txt").exists()


def test_append_file(tools, vault):
    tools.write_file("log.txt", "first\n")
    tools.append_file("log.txt", "second\n")
    content = (vault / "log.txt").read_text(encoding="utf-8")
    assert "first" in content
    assert "second" in content


# ------------------------------------------------------------------
# summarize_to_cache
# ------------------------------------------------------------------

def test_summarize_to_cache(tools):
    result = tools.summarize_to_cache("hello.py")
    assert result["ok"]
    assert result["item_id"] == "hello.py"
    assert len(result["output"]) > 0


def test_summarize_missing(tools):
    result = tools.summarize_to_cache("missing.py")
    assert not result["ok"]


# ------------------------------------------------------------------
# retrieve_candidates (without retriever)
# ------------------------------------------------------------------

def test_retrieve_no_retriever(tools):
    result = tools.retrieve_candidates("query")
    assert not result["ok"]


# ------------------------------------------------------------------
# pin / unpin (without context_manager)
# ------------------------------------------------------------------

def test_pin_no_cm(tools):
    result = tools.pin("item")
    assert not result["ok"]


def test_unpin_no_cm(tools):
    result = tools.unpin("item")
    assert not result["ok"]


# ------------------------------------------------------------------
# run_command
# ------------------------------------------------------------------

def test_run_command_disabled(tools):
    result = tools.run_command("echo hello")
    assert not result["ok"]
    assert "disabled" in result["output"]


def test_run_command_enabled(vault):
    ts = ToolSet(vault, allow_run_command=True)
    result = ts.run_command("echo hello")
    assert result["ok"]
    assert "hello" in result["output"]


def test_run_command_failure(vault):
    ts = ToolSet(vault, allow_run_command=True)
    result = ts.run_command("exit 1")
    assert not result["ok"]


# ------------------------------------------------------------------
# dispatch
# ------------------------------------------------------------------

def test_dispatch_known_tool(tools):
    result = tools.dispatch("list_dir", {"path": ".", "depth": 1})
    assert result["ok"]


def test_dispatch_unknown_tool(tools):
    result = tools.dispatch("nonexistent_tool", {})
    assert not result["ok"]
    assert "Unknown tool" in result["output"]


def test_dispatch_bad_args(tools):
    result = tools.dispatch("read_file", {"bad_arg": "x"})
    assert not result["ok"]
