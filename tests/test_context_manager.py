"""Tests for ctxvault.context_manager."""

import time
import pytest

from ctxvault.context_manager import ContextManager, _approx_tokens


def test_approx_tokens():
    assert _approx_tokens("") == 1
    assert _approx_tokens("hello world") == 2  # 11 // 4 = 2
    assert _approx_tokens("a" * 400) == 100


def test_add_and_used():
    cm = ContextManager(budget=1000)
    cm.add("item1", "hello world", tokens=10)
    assert cm.used == 10
    cm.add("item2", "foo bar baz", tokens=5)
    assert cm.used == 15


def test_remove():
    cm = ContextManager(budget=1000)
    cm.add("item1", "content", tokens=20)
    assert cm.used == 20
    content = cm.remove("item1")
    assert content == "content"
    assert cm.used == 0


def test_fraction_used():
    cm = ContextManager(budget=1000)
    cm.add("item1", "x", tokens=700)
    assert abs(cm.fraction_used() - 0.7) < 0.001


def test_zone_normal():
    cm = ContextManager(budget=1000)
    cm.add("x", "y", tokens=600)
    assert cm.zone() == "normal"


def test_zone_advisory():
    cm = ContextManager(budget=1000)
    cm.add("x", "y", tokens=750)
    assert cm.zone() == "advisory"


def test_zone_critical():
    cm = ContextManager(budget=1000)
    cm.add("x", "y", tokens=900)
    assert cm.zone() == "critical"


def test_eviction_advisory():
    cm = ContextManager(budget=1000)
    for i in range(5):
        cm.add(f"item{i}", "x" * 40, tokens=160, importance=0.5)
    # 5 * 160 = 800 → advisory
    assert cm.zone() == "advisory"
    evicted = cm.maybe_evict()
    assert len(evicted) > 0
    assert cm.zone() in ("normal", "advisory")


def test_pin_prevents_eviction():
    cm = ContextManager(budget=1000)
    cm.add("important", "x" * 40, tokens=500, importance=0.1)
    cm.pin("important")
    cm.add("filler", "y" * 40, tokens=400, importance=0.1)
    # 900 tokens → advisory
    evicted = cm.maybe_evict()
    assert "important" not in evicted


def test_update_item():
    cm = ContextManager(budget=1000)
    cm.add("item1", "short", tokens=5)
    cm.add("item1", "longer content", tokens=15)
    assert cm.used == 15


def test_status_line():
    cm = ContextManager(budget=1000)
    cm.add("x", "y", tokens=680)
    line = cm.status_line(graph_active=True, step_count=3)
    assert "CTX:" in line
    assert "ZONE:" in line
    assert "GRAPH: active" in line
    assert "STEPS: 3" in line


def test_compress_history():
    cm = ContextManager(budget=1000)
    history = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
    kept, flushed = cm.compress_history(history)
    assert len(flushed) == 5
    assert len(kept) == 5
    assert flushed[0]["content"] == "msg0"
    assert kept[0]["content"] == "msg5"


def test_compress_short_history():
    cm = ContextManager(budget=1000)
    history = [{"role": "user", "content": "only"}]
    kept, flushed = cm.compress_history(history)
    assert kept == history
    assert flushed == []


def test_eviction_callback():
    evicted_ids = []
    cm = ContextManager(
        budget=1000,
        on_evict=lambda item_id, reason: evicted_ids.append(item_id),
    )
    cm.add("low", "x" * 40, tokens=500, importance=0.1)
    cm.add("medium", "y" * 40, tokens=400, importance=0.5)
    cm.maybe_evict()
    assert len(evicted_ids) > 0


def test_get_returns_content():
    cm = ContextManager(budget=1000)
    cm.add("key", "my content", tokens=10)
    assert cm.get("key") == "my content"
    assert cm.get("missing") is None
