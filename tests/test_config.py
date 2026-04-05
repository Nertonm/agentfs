"""Tests for ctxvault.config."""

import os
import importlib


def test_defaults():
    import ctxvault.config as cfg
    assert cfg.CONTEXT_BUDGET == 4096
    assert cfg.RRF_K == 60
    assert cfg.ZONE_ADVISORY_THRESHOLD == 0.70
    assert cfg.ZONE_CRITICAL_THRESHOLD == 0.85
    assert cfg.THRESHOLD_GRAPH_MIN == 5
    assert cfg.EMBEDDING_DIM == 384


def test_env_override(monkeypatch):
    monkeypatch.setenv("CTX_BUDGET", "2048")
    monkeypatch.setenv("RRF_K", "20")
    # Re-import to pick up env vars
    import ctxvault.config as cfg
    importlib.reload(cfg)
    assert cfg.CONTEXT_BUDGET == 2048
    assert cfg.RRF_K == 20
    # Restore
    importlib.reload(cfg)
