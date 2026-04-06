"""
config.py - Configuration constants for agentfs.

Reads dynamically from config.toml at module load time, falling
back to sensible defaults. Variables are exposed at module scope
for easy importing (e.g. `from config import CONTEXT_BUDGET`).
"""

import os
from pathlib import Path

# Try to use standard library tomllib if available (Python 3.11+)
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

# Global configuration map loaded exactly once
_cfg = {}

def _load_config():
    global _cfg
    config_path = Path(__file__).parent / "config.toml"
    if tomllib and config_path.exists():
        try:
            with open(config_path, "rb") as f:
                _cfg.update(tomllib.load(f))
        except Exception as e:
            print(f"[WARN] ailed to parse config.toml: {e}")
    else:
        if not tomllib:
            print("[WARN] Neither 'tomllib' nor 'tomli' found. Using default configs.")

_load_config()

def _get_nested(keys_path: str, default):
    """Helper to traverse `_cfg` via dot-notation (e.g., 'agent.soft_limit')."""
    keys = keys_path.split(".")
    val = _cfg
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val

# ---------------------------------------------------------------------------
# LLM Context & Server
# ---------------------------------------------------------------------------
LLAMA_SERVER_URL: str = _get_nested("llama.url", "http://localhost:8080/v1")
CONTEXT_BUDGET: int = int(_get_nested("llama.context_tokens", 4096))
LLAMA_TEMPERATURE: float = float(_get_nested("llama.temperature", 0.7))

# Step-wise budget allocation references
SYSTEM_PROMPT_TOKENS: int = 250
TOOLS_TOKENS: int = 150
VAULT_MAP_TOKENS: int = 250
MODEL_RESPONSE_TOKENS: int = int(_get_nested("llama.max_tokens", 800))

# ---------------------------------------------------------------------------
# Agent Memory Limits (Pressure Zones)
# ---------------------------------------------------------------------------
ZONE_ADVISORY_THRESHOLD: float = float(_get_nested("agent.soft_limit", 0.70))
ZONE_CRITICAL_THRESHOLD: float = float(_get_nested("agent.hard_limit", 0.85))
CACHE_MAX_ITEMS: int = int(_get_nested("agent.lru_slots", 8))

# ---------------------------------------------------------------------------
# Retrieval / Indexing
# ---------------------------------------------------------------------------
RR_K: int = int(_get_nested("retriever.rrf_k", 60))
RETRIEVAL_TOP_K: int = int(_get_nested("retriever.top_k", 10))

# Graph cold-start limit: requires >= this many nodes with degree >= 2
THRESHOLD_GRAPH_MIN: int = int(_get_nested("retriever.graph_min_nodes", 5))

CHUNK_MAX_LINES: int = int(_get_nested("indexer.chunk_max_lines", 80))
CHUNK_OVERLAP_LINES: int = int(_get_nested("indexer.chunk_overlap_lines", 5))
WATCHER_DEBOUNCE_S: float = float(_get_nested("indexer.watch_debounce", 1.5))

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = _get_nested("indexer.embedding_model", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Paths and ile names
# ---------------------------------------------------------------------------
NOTEBOOK_DIR: str = ".agent-notes"
STATE_ILE: str = "STATE.md"
SESSION_REPORT_ILE: str = "SESSION_REPORT.md"
INDEX_DIR: str = ".vault-index"
DB_NAME: str = "index.sqlite"
