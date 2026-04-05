"""
Configuration constants for agentfs / ctxvault.

All values are intentionally importable at module level so that
any module can do ``from ctxvault.config import CONTEXT_BUDGET``.
Override at runtime by mutating these module-level names before the
rest of the system is initialised (useful for tests).
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# LLM context budget (tokens)
# ---------------------------------------------------------------------------
CONTEXT_BUDGET: int = int(os.environ.get("CTX_BUDGET", 4096))

# Allocation within one step
SYSTEM_PROMPT_TOKENS: int = 250
TOOLS_TOKENS: int = 150
VAULT_MAP_TOKENS: int = 200
HISTORY_TOKENS_PER_STEP: int = 30
MODEL_RESPONSE_TOKENS: int = 500

# ---------------------------------------------------------------------------
# Memory-pressure thresholds (fraction of CONTEXT_BUDGET)
# ---------------------------------------------------------------------------
ZONE_ADVISORY_THRESHOLD: float = 0.70   # start evicting low-priority LRU
ZONE_CRITICAL_THRESHOLD: float = 0.85   # compress + flush to notebook

# ---------------------------------------------------------------------------
# Retrieval / RRF
# ---------------------------------------------------------------------------
RRF_K: int = int(os.environ.get("RRF_K", 60))   # test k∈{20,40,60}
RETRIEVAL_TOP_K: int = 10                        # candidates to return

# ---------------------------------------------------------------------------
# Graph expansion
# ---------------------------------------------------------------------------
# Disable graph expansion when fewer than this many nodes have degree ≥ 2.
THRESHOLD_GRAPH_MIN: int = 5

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_MAX_LINES: int = 80       # hard cap per chunk (fallback regex splitter)
CHUNK_OVERLAP_LINES: int = 5    # lines of overlap between adjacent chunks

# ---------------------------------------------------------------------------
# File-system watcher
# ---------------------------------------------------------------------------
WATCHER_DEBOUNCE_S: float = 1.5   # seconds

# ---------------------------------------------------------------------------
# Cache / eviction
# ---------------------------------------------------------------------------
CACHE_MAX_ITEMS: int = 200        # soft limit; eviction starts here
SUMMARY_MAX_LINES: int = 30       # per-file summary length cap

# ---------------------------------------------------------------------------
# Notebook
# ---------------------------------------------------------------------------
NOTEBOOK_DIR: str = os.environ.get("NOTEBOOK_DIR", ".agent-notes")
STATE_FILE: str = "STATE.md"
SESSION_REPORT_FILE: str = "SESSION_REPORT.md"

# ---------------------------------------------------------------------------
# LLM server (llama.cpp compatible)
# ---------------------------------------------------------------------------
LLAMA_SERVER_URL: str = os.environ.get(
    "LLAMA_SERVER_URL", "http://localhost:8080"
)
LLAMA_COMPLETIONS_PATH: str = "/completion"
LLAMA_TEMPERATURE: float = float(os.environ.get("LLAMA_TEMP", 0.7))
LLAMA_MAX_TOKENS: int = MODEL_RESPONSE_TOKENS

# ---------------------------------------------------------------------------
# Embeddings (optional — sentence-transformers)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.environ.get(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)
EMBEDDING_DIM: int = 384

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH: str = os.environ.get("DB_PATH", "ctxvault/index.sqlite")

# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
REREAD_WARN_THRESHOLD: int = 3    # warn if same file read ≥N times in 2 steps
