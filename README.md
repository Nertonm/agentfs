# agentfs — Local-first AI Agent with Persistent External Memory

A local-first AI agent with persistent external memory, dynamic vault navigation,
and context paging for small-context LLMs (2k–8k tokens).  
Target hardware: 8 GB VRAM GPU (AMD/Vulkan) via **llama.cpp**.

---

## Architecture

The agent treats the LLM context window as RAM and external storage as virtual
memory — inspired by MemGPT's OS metaphor.

### Memory layers

| Layer | Contents | Notes |
|---|---|---|
| **Working context** (always in prompt) | `VAULT_MAP.md`, `STATE.md` | ≤ 200 + 200 tokens |
| **Context cache** (outside LLM, fast) | Per-file summaries (10–30 lines each) | LRU + importance eviction |
| **Long-term memory** (SQLite + Markdown) | `index.sqlite`: chunks, backlinks, FTS5 | Hybrid BM25 + cosine + graph |

### Memory pressure zones

| Zone | Threshold | Action |
|---|---|---|
| Normal | < 70% | No action |
| Advisory | 70–85% | Evict low-priority LRU chunks |
| Critical | > 85% | Compress oldest history half + flush to notebook |

### Retrieval (RRF fusion)

```
RRF_score(d) = Sum 1/(k + rank_i(d)),  k=60  (configurable via RRF_K)
```

Three ranked lists merged:
1. **BM25** — FTS5 full-text search (always available, stdlib only)
2. **Semantic** — cosine similarity on `all-MiniLM-L6-v2` 384-d embeddings (optional)
3. **Graph expansion** — wikilink/backlink 1-hop neighbours (disabled when graph < 5 active nodes)

---

## File structure

```
ctxvault/
├── agent.py            # Main agent loop (ReAct: reason → JSON tool call)
├── multi_agent.py      # Planner / Executor split
├── vault_indexer.py    # Scanner, chunker, BM25 + semantic + graph indexer
├── retriever.py        # RRF fusion retriever
├── context_manager.py  # Token budget, eviction zones, compression
├── agent_notebook.py   # Markdown notes, STATE.md management
├── tools.py            # All 10 tool implementations
├── config.py           # RRF_K, CONTEXT_BUDGET, thresholds, etc.
├── VAULT_MAP.md        # Auto-generated directory tree + backlink graph
└── .agent-notes/
    ├── STATE.md        # Current objective / plan / scratchpad
    └── SESSION_REPORT.md   # Auto-generated session summary

tests/
├── test_config.py
├── test_context_manager.py
├── test_vault_indexer.py
├── test_retriever.py
├── test_agent_notebook.py
├── test_tools.py
└── test_agent.py
```

---

## Quick start

### 1. Install dependencies

```bash
# Minimal (stdlib only — BM25 + graph search)
pip install .

# Full (adds semantic search)
pip install ".[semantic]"
```

### 2. Start llama.cpp server

```bash
# AMD/Vulkan example
./server -m model.gguf --host 0.0.0.0 --port 8080 -ngl 99 -GGML_VULKAN=ON
```

### 3. Run the agent

```python
from ctxvault.agent import create_agent

agent = create_agent(vault_root=".")
result = agent.run("Summarise the project and list all public functions.")
print(result)
```

Or the multi-agent (Planner + Executor):

```python
from ctxvault.multi_agent import MultiAgent

ma = MultiAgent.create(vault_root=".")
result = ma.run("Refactor the config module to add a new threshold.")
print(result)
```

### 4. Run from CLI

```bash
python -m ctxvault.agent "Describe the project"
python -m ctxvault.multi_agent "List all TODO comments"
```

---

## Tool set

| Tool | Signature | Description |
|---|---|---|
| `list_dir` | `(path, depth, filters)` | Directory listing |
| `search_text` | `(query, paths, regex)` | Text/regex search across files |
| `read_file` | `(path, start_line, end_line)` | Paginated file reading |
| `read_symbols` | `(path)` | Extract functions/classes (regex; tree-sitter upgrade path) |
| `write_file` | `(path, content)` | Write a file |
| `append_file` | `(path, content)` | Append to agent notebook or any file |
| `summarize_to_cache` | `(item_id)` | Generate a short summary and store in context cache |
| `retrieve_candidates` | `(query, k, filters)` | Hybrid BM25 + semantic + graph retrieval |
| `pin` / `unpin` | `(item_id)` | Pin/unpin cache items (pinned items are never evicted) |
| `run_command` | `(cmd)` | Optional sandboxed shell command |

All tools return compact JSON: `{"ok": bool, "output": str, ...pagination...}`

---

## Configuration

All defaults are in `ctxvault/config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `CTX_BUDGET` | `4096` | Total token budget per step |
| `RRF_K` | `60` | RRF fusion k parameter (try 20, 40, 60) |
| `LLAMA_SERVER_URL` | `http://localhost:8080` | llama.cpp server URL |
| `LLAMA_TEMP` | `0.7` | Sampling temperature |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `DB_PATH` | `ctxvault/index.sqlite` | SQLite database path |
| `NOTEBOOK_DIR` | `.agent-notes` | Agent notebook directory |

---

## Observability

Every turn emits a status line:

```
[CTX: 68% | ZONE: normal | GRAPH: active | STEPS: 4]
```

- Every eviction is logged to `.agent-notes/YYYY-MM-DD_eviction-log.md`
- Re-read warning if `read_file` called 3 or more times on same file in 60 s
- Session report auto-generated on exit: stats, most-accessed files, decisions

---

## Stack

- **Runtime**: llama.cpp server (Vulkan backend)
- **DB**: SQLite only — no external vector server
- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers (22 MB, CPU, optional)
- **Watcher**: polling-based watcher with 1.5 s debounce for incremental reindex
- **Chunking**: function/class boundaries (regex fallback; tree-sitter upgrade path)
- **Python**: stdlib-first; optional deps: `numpy`, `sentence-transformers`

---

## Development

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Key constraints

- Works 100% offline — no external API calls
- No mandatory Docker or external services
- All data in SQLite and Markdown — fully auditable
- Graceful degradation: works with stdlib only (BM25 + graph), improves with optional deps
- Context shift prevention: compress at 85% usage, not 95-100%
