"""
ctxvault — local-first AI agent with persistent external memory.

Modules
-------
config          Configuration constants
vault_indexer   File scanner, chunker, and SQLite indexer
retriever       RRF fusion retriever (BM25 + semantic + graph)
context_manager Token budget, eviction zones, compression
agent_notebook  Markdown notebook and STATE.md
tools           All 10 tool implementations
agent           Main ReAct agent loop
multi_agent     Planner / Executor split
"""
