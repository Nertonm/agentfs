"""
retriever.py — Reciprocal Rank Fusion of BM25 + semantic + graph neighbours.

Retrieval pipeline
------------------
1. **BM25** — full-text search via SQLite FTS5 (always available).
2. **Semantic** — cosine similarity against stored float32 BLOB embeddings
   (requires sentence-transformers + numpy).
3. **Graph expansion** — follow wikilink/backlink neighbours of top-ranked
   chunks (disabled when graph has < THRESHOLD_GRAPH_MIN nodes with degree ≥ 2).
4. **RRF fusion** — merge ranked lists with
   RRF_score(d) = Σ 1/(k + rank_i(d)), k = RRF_K.

Graceful degradation
--------------------
* stdlib only → BM25 + graph.
* numpy available → cosine similarity.
* sentence-transformers available → query embedding.
"""

from __future__ import annotations

import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ctxvault.config import (
    DB_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
    RRF_K,
    THRESHOLD_GRAPH_MIN,
)

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    import numpy as np  # type: ignore

    _HAS_NP = True
except ImportError:
    _HAS_NP = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _HAS_ST = True
except ImportError:
    _HAS_ST = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpack(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine(a: List[float], b: List[float]) -> float:
    if not _HAS_NP:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb + 1e-9)
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


def _rrf_merge(
    ranked_lists: List[List[int]],
    k: int = RRF_K,
) -> List[Tuple[int, float]]:
    """Merge ranked lists via RRF. Returns (chunk_id, score) sorted desc."""
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """Hybrid BM25 + semantic + graph retriever with RRF fusion."""

    def __init__(self, db_path: str | Path = DB_PATH) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._st_model: Optional[Any] = None
        if _HAS_ST:
            try:
                self._st_model = SentenceTransformer(EMBEDDING_MODEL)
            except Exception:
                pass

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = RETRIEVAL_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-k chunk dicts with metadata."""
        bm25_ids = self._bm25(query, k * 3)
        sem_ids = self._semantic(query, k * 3)

        use_graph = self._graph_active()
        graph_ids: List[int] = []
        if use_graph:
            seed_ids = bm25_ids[:5] + sem_ids[:5]
            graph_ids = self._graph_expand(seed_ids, k * 2)

        ranked_lists = [l for l in [bm25_ids, sem_ids, graph_ids] if l]
        merged = _rrf_merge(ranked_lists)[:k]

        results = []
        for chunk_id, score in merged:
            row = self._conn.execute(
                "SELECT c.id, c.file_id, c.chunk_index, c.start_line, "
                "c.end_line, c.content, c.summary, f.path "
                "FROM chunks c JOIN files f ON c.file_id=f.id "
                "WHERE c.id=?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                continue
            chunk = {
                "chunk_id": row[0],
                "file_id": row[1],
                "chunk_index": row[2],
                "start_line": row[3],
                "end_line": row[4],
                "content": row[5],
                "summary": row[6],
                "path": row[7],
                "rrf_score": round(score, 6),
            }
            if filters:
                if not self._matches_filters(chunk, filters):
                    continue
            results.append(chunk)
        return results

    # ------------------------------------------------------------------
    # BM25 via FTS5
    # ------------------------------------------------------------------

    def _bm25(self, query: str, limit: int) -> List[int]:
        try:
            rows = self._conn.execute(
                "SELECT chunk_id FROM fts_index "
                "WHERE content MATCH ? "
                "ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
            return [r[0] for r in rows]
        except sqlite3.OperationalError:
            return []

    # ------------------------------------------------------------------
    # Semantic (cosine) search
    # ------------------------------------------------------------------

    def _semantic(self, query: str, limit: int) -> List[int]:
        if self._st_model is None:
            return []
        try:
            q_vec: List[float] = self._st_model.encode(
                [query], show_progress_bar=False
            )[0].tolist()
        except Exception:
            return []

        rows = self._conn.execute(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return []

        scored: List[Tuple[float, int]] = []
        for chunk_id, blob in rows:
            vec = _unpack(blob)
            sim = _cosine(q_vec, vec)
            scored.append((sim, chunk_id))
        scored.sort(reverse=True)
        return [cid for _, cid in scored[:limit]]

    # ------------------------------------------------------------------
    # Graph expansion
    # ------------------------------------------------------------------

    def _graph_active(self) -> bool:
        """Return True if graph has ≥ THRESHOLD_GRAPH_MIN nodes with degree ≥ 2."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT source_file FROM backlinks "
            "  GROUP BY source_file HAVING COUNT(*)>=2 "
            "  UNION "
            "  SELECT target_file FROM backlinks "
            "  GROUP BY target_file HAVING COUNT(*)>=2"
            ")"
        ).fetchone()
        return (row[0] if row else 0) >= THRESHOLD_GRAPH_MIN

    def _graph_expand(self, seed_chunk_ids: List[int], limit: int) -> List[int]:
        """Return chunk IDs reachable via wikilink graph from seed files."""
        if not seed_chunk_ids:
            return []
        # Get seed file paths
        placeholders = ",".join("?" * len(seed_chunk_ids))
        file_rows = self._conn.execute(
            f"SELECT DISTINCT f.path FROM chunks c "
            f"JOIN files f ON c.file_id=f.id "
            f"WHERE c.id IN ({placeholders})",
            seed_chunk_ids,
        ).fetchall()
        seed_files = [r[0] for r in file_rows]
        if not seed_files:
            return []

        # Neighbours via backlinks (1-hop)
        fp = ",".join("?" * len(seed_files))
        neighbour_rows = self._conn.execute(
            f"SELECT DISTINCT target_file FROM backlinks "
            f"WHERE source_file IN ({fp}) "
            f"UNION "
            f"SELECT DISTINCT source_file FROM backlinks "
            f"WHERE target_file IN ({fp})",
            seed_files + seed_files,
        ).fetchall()
        neighbour_files = [r[0] for r in neighbour_rows]
        if not neighbour_files:
            return []

        # Fetch chunk IDs for neighbour files
        nfp = ",".join("?" * len(neighbour_files))
        chunk_rows = self._conn.execute(
            f"SELECT c.id FROM chunks c JOIN files f ON c.file_id=f.id "
            f"WHERE f.path IN ({nfp}) ORDER BY c.importance DESC LIMIT ?",
            neighbour_files + [limit],
        ).fetchall()
        return [r[0] for r in chunk_rows]

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_filters(chunk: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        if "path_prefix" in filters:
            if not chunk["path"].startswith(filters["path_prefix"]):
                return False
        if "min_importance" in filters:
            pass  # importance check would require fetching it
        return True
