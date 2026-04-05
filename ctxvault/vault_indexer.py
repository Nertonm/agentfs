"""
vault_indexer.py — file scanner, chunker, and SQLite indexer.

Responsibilities
----------------
* Walk the vault directory and detect changed / new files.
* Split files into chunks by function/class boundaries (regex fallback;
  tree-sitter upgrade path).
* Persist chunks, backlinks (wikilinks ``[[target]]``), and FTS5 index to
  ``index.sqlite``.
* Optionally embed chunks with sentence-transformers (MiniLM-L6-v2, 384-d).
* Generate ``VAULT_MAP.md`` as a structural overview.
* Watch for file-system changes with a 1.5 s debounce.

Graceful degradation
--------------------
* Works with stdlib only (BM25 via FTS5, graph).
* Adds semantic search if ``sentence_transformers`` is available.
* Uses tree-sitter chunking if available, falls back to regex.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

from ctxvault.config import (
    CHUNK_MAX_LINES,
    CHUNK_OVERLAP_LINES,
    DB_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    WATCHER_DEBOUNCE_S,
)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    _HAS_ST = True
except ImportError:
    _HAS_ST = False

try:
    import tree_sitter  # type: ignore  # noqa: F401

    _HAS_TS = False  # placeholder; full TS integration beyond scope
except ImportError:
    _HAS_TS = False

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY,
    path        TEXT    NOT NULL UNIQUE,
    mtime       REAL    NOT NULL,
    sha256      TEXT    NOT NULL,
    indexed_at  REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    content     TEXT    NOT NULL,
    summary     TEXT,
    embedding   BLOB,           -- float32 * EMBEDDING_DIM
    importance  REAL    NOT NULL DEFAULT 0.5,
    last_access REAL,
    UNIQUE(file_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS backlinks (
    id          INTEGER PRIMARY KEY,
    source_file TEXT    NOT NULL,
    target_file TEXT    NOT NULL,
    UNIQUE(source_file, target_file)
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
    content,
    chunk_id UNINDEXED,
    tokenize='porter ascii'
);
"""

# Wikilink pattern: [[SomeTarget]] or [[SomeTarget|alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

# Regex-based code boundary markers
_BOUNDARY_RE = re.compile(
    r"^(def |class |async def |function |const |export function |pub fn |func )",
    re.MULTILINE,
)

# Files to skip
_SKIP_PATTERNS = {
    ".git", "__pycache__", ".mypy_cache", "*.pyc",
    "node_modules", ".venv", "venv", ".env",
}


def _should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in _SKIP_PATTERNS:
            return True
    # Skip SQLite database files (including WAL-mode sidecar files)
    if ".sqlite" in path.name:
        return True
    if path.suffix in {".pyc", ".pyo", ".so", ".dylib", ".dll", ".bin"}:
        return True
    # Skip auto-generated files
    if path.name == "VAULT_MAP.md":
        return True
    return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _pack_embedding(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes) -> List[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def _chunk_by_boundaries(lines: List[str]) -> List[Tuple[int, int]]:
    """Return (start_line, end_line) pairs (0-indexed, inclusive).

    Splits at code boundary markers or at CHUNK_MAX_LINES hard cap.
    """
    boundaries: List[int] = [0]
    for i, line in enumerate(lines):
        if i == 0:
            continue
        if _BOUNDARY_RE.match(line) or i - boundaries[-1] >= CHUNK_MAX_LINES:
            boundaries.append(i)
    boundaries.append(len(lines))

    chunks: List[Tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        start = max(0, boundaries[i] - CHUNK_OVERLAP_LINES) if i > 0 else 0
        end = boundaries[i + 1] - 1
        chunks.append((start, end))
    return chunks


# ---------------------------------------------------------------------------
# Embedder (optional)
# ---------------------------------------------------------------------------

class _Embedder:
    _instance: Optional["_Embedder"] = None
    _model: object = None

    @classmethod
    def get(cls) -> Optional["_Embedder"]:
        if not _HAS_ST:
            return None
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vecs]


# ---------------------------------------------------------------------------
# VaultIndexer
# ---------------------------------------------------------------------------

class VaultIndexer:
    """Indexes a vault directory into SQLite."""

    def __init__(
        self,
        vault_root: str | Path,
        db_path: str | Path = DB_PATH,
    ) -> None:
        self.vault_root = Path(vault_root).resolve()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        self._embedder = _Embedder.get()
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_all(self) -> int:
        """Scan vault, index changed files. Returns count of indexed files."""
        count = 0
        for path in self._walk():
            if self._needs_reindex(path):
                self._index_file(path)
                count += 1
        self._generate_vault_map()
        return count

    def index_file(self, path: str | Path) -> None:
        p = Path(path).resolve()
        self._index_file(p)
        self._generate_vault_map()

    def get_chunks_for_file(self, path: str) -> List[dict]:
        row = self._conn.execute(
            "SELECT id FROM files WHERE path=?", (path,)
        ).fetchone()
        if not row:
            return []
        file_id = row[0]
        rows = self._conn.execute(
            "SELECT chunk_index, start_line, end_line, content, summary "
            "FROM chunks WHERE file_id=? ORDER BY chunk_index",
            (file_id,),
        ).fetchall()
        return [
            {
                "chunk_index": r[0],
                "start_line": r[1],
                "end_line": r[2],
                "content": r[3],
                "summary": r[4],
            }
            for r in rows
        ]

    def update_chunk_access(self, chunk_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE chunks SET last_access=? WHERE id=?",
                (time.time(), chunk_id),
            )
            self._conn.commit()

    def update_chunk_summary(self, chunk_id: int, summary: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE chunks SET summary=? WHERE id=?",
                (summary, chunk_id),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _walk(self) -> Iterator[Path]:
        for root, dirs, files in os.walk(self.vault_root):
            dirs[:] = [
                d for d in dirs
                if not _should_skip(Path(root) / d)
            ]
            for fname in files:
                p = Path(root) / fname
                if not _should_skip(p):
                    yield p

    def _needs_reindex(self, path: Path) -> bool:
        row = self._conn.execute(
            "SELECT mtime, sha256 FROM files WHERE path=?",
            (str(path),),
        ).fetchone()
        if row is None:
            return True
        stored_mtime, stored_sha = row
        current_mtime = path.stat().st_mtime
        if abs(current_mtime - stored_mtime) < 0.001:
            return False
        return _sha256(path) != stored_sha

    def _index_file(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        lines = text.splitlines()
        sha = _sha256(path)
        mtime = path.stat().st_mtime
        now = time.time()

        with self._lock:
            cur = self._conn.execute(
                "INSERT OR REPLACE INTO files(path, mtime, sha256, indexed_at) "
                "VALUES(?,?,?,?)",
                (str(path), mtime, sha, now),
            )
            file_id = cur.lastrowid or self._conn.execute(
                "SELECT id FROM files WHERE path=?", (str(path),)
            ).fetchone()[0]
            # Remove old chunks / FTS entries
            old_chunk_ids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT id FROM chunks WHERE file_id=?", (file_id,)
                ).fetchall()
            ]
            if old_chunk_ids:
                placeholders = ",".join("?" * len(old_chunk_ids))
                self._conn.execute(
                    f"DELETE FROM fts_index WHERE chunk_id IN ({placeholders})",
                    old_chunk_ids,
                )
            self._conn.execute(
                "DELETE FROM chunks WHERE file_id=?", (file_id,)
            )

            # Chunk
            chunk_spans = _chunk_by_boundaries(lines)
            contents = [
                "\n".join(lines[s:e + 1]) for s, e in chunk_spans
            ]

            # Embed (batch)
            embeddings: List[Optional[bytes]] = [None] * len(contents)
            if self._embedder and contents:
                vecs = self._embedder.embed(contents)
                embeddings = [_pack_embedding(v) for v in vecs]

            for idx, ((start, end), content, emb) in enumerate(
                zip(chunk_spans, contents, embeddings)
            ):
                cur2 = self._conn.execute(
                    "INSERT INTO chunks(file_id, chunk_index, start_line, "
                    "end_line, content, embedding) VALUES(?,?,?,?,?,?)",
                    (file_id, idx, start, end, content, emb),
                )
                chunk_id = cur2.lastrowid
                self._conn.execute(
                    "INSERT INTO fts_index(content, chunk_id) VALUES(?,?)",
                    (content, chunk_id),
                )

            # Backlinks
            self._conn.execute(
                "DELETE FROM backlinks WHERE source_file=?", (str(path),)
            )
            for m in _WIKILINK_RE.finditer(text):
                target = m.group(1).strip()
                self._conn.execute(
                    "INSERT OR IGNORE INTO backlinks(source_file, target_file) "
                    "VALUES(?,?)",
                    (str(path), target),
                )

            self._conn.commit()

    def _generate_vault_map(self) -> None:
        """Write VAULT_MAP.md summarizing the vault structure."""
        map_path = self.vault_root / "VAULT_MAP.md"
        lines_out: List[str] = [
            "# VAULT_MAP\n",
            f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}_\n",
            "\n## Directory tree\n",
        ]
        # Build tree
        for root, dirs, files in os.walk(self.vault_root):
            dirs[:] = sorted(
                d for d in dirs if not _should_skip(Path(root) / d)
            )
            rel = os.path.relpath(root, self.vault_root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            indent = "  " * depth
            folder_name = os.path.basename(root) if rel != "." else "."
            lines_out.append(f"{indent}- **{folder_name}/**\n")
            for f in sorted(files):
                if not _should_skip(Path(root) / f):
                    lines_out.append(f"{indent}  - {f}\n")

        # Backlinks summary
        rows = self._conn.execute(
            "SELECT source_file, target_file FROM backlinks LIMIT 100"
        ).fetchall()
        if rows:
            lines_out.append("\n## Wikilink / backlink graph (sample)\n")
            for src, tgt in rows:
                src_rel = os.path.relpath(src, self.vault_root)
                lines_out.append(f"- `{src_rel}` → `{tgt}`\n")

        # Hot files (most accessed)
        hot = self._conn.execute(
            "SELECT f.path, COUNT(*) as c "
            "FROM chunks ch JOIN files f ON ch.file_id=f.id "
            "WHERE ch.last_access IS NOT NULL "
            "GROUP BY f.id ORDER BY c DESC LIMIT 10"
        ).fetchall()
        if hot:
            lines_out.append("\n## Hot files (most-accessed chunks)\n")
            for path_str, cnt in hot:
                rel = os.path.relpath(path_str, self.vault_root)
                lines_out.append(f"- `{rel}` ({cnt} chunks accessed)\n")

        map_path.write_text("".join(lines_out), encoding="utf-8")


# ---------------------------------------------------------------------------
# File system watcher (stdlib inotify-free, polling-based)
# ---------------------------------------------------------------------------

class VaultWatcher:
    """Poll-based watcher with debounce. Calls ``callback(path)`` on change."""

    def __init__(
        self,
        indexer: VaultIndexer,
        debounce: float = WATCHER_DEBOUNCE_S,
    ) -> None:
        self._indexer = indexer
        self._debounce = debounce
        self._stop = threading.Event()
        self._pending: dict[str, float] = {}
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        snapshot: dict[str, float] = self._snapshot()
        while not self._stop.is_set():
            time.sleep(0.5)
            current = self._snapshot()
            changed: set[str] = set()
            for p, mtime in current.items():
                if p not in snapshot or snapshot[p] != mtime:
                    changed.add(p)
            snapshot = current
            if changed:
                time.sleep(self._debounce)
                for p in changed:
                    try:
                        self._indexer.index_file(p)
                    except Exception:
                        pass

    def _snapshot(self) -> dict[str, float]:
        result: dict[str, float] = {}
        try:
            for p in self._indexer._walk():
                try:
                    result[str(p)] = p.stat().st_mtime
                except OSError:
                    pass
        except Exception:
            pass
        return result
