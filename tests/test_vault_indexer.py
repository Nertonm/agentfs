"""Tests for ctxvault.vault_indexer."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from ctxvault.vault_indexer import (
    VaultIndexer,
    _chunk_by_boundaries,
    _sha256,
    _should_skip,
)


def test_should_skip_git():
    assert _should_skip(Path(".git/config"))


def test_should_skip_pycache():
    assert _should_skip(Path("src/__pycache__/foo.pyc"))


def test_should_skip_normal():
    assert not _should_skip(Path("ctxvault/config.py"))


def test_sha256(tmp_path):
    f = tmp_path / "file.txt"
    f.write_bytes(b"hello")
    h1 = _sha256(f)
    assert len(h1) == 64
    f.write_bytes(b"world")
    h2 = _sha256(f)
    assert h1 != h2


def test_chunk_by_boundaries_single():
    lines = ["line1", "line2", "line3"]
    chunks = _chunk_by_boundaries(lines)
    assert len(chunks) == 1
    assert chunks[0] == (0, 2)


def test_chunk_by_boundaries_split_on_def():
    lines = ["x = 1", "def foo():", "    pass", "def bar():", "    return 1"]
    chunks = _chunk_by_boundaries(lines)
    assert len(chunks) >= 2


def test_chunk_by_boundaries_hard_cap():
    lines = [f"line{i}" for i in range(200)]
    chunks = _chunk_by_boundaries(lines)
    assert len(chunks) > 1
    for start, end in chunks:
        assert end - start <= 80 + 5  # cap + overlap


@pytest.fixture
def indexed_vault(tmp_path):
    """Create a small vault and index it."""
    (tmp_path / "file_a.py").write_text(
        "def hello():\n    return 'hello'\n\ndef world():\n    return 'world'\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.md").write_text(
        "# Notes\nSee [[file_a]] for more.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "test.sqlite"
    indexer = VaultIndexer(tmp_path, db_path=db_path)
    indexer.index_all()
    return indexer, tmp_path, db_path


def test_index_creates_files_table(indexed_vault):
    indexer, tmp_path, db_path = indexed_vault
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT path FROM files").fetchall()
    paths = [r[0] for r in rows]
    assert any("file_a.py" in p for p in paths)
    assert any("notes.md" in p for p in paths)
    conn.close()
    indexer.close()


def test_index_creates_chunks(indexed_vault):
    indexer, tmp_path, db_path = indexed_vault
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    assert rows[0] > 0
    conn.close()
    indexer.close()


def test_index_creates_backlinks(indexed_vault):
    indexer, tmp_path, db_path = indexed_vault
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT source_file, target_file FROM backlinks").fetchall()
    assert any("file_a" in r[1] for r in rows)
    conn.close()
    indexer.close()


def test_index_creates_fts(indexed_vault):
    indexer, tmp_path, db_path = indexed_vault
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT chunk_id FROM fts_index WHERE content MATCH 'hello'"
    ).fetchall()
    assert len(rows) > 0
    conn.close()
    indexer.close()


def test_vault_map_generated(indexed_vault):
    indexer, tmp_path, db_path = indexed_vault
    map_path = tmp_path / "VAULT_MAP.md"
    assert map_path.exists()
    content = map_path.read_text(encoding="utf-8")
    assert "VAULT_MAP" in content
    assert "file_a.py" in content
    indexer.close()


def test_no_reindex_unchanged(indexed_vault):
    indexer, tmp_path, db_path = indexed_vault
    # Second call — nothing changed
    count = indexer.index_all()
    assert count == 0
    indexer.close()


def test_reindex_on_change(indexed_vault):
    indexer, tmp_path, db_path = indexed_vault
    # Modify a file
    (tmp_path / "file_a.py").write_text(
        "def hello():\n    return 'changed'\n", encoding="utf-8"
    )
    count = indexer.index_all()
    assert count >= 1
    indexer.close()


def test_get_chunks_for_file(indexed_vault):
    indexer, tmp_path, db_path = indexed_vault
    path = str(tmp_path / "file_a.py")
    chunks = indexer.get_chunks_for_file(path)
    assert len(chunks) > 0
    assert "content" in chunks[0]
    indexer.close()
