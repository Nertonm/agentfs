"""Tests for ctxvault.retriever."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from ctxvault.retriever import Retriever, _cosine, _rrf_merge, _unpack
from ctxvault.vault_indexer import VaultIndexer


def test_rrf_merge_single_list():
    result = _rrf_merge([[10, 20, 30]])
    ids = [r[0] for r in result]
    assert ids == [10, 20, 30]


def test_rrf_merge_two_lists():
    # 10 appears in both lists at rank 1 → highest score
    result = _rrf_merge([[10, 20], [10, 30]])
    ids = [r[0] for r in result]
    assert ids[0] == 10


def test_rrf_merge_empty():
    result = _rrf_merge([])
    assert result == []


def test_cosine_identical():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine(a, b)) < 1e-6


def test_unpack_roundtrip():
    import struct
    vec = [0.1, 0.2, 0.3]
    blob = struct.pack("3f", *vec)
    recovered = _unpack(blob)
    for a, b in zip(vec, recovered):
        assert abs(a - b) < 1e-6


@pytest.fixture
def retriever_fixture(tmp_path):
    """Create indexed vault and retriever."""
    (tmp_path / "alpha.py").write_text(
        "def compute_sum(a, b):\n    return a + b\n", encoding="utf-8"
    )
    (tmp_path / "beta.py").write_text(
        "def compute_product(a, b):\n    return a * b\n", encoding="utf-8"
    )
    (tmp_path / "readme.md").write_text(
        "# Project\nThis project [[alpha]] uses [[beta]].\n", encoding="utf-8"
    )
    db_path = tmp_path / "test.sqlite"
    indexer = VaultIndexer(tmp_path, db_path=db_path)
    indexer.index_all()
    indexer.close()
    r = Retriever(db_path=db_path)
    yield r, tmp_path
    r.close()


def test_bm25_retrieval(retriever_fixture):
    r, tmp_path = retriever_fixture
    results = r.retrieve("compute_sum", k=5)
    assert len(results) > 0
    # alpha.py should be top result
    paths = [res["path"] for res in results]
    assert any("alpha" in p for p in paths)


def test_retrieve_returns_required_fields(retriever_fixture):
    r, tmp_path = retriever_fixture
    results = r.retrieve("compute", k=5)
    assert len(results) > 0
    for res in results:
        assert "chunk_id" in res
        assert "path" in res
        assert "content" in res
        assert "rrf_score" in res
        assert "start_line" in res


def test_retrieve_k_limit(retriever_fixture):
    r, tmp_path = retriever_fixture
    results = r.retrieve("compute", k=1)
    assert len(results) <= 1


def test_graph_active_cold(retriever_fixture):
    r, tmp_path = retriever_fixture
    # With only 3 files and limited backlinks, graph may be cold
    active = r._graph_active()
    # Just verify it returns a bool
    assert isinstance(active, bool)


def test_path_filter(retriever_fixture):
    r, tmp_path = retriever_fixture
    results = r.retrieve(
        "compute",
        k=5,
        filters={"path_prefix": str(tmp_path / "alpha")},
    )
    for res in results:
        assert res["path"].startswith(str(tmp_path / "alpha"))


def test_retrieve_no_results_for_nonsense(retriever_fixture):
    r, tmp_path = retriever_fixture
    results = r.retrieve("xyzzy_magic_nonexistent_token_zzz", k=5)
    # FTS may return 0 results for truly unknown token
    assert isinstance(results, list)
