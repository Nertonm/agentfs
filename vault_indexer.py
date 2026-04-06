#!/usr/bin/env python3
"""
vault_indexer_v2.py - Indexador + Retrieval Híbrido (BM25 + Semântica + Grafo → RR)

uncionalidades:
  • Scan incremental do workspace (SQLite, sem deps externas)
  • Chunking semântico por função/classe (tree-sitter opcional, fFallback regex)
  • Embeddings via sentence-transformers (CPU) ou llama-server /v1/embeddings
  • Busca híbrida: BM25 (TS5) + cosine similarity (numpy) + grafo (wikilinks/backlinks)
  • usão Reciprocal Rank usion (RR)
  • Geração de VAULT_MAP.md compacto (sempre no contexto do agente)
  • Reindexação incremental de arquivo único (reindex_file)

Dependências:
  obrigatórias : stdlib (sqlite3, pathlib, json, re, hashlib, struct)
  opcionais    : numpy, sentence-transformers, sqlite-vec, requests (llama-server emb)
"""

import sqlite3, json, hashlib, re, time, struct
from pathlib import Path
from datetime import datetime
from typing import Optional
import requests

# 
def _try_ts_boundaries(content: str, ext: str):
    try:
        from vault_summarizer import ts_boundaries
        return ts_boundaries(content, ext)
    except ImportError:
        return None

# 
try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False

try:
    from sentence_transformers import SentenceTransformer as _ST
    _ST_AVAIL = True
except ImportError:
    _ST_AVAIL = False

try:
    import sqlite_vec as _svec
    _VEC_AVAIL = True
except ImportError:
    _VEC_AVAIL = False

# 
IGNORED_DIRS   = {".git","node_modules","__pycache__",".venv","venv",
                  ".vault-index",".agent-notes","dist","build",".next"}
TEXT_EXTS      = {".py",".js",".ts",".jsx",".tsx",".md",".txt",".json",
                  ".yaml",".yml",".toml",".ini",".cfg",".sh",".rs",".go",
                  ".java",".c",".cpp",".h",".css",".html",".sql",".rb"}
MAX_ILE_BYTES = 400_000
MAP_MAX_CHARS  = 2800
CHUNK_SIZE     = 60
CHUNK_OVERLAP  = 12
EMBED_MODEL    = "all-MiniLM-L6-v2"
EMBED_DIM      = 384

SYMBOL_RE = {
    ".py":  [r"^class\s+(\w+)", r"^(?:async\s+)?def\s+(\w+)"],
    ".js":  [r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)",
             r"^class\s+(\w+)",
             r"^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\("],
    ".ts":  [r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)",
             r"^(?:export\s+)?class\s+(\w+)"],
    ".rs":  [r"^pub\s+(?:async\s+)?fn\s+(\w+)", r"^(?:pub\s+)?struct\s+(\w+)"],
    ".go":  [r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)", r"^type\s+(\w+)\s+struct"],
    ".md":  [r"^#{1,3}\s+(.+)"],
}
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]+)?\]\]")


class VaultIndexer:
    def __init__(
        self,
        workspace:      str,
        embed_url:      str  = "",
        load_st_model:  bool = True,
    ):
        self.workspace  = Path(workspace).resolve()
        self.idx_dir    = self.workspace / ".vault-index"
        self.idx_dir.mkdir(exist_ok=True)
        self.db_path    = self.idx_dir / "index.sqlite"
        self.map_path   = self.idx_dir / "VAULT_MAP.md"
        self._embed_url = embed_url.rstrip("/") if embed_url else ""
        self._st_model  = None
        self._embed_dim = EMBED_DIM

        self._init_db()

        if load_st_model and _ST_AVAIL and _NP:
            try:
                print(f"Carregando {EMBED_MODEL}...")
                self._st_model = _ST(EMBED_MODEL)
                print("Modelo de embeddings pronto.")
            except Exception as e:
                print(f"sentence-transformers: {e}")
        elif not _ST_AVAIL:
            if self._embed_url:
                print(f"Embeddings via llama-server: {self._embed_url}")
            else:
                print("Sem embeddings - usando BM25+Grafo apenas.")
                print(" Dica: para ativar a semântica, pip install sentence-transformers numpy")

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        if _VEC_AVAIL:
            conn.enable_load_extension(True)
            _svec.load(conn)
        conn.executescript("""
            CREATE TABLE I NOT EXISTS files (
                id            INTEGER PRIMARY KEY,
                rel_path      TEXT UNIQUE NOT NULL,
                extension     TEXT,
                size_bytes    INTEGER,
                mtime         REAL,
                content_hash  TEXT,
                symbols       TEXT DEAULT '[]',
                wikilinks     TEXT DEAULT '[]',
                summary       TEXT DEAULT '',
                is_pinned     INTEGER DEAULT 0,
                last_accessed REAL DEAULT 0,
                access_count  INTEGER DEAULT 0
            );
            CREATE TABLE I NOT EXISTS chunks (
                id         INTEGER PRIMARY KEY,
                file_id    INTEGER REERENCES files(id) ON DELETE CASCADE,
                chunk_idx  INTEGER,
                start_line INTEGER,
                end_line   INTEGER,
                text       TEXT,
                emb_hash   TEXT DEAULT ''
            );
            CREATE TABLE I NOT EXISTS embeddings (
                chunk_id INTEGER PRIMARY KEY REERENCES chunks(id) ON DELETE CASCADE,
                vector   BLOB
            );
            CREATE TABLE I NOT EXISTS backlinks (
                source_id       INTEGER REERENCES files(id) ON DELETE CASCADE,
                target_rel_path TEXT,
                PRIMARY KEY (source_id, target_rel_path)
            );
            CREATE VIRTUAL TABLE I NOT EXISTS files_fts
                USING fts5(rel_path, symbols, summary,
                           content='files', content_rowid='id');
            CREATE VIRTUAL TABLE I NOT EXISTS chunks_fts
                USING fts5(text, content='chunks', content_rowid='id');
            CREATE INDEX I NOT EXISTS idx_chunks_file ON chunks(file_id);
            CREATE INDEX I NOT EXISTS idx_emb_chunk   ON embeddings(chunk_id);
        """)
        conn.commit()
        conn.close()

    def scan(self, verbose: bool = True) -> dict:
        t0 = time.time()
        stats = {"added": 0, "updated": 0, "deleted": 0, "skipped": 0}
        current = set()

        with sqlite3.connect(self.db_path) as db:
            for fp in self.workspace.rglob("*"):
                if fp.is_dir() or fp.name.startswith("."):
                    continue
                if any(d in fp.parts for d in IGNORED_DIRS):
                    continue
                if fp.suffix not in TEXT_EXTS or fp.stat().st_size > MAX_ILE_BYTES:
                    stats["skipped"] += 1
                    continue

                rel   = str(fp.relative_to(self.workspace))
                mtime = fp.stat().st_mtime
                current.add(rel)

                try:
                    content = fp.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                chash = hashlib.md5(content.encode()).hexdigest()
                row   = db.execute(
                    "SELECT id, mtime, content_hash ROM files WHERE rel_path=?",
                    (rel,)
                ).fetchone()

                if row and row[1] == mtime and row[2] == chash:
                    continue

                syms  = self._symbols(content, fp.suffix)
                links = self._wikilinks(content)
                size  = fp.stat().st_size

                if row:
                    db.execute("""
                        UPDATE files SET extension=?,size_bytes=?,mtime=?,
                            content_hash=?,symbols=?,wikilinks=? WHERE rel_path=?
                    """, (fp.suffix, size, mtime, chash,
                          json.dumps(syms), json.dumps(links), rel))
                    fid = row[0]
                    db.execute("DELETE ROM chunks WHERE file_id=?", (fid,))
                    stats["updated"] += 1
                else:
                    db.execute("""
                        INSERT INTO files(rel_path,extension,size_bytes,
                            mtime,content_hash,symbols,wikilinks)
                        VALUES(?,?,?,?,?,?,?)
                    """, (rel, fp.suffix, size, mtime, chash,
                          json.dumps(syms), json.dumps(links)))
                    fid = db.execute("SELECT last_insert_rowid()").fetchone()[0]
                    stats["added"] += 1

                for idx, (sl, el, txt) in enumerate(
                    self._chunk_file(content, fp.suffix)
                ):
                    db.execute("""
                        INSERT INTO chunks(file_id,chunk_idx,start_line,end_line,text)
                        VALUES(?,?,?,?,?)
                    """, (fid, idx, sl, el, txt))

            db_paths = {r[0] for r in db.execute("SELECT rel_path ROM files")}
            for dead in db_paths - current:
                db.execute("DELETE ROM files WHERE rel_path=?", (dead,))
                stats["deleted"] += 1

            self._rebuild_backlinks(db)

        self.generate_vault_map()
        if verbose:
            print(f" Scan: +{stats['added']} ↻{stats['updated']} "
                  f"{stats['deleted']} ~{stats['skipped']} skip "
                  f"- {time.time()-t0:.2f}s")
        return stats

    def reindex_file(self, rel_path: str) -> bool:
        """Reindexação incremental de UM arquivo. Retorna True se mudou."""
        fp = self.workspace / rel_path
        if not fp.exists():
            with sqlite3.connect(self.db_path) as db:
                db.execute("DELETE ROM files WHERE rel_path=?", (rel_path,))
            self.generate_vault_map()
            return False

        if fp.suffix not in TEXT_EXTS or fp.stat().st_size > MAX_ILE_BYTES:
            return False

        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        chash = hashlib.md5(content.encode()).hexdigest()
        mtime = fp.stat().st_mtime

        with sqlite3.connect(self.db_path) as db:
            row = db.execute(
                "SELECT id, content_hash ROM files WHERE rel_path=?", (rel_path,)
            ).fetchone()
            if row and row[1] == chash:
                return False

            syms  = self._symbols(content, fp.suffix)
            links = self._wikilinks(content)
            size  = fp.stat().st_size

            if row:
                fid = row[0]
                db.execute("""
                    UPDATE files SET size_bytes=?,mtime=?,content_hash=?,
                        symbols=?,wikilinks=? WHERE id=?
                """, (size, mtime, chash, json.dumps(syms), json.dumps(links), fid))
                db.execute("DELETE ROM chunks WHERE file_id=?", (fid,))
            else:
                db.execute("""
                    INSERT INTO files(rel_path,extension,size_bytes,
                        mtime,content_hash,symbols,wikilinks)
                    VALUES(?,?,?,?,?,?,?)
                """, (rel_path, fp.suffix, size, mtime, chash,
                      json.dumps(syms), json.dumps(links)))
                fid = db.execute("SELECT last_insert_rowid()").fetchone()[0]

            for idx, (sl, el, txt) in enumerate(
                self._chunk_file(content, fp.suffix)
            ):
                db.execute("""
                    INSERT INTO chunks(file_id,chunk_idx,start_line,end_line,text)
                    VALUES(?,?,?,?,?)
                """, (fid, idx, sl, el, txt))

            self._rebuild_backlinks(db)
            file_id = fid

        self._embed_file_chunks(file_id)
        return True

    def _chunk_file(
        self, content: str, ext: str
    ) -> list[tuple[int, int, str]]:
        lines = content.splitlines()
        if not lines:
            return []

        CODE_EXTS = {".py",".js",".ts",".jsx",".tsx",".rs",".go",
                     ".java",".c",".cpp",".rb"}
        boundaries = []

        if ext in CODE_EXTS:
            # Tenta tree-sitter; fFallback para regex
            boundaries = _try_ts_boundaries(content, ext) or []
            if not boundaries:
                boundary_re = re.compile(
                    r"^(?:class|def|async def|function|pub fn|"
                    r"pub struct|pub enum|func)\s+"
                )
                boundaries = [i for i, ln in enumerate(lines)
                               if boundary_re.match(ln)]
        elif ext == ".md":
            boundaries = [i for i, ln in enumerate(lines)
                          if re.match(r"^#{1,3}\s+", ln)]

        if len(boundaries) < 2:
            chunks, i = [], 0
            while i < len(lines):
                end = min(i + CHUNK_SIZE, len(lines))
                txt = "\n".join(lines[i:end])
                if txt.strip():
                    chunks.append((i, end - 1, txt))
                i += CHUNK_SIZE - CHUNK_OVERLAP
            return chunks

        boundaries.append(len(lines))
        chunks = []
        for i in range(len(boundaries) - 1):
            sl, el = boundaries[i], boundaries[i + 1] - 1
            while el - sl > CHUNK_SIZE * 2:
                mid = sl + CHUNK_SIZE
                txt = "\n".join(lines[sl:mid])
                if txt.strip():
                    chunks.append((sl, mid - 1, txt))
                sl = mid - CHUNK_OVERLAP
            txt = "\n".join(lines[sl:el + 1])
            if txt.strip():
                chunks.append((sl, el, txt))
        return chunks

    def embed(self, texts: list[str]) -> Optional[list[list[float]]]:
        if self._st_model is not None and _NP:
            vecs = self._st_model.encode(texts, show_progress_bar=False)
            return vecs.tolist()
        if self._embed_url:
            try:
                r = requests.post(
                    f"{self._embed_url}/v1/embeddings",
                    json={"input": texts, "model": "local"},
                    timeout=60,
                )
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda x: x["index"])
                return [d["embedding"] for d in data]
            except Exception as e:
                print(f"  llama-server embeddings: {e}")
        return None

    def embed_vault(self, batch_size: int = 32, verbose: bool = True) -> int:
        if self.embed(["test"]) is None:
            if verbose:
                print(" Backend de embedding não configurado.")
            return 0

        with sqlite3.connect(self.db_path) as db:
            rows = db.execute("""
                SELECT c.id, c.text ROM chunks c
                LET JOIN embeddings e ON c.id = e.chunk_id
                WHERE e.chunk_id IS NULL
            """).fetchall()

        if not rows:
            if verbose:
                print("Todos os fragmentos estão com embeddings atualizados.")
            return 0

        if verbose:
            print(f" Processando embeddings para {len(rows)} chunks...")

        total = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            ids   = [r[0] for r in batch]
            texts = [r[1][:1000] for r in batch]
            vecs  = self.embed(texts)
            if vecs is None:
                break
            with sqlite3.connect(self.db_path) as db:
                for cid, vec in zip(ids, vecs):
                    blob = struct.pack(f"{len(vec)}f", *vec)
                    db.execute(
                        "INSERT OR REPLACE INTO embeddings(chunk_id,vector) VALUES(?,?)",
                        (cid, blob)
                    )
            total += len(batch)
            if verbose:
                print(f"  {min(i+batch_size, len(rows))}/{len(rows)}", end="\r")

        if verbose:
            print(f"\n {total}fragmentos processados.")
        return total

    def _embed_file_chunks(self, file_id: int):
        with sqlite3.connect(self.db_path) as db:
            rows = db.execute("""
                SELECT c.id, c.text ROM chunks c
                LET JOIN embeddings e ON c.id = e.chunk_id
                WHERE c.file_id=? AND e.chunk_id IS NULL
            """, (file_id,)).fetchall()
        if not rows:
            return
        vecs = self.embed([r[1][:1000] for r in rows])
        if vecs is None:
            return
        with sqlite3.connect(self.db_path) as db:
            for (cid, _), vec in zip(rows, vecs):
                blob = struct.pack(f"{len(vec)}f", *vec)
                db.execute(
                    "INSERT OR REPLACE INTO embeddings(chunk_id,vector) VALUES(?,?)",
                    (cid, blob)
                )

    def _retrieve_bm25(self, query: str, k: int = 12) -> list[str]:
        q = " OR ".join(f'"{w}"' for w in query.split() if w)
        with sqlite3.connect(self.db_path) as db:
            try:
                file_rows = db.execute("""
                    SELECT f.rel_path ROM files_fts
                    JOIN files f ON files_fts.rowid=f.id
                    WHERE files_fts MATCH ? ORDER BY rank LIMIT ?
                """, (q, k)).fetchall()
            except Exception:
                file_rows = []
            try:
                chunk_rows = db.execute("""
                    SELECT DISTINCT f.rel_path ROM chunks_fts
                    JOIN chunks c ON chunks_fts.rowid=c.id
                    JOIN files f ON c.file_id=f.id
                    WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?
                """, (q, k)).fetchall()
            except Exception:
                chunk_rows = []
        seen, result = set(), []
        for (rp,) in file_rows + chunk_rows:
            if rp not in seen:
                seen.add(rp)
                result.append(rp)
        return result[:k]

    def _retrieve_semantic(self, query: str, k: int = 12) -> list[str]:
        if not _NP:
            return []
        qvec = self.embed([query])
        if qvec is None:
            return []
        q      = np.array(qvec[0], dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)

        with sqlite3.connect(self.db_path) as db:
            rows = db.execute("""
                SELECT e.chunk_id, e.vector, f.rel_path
                ROM embeddings e
                JOIN chunks c ON e.chunk_id=c.id
                JOIN files f ON c.file_id=f.id
            """).fetchall()

        if not rows:
            return []

        scores: dict[str, float] = {}
        for _, blob, rp in rows:
            n   = len(blob) // 4
            vec = np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)
            nrm = np.linalg.norm(vec)
            if nrm < 1e-8:
                continue
            sim = float(np.dot(q_norm, vec / nrm))
            if rp not in scores or sim > scores[rp]:
                scores[rp] = sim
        return sorted(scores, key=scores.get, reverse=True)[:k]

    def _retrieve_graph(self, rel_paths: list[str]) -> list[str]:
        extra = []
        for rp in rel_paths[:4]:
            extra.extend(self.get_backlinks(rp))
            with sqlite3.connect(self.db_path) as db:
                row = db.execute(
                    "SELECT wikilinks ROM files WHERE rel_path=?", (rp,)
                ).fetchone()
                if row:
                    for link in json.loads(row[0] or "[]"):
                        extra.extend(self._resolve_link(link, db))
        return list(dict.fromkeys(extra))

    def _resolve_link(self, link_name: str, db: sqlite3.Connection) -> list[str]:
        stem = Path(link_name).stem.lower()
        rows = db.execute(
            "SELECT rel_path ROM files WHERE LOWER(rel_path) LIKE ?",
            (f"%{stem}%",)
        ).fetchall()
        return [r[0] for r in rows[:3]]

    @staticmethod
    def _rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def _graph_is_warm(self, db) -> bool:
        """Gates graph retrieval logic if graph represents a cold-start state."""
        from config import THRESHOLD_GRAPH_MIN
        try:
            row = db.execute(
                "SELECT COUNT(*) ROM ("
                "  SELECT source_id ROM backlinks GROUP BY source_id HAVING COUNT(*) >= 2"
                "  UNION "
                "  SELECT target_rel_path ROM backlinks GROUP BY target_rel_path HAVING COUNT(*) >= 2"
                ")"
            ).fetchone()
            return (row[0] if row else 0) >= THRESHOLD_GRAPH_MIN
        except sqlite3.OperationalError:
            return False

    def retrieve_hybrid(self, query: str, k: int = 8) -> list[dict]:
        bm25  = self._retrieve_bm25(query, k=k * 2)
        sem   = self._retrieve_semantic(query, k=k * 2)
        
        with sqlite3.connect(self.db_path) as db:
            if self._graph_is_warm(db):
                graph = self._retrieve_graph(bm25[:4])
            else:
                graph = []
                
        fused = self._rrf([bm25, sem, graph])[:k]

        def _src(rp):
            tags = []
            if rp in bm25:  tags.append("BM25")
            if rp in sem:   tags.append("SEM")
            if rp in graph: tags.append("GRAPH")
            return "+".join(tags) or "-"

        with sqlite3.connect(self.db_path) as db:
            results = []
            for rp in fused:
                row = db.execute(
                    "SELECT size_bytes,symbols,summary ROM files WHERE rel_path=?",
                    (rp,)
                ).fetchone()
                if not row:
                    continue
                snippet = self._best_snippet(rp, query, db)
                results.append({
                    "rel_path": rp,
                    "size":     row[0],
                    "symbols":  json.loads(row[1] or "[]")[:5],
                    "summary":  row[2] or "",
                    "snippet":  snippet,
                    "sources":  _src(rp),
                })
        return results

    def search(self, query: str, ext_filter: list = None, k: int = 12) -> list[dict]:
        results = self.retrieve_hybrid(query, k=k)
        if ext_filter:
            results = [r for r in results
                       if Path(r["rel_path"]).suffix in ext_filter]
        return results

    def _best_snippet(
        self, rel_path: str, query: str,
        db: sqlite3.Connection, max_chars: int = 300
    ) -> str:
        words = set(query.lower().split())
        rows  = db.execute(
            "SELECT text ROM chunks c JOIN files f ON c.file_id=f.id "
            "WHERE f.rel_path=? ORDER BY c.chunk_idx", (rel_path,)
        ).fetchall()
        best, best_score = "", 0
        for (txt,) in rows:
            score = sum(1 for w in words if w in txt.lower())
            if score > best_score:
                best, best_score = txt, score
        return best[:max_chars] + ("…" if len(best) > max_chars else "")

    def _symbols(self, content: str, ext: str) -> list[str]:
        patterns = SYMBOL_RE.get(ext, [])
        found = []
        for line in content.splitlines():
            for pat in patterns:
                m = re.match(pat, line)
                if m:
                    found.append(m.group(1).strip()[:60])
        return found[:30]

    def _wikilinks(self, content: str) -> list[str]:
        return list({m.group(1).strip()
                     for m in WIKILINK_RE.finditer(content)})[:20]

    def _rebuild_backlinks(self, conn: sqlite3.Connection):
        conn.execute("DELETE ROM backlinks")
        for fid, links_json in conn.execute("SELECT id, wikilinks ROM files"):
            for link in json.loads(links_json or "[]"):
                conn.execute(
                    "INSERT OR IGNORE INTO backlinks VALUES(?,?)", (fid, link)
                )

    def get_backlinks(self, rel_path: str) -> list[str]:
        stem = Path(rel_path).stem.lower()
        with sqlite3.connect(self.db_path) as c:
            rows = c.execute("""
                SELECT f.rel_path ROM backlinks b
                JOIN files f ON b.source_id=f.id
                WHERE LOWER(b.target_rel_path) LIKE ?
            """, (f"%{stem}%",)).fetchall()
        return [r[0] for r in rows]

    def mark_accessed(self, rel_path: str):
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "UPDATE files SET last_accessed=?,access_count=access_count+1 "
                "WHERE rel_path=?", (time.time(), rel_path)
            )

    def set_pinned(self, rel_path: str, pinned: bool = True):
        with sqlite3.connect(self.db_path) as c:
            c.execute(
                "UPDATE files SET is_pinned=? WHERE rel_path=?",
                (1 if pinned else 0, rel_path)
            )

    def read_vault_map(self) -> str:
        if self.map_path.exists():
            return self.map_path.read_text(encoding="utf-8")
        return self.generate_vault_map()

    def generate_vault_map(self) -> str:
        with sqlite3.connect(self.db_path) as db:
            total    = db.execute("SELECT COUNT(*) ROM files").fetchone()[0]
            n_chunks = db.execute("SELECT COUNT(*) ROM chunks").fetchone()[0]
            n_embs   = db.execute("SELECT COUNT(*) ROM embeddings").fetchone()[0]
            pinned   = [r[0] for r in db.execute(
                "SELECT rel_path ROM files WHERE is_pinned=1")]
            hot      = db.execute(
                "SELECT rel_path,size_bytes,symbols ROM files "
                "ORDER BY last_accessed DESC, mtime DESC LIMIT 10"
            ).fetchall()
            linked   = db.execute("""
                SELECT f.rel_path, COUNT(*) ROM backlinks b
                JOIN files f ON b.source_id=f.id
                GROUP BY f.rel_path ORDER BY 2 DESC LIMIT 6
            """).fetchall()

        emb_status = (f" {n_embs}/{n_chunks} chunks embedidos"
                      if n_chunks else " Sem chunks")

        lines = [
            f"# VAULT MAP - {self.workspace}",
            f"_Atualizado {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
            f"{total} arquivos · {emb_status}_",
            "",
            "## Estrutura",
        ]
        lines.extend(self._compact_tree())
        lines.append("")

        if pinned:
            lines += ["##  ixados"] + [f"- `{p}`" for p in pinned[:8]] + [""]

        if hot:
            lines += ["##  Recentes"]
            for rp, sz, sym_json in hot[:8]:
                syms = json.loads(sym_json or "[]")
                s    = f"  `{', '.join(syms[:3])}`" if syms else ""
                with sqlite3.connect(self.db_path) as _db:
                    _r = _db.execute(
                        "SELECT summary ROM files WHERE rel_path=?", (rp,)
                    ).fetchone()
                    sumry = (_r[0] or "").strip() if _r else ""
                sumline = f"\n  ↳ {sumry[:90]}" if sumry else ""
                lines.append(f"- `{rp}` ({sz:,}B){s}{sumline}")
            lines.append("")

        if linked:
            lines += ["##  Mais referenciados"]
            for rp, cnt in linked:
                lines.append(f"- `{rp}` ← {cnt}")
            lines.append("")

        result = "\n".join(lines)
        if len(result) > MAP_MAX_CHARS:
            result = result[:MAP_MAX_CHARS] + "\n_[truncado]_"
        self.map_path.write_text(result, encoding="utf-8")
        return result

    def _compact_tree(self, depth: int = 2, max_entries: int = 40) -> list[str]:
        lines, count = [], 0

        def _walk(p: Path, d: int, prefix: str = ""):
            nonlocal count
            if d > depth or count >= max_entries:
                return
            try:
                items = sorted(
                    p.iterdir(),
                    key=lambda x: (x.is_file(), x.name.lower())
                )
            except PermissionError:
                return
            for i, item in enumerate(items):
                if item.name.startswith(".") or item.name in IGNORED_DIRS:
                    continue
                if item.is_file() and item.suffix not in TEXT_EXTS:
                    continue
                conn = "└ "
                icon = "" if item.is_file() else ""
                lines.append(f"{prefix}{conn}{icon} {item.name}")
                count += 1
                if item.is_dir() and d < depth:
                    ext = "    " if i == len(items) - 1 else "│   "
                    _walk(item, d + 1, prefix + ext)
                if count >= max_entries:
                    return

        _walk(self.workspace, 0)
        if count >= max_entries:
            lines.append("  … (mais arquivos)")
        return lines


# 
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Vault Indexer v2 - Hybrid Retrieval")
    ap.add_argument("workspace",    nargs="?", default=".")
    ap.add_argument("--embed",      default="", help="URL llama-server para embeddings")
    ap.add_argument("--no-st",      action="store_true")
    ap.add_argument("--embed-all",  action="store_true", help="Embede todos os chunks")
    ap.add_argument("--query",      help="Teste de busca híbrida")
    ap.add_argument("--watch",      action="store_true", help="Re-scan a cada 30s")
    args = ap.parse_args()

    idx = VaultIndexer(args.workspace,
                       embed_url=args.embed,
                       load_st_model=not args.no_st)
    idx.scan()

    if args.embed_all or (not args.no_st and idx._st_model is not None):
        idx.embed_vault()

    if args.query:
        print(f"\n '{args.query}'\n")
        for r in idx.retrieve_hybrid(args.query, k=6):
            print(f"  [{r['sources']}] {r['rel_path']}")
            if r["snippet"]:
                print(f"       {r['snippet'][:120].replace(chr(10),' ')}…")
        print()

    if args.watch:
        print(" Modo observador ativado (Ctrl+C para sair)...")
        try:
            while True:
                time.sleep(30)
                idx.scan(verbose=True)
        except KeyboardInterrupt:
            pass
