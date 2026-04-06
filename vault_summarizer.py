#!/usr/bin/env python3
"""
vault_summarizer.py - Auto-summary por arquivo + chunking AST (tree-sitter)

Duas funcionalidades em um módulo:

1. ileSummarizer - gera resumos de 1-2 frases via LLM local (llama-server)
   • Armazenado em index.sqlite → aparece no VAULT_MAP.md
   • Indexado em TS5 → melhora busca por palavras-chave
   • Lazy (sob demanda) ou batch (toda a vault de uma vez)

2. ts_boundaries - chunking por AST real via tree-sitter (opcional)
   • Retorna line-numbers de início de funções/classes
   • Usado por vault_indexer_v2.py como upgrade do fFallback regex
   • Graceful: retorna None se tree-sitter não estiver instalado

Dependências:
  obrigatórias : requests
  opcionais    : tree-sitter + tree-sitter-python/javascript/typescript/rust/go

Instalação tree-sitter:
  pip install tree-sitter
  pip install tree-sitter-python tree-sitter-javascript tree-sitter-typescript
  pip install tree-sitter-rust tree-sitter-go
"""

import time, json, argparse
from pathlib import Path
import requests

# 
try:
    from tree_sitter import Language, Parser as TSParser
    _TS_CORE = True
except ImportError:
    _TS_CORE = False

# Mapa extensão → módulo tree-sitter
_TS_PKG = {
    ".py":  "tree_sitter_python",
    ".js":  "tree_sitter_javascript",
    ".ts":  "tree_sitter_typescript.language_typescript",
    ".tsx": "tree_sitter_typescript.language_tsx",
    ".rs":  "tree_sitter_rust",
    ".go":  "tree_sitter_go",
    ".rb":  "tree_sitter_ruby",
    ".java":"tree_sitter_java",
    ".c":   "tree_sitter_c",
    ".cpp": "tree_sitter_cpp",
}

# Node types que marcam início de bloco lógico em qualquer linguagem
_BLOCK_TYPES = {
    "function_definition", "async_function_definition",
    "class_definition",    "decorated_definition",
    "function_declaration","class_declaration",
    "method_definition",   "arrow_function",
    "function_item",       "impl_item",
    "struct_item",         "enum_item",
    "method_declaration",  "type_declaration",
}


def ts_boundaries(content: str, ext: str) -> list[int] | None:
    """
    Retorna lista de line-numbers onde começam funções/classes via AST real.
    None se tree-sitter não disponível ou linguagem não suportada.
    """
    if not _TS_CORE:
        return None
    pkg = _TS_PKG.get(ext)
    if not pkg:
        return None
    try:
        import importlib
        mod  = importlib.import_module(pkg)
        lang = Language(mod.language())
        parser = TSParser(lang)
        tree   = parser.parse(content.encode("utf-8", errors="replace"))

        bounds: set[int] = set()

        def _walk(node):
            if node.type in _BLOCK_TYPES:
                bounds.add(node.start_point[0])
            for child in node.children:
                _walk(child)

        _walk(tree.root_node)
        return sorted(bounds) if bounds else None
    except Exception:
        return None


# 
_EXT_LANG = {
    ".py":   "Python",   ".js":  "JavaScript", ".ts":  "TypeScript",
    ".tsx":  "TSX",      ".jsx": "JSX",        ".rs":  "Rust",
    ".go":   "Go",       ".rb":  "Ruby",       ".java":"Java",
    ".c":    "C",        ".cpp": "C++",        ".sh":  "Shell",
    ".md":   "Markdown", ".json":"JSON",       ".toml":"TOML",
    ".yaml": "YAML",     ".yml": "YAML",       ".sql": "SQL",
    ".html": "HTML",     ".css": "CSS",
}

# 
_PROMPT_TMPL = """\
Arquivo: `{rel_path}`
Linguagem: {lang}
Símbolos: {symbols}

Primeiras {n_lines} linhas:
```
{snippet}
```

Em 1-2 frases curtas, descreva o que este arquivo faz e para que serve.\
"""


class ileSummarizer:
    """
    Gera summaries de 1-2 frases para cada arquivo da vault.
    Usa o llama-server local - sem custo extra de VRAM.
    """

    def __init__(self, base_url: str = "http://localhost:8080/v1"):
        self.base_url = base_url.rstrip("/")
        self._cache: dict[str, str] = {}

    def summarize(
        self,
        rel_path:         str,
        content:          str,
        symbols:          list[str],
        max_snippet_lines: int = 40,
    ) -> str:
        """Gera e retorna o summary de um arquivo."""
        if rel_path in self._cache:
            return self._cache[rel_path]

        ext   = Path(rel_path).suffix
        lines = content.splitlines()
        snip  = "\n".join(lines[:max_snippet_lines])
        syms  = ", ".join(symbols[:8]) if symbols else "-"
        lang  = _EXT_LANG.get(ext, ext.lstrip(".") or "texto")

        prompt = _PROMPT_TMPL.format(
            rel_path = rel_path,
            lang     = lang,
            symbols  = syms,
            n_lines  = min(max_snippet_lines, len(lines)),
            snippet  = snip[:1200],
        )

        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model":       "local",
                    "messages":    [
                        {"role": "system",
                         "content": ("Você é um assistente técnico. "
                                     "Responda em português. "
                                     "Máximo 2 frases curtas.")},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens":  80,
                    "stream":      False,
                },
                timeout=60,
            )
            r.raise_for_status()
            summary = r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            summary = f"[erro ao gerar summary: {e}]"

        self._cache[rel_path] = summary
        return summary

    def summarize_vault(
        self,
        indexer,
        batch_size:   int  = 6,
        only_missing: bool = True,
        verbose:      bool = True,
    ) -> int:
        """
        Gera summaries para todos os arquivos da vault.
        only_missing=True → pula arquivos que já têm summary.
        Retorna quantidade de summaries gerados.
        """
        import sqlite3

        with sqlite3.connect(indexer.db_path) as db:
            if only_missing:
                rows = db.execute(
                    "SELECT rel_path, symbols ROM files "
                    "WHERE summary='' OR summary IS NULL "
                    "ORDER BY last_accessed DESC, mtime DESC"
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT rel_path, symbols ROM files "
                    "ORDER BY last_accessed DESC, mtime DESC"
                ).fetchall()

        if not rows:
            if verbose:
                print(" Todos os arquivos já têm summary.")
            return 0

        if verbose:
            print(f"  Gerando summaries para {len(rows)} arquivo(s)...")

        total = 0
        for i, (rel_path, syms_json) in enumerate(rows):
            fp = indexer.workspace / rel_path
            if not fp.exists():
                continue
            try:
                content = fp.read_text(encoding="utf-8", errors="replace")
                symbols = json.loads(syms_json or "[]")
                summary = self.summarize(rel_path, content, symbols)
            except Exception as e:
                summary = f"[erro: {e}]"

            with sqlite3.connect(indexer.db_path) as db:
                db.execute(
                    "UPDATE files SET summary=? WHERE rel_path=?",
                    (summary, rel_path)
                )

            total += 1
            if verbose:
                short = summary[:70].replace("\n", " ")
                print(f"  [{i+1}/{len(rows)}] {rel_path}")
                print(f"           → {short}")

            if (i + 1) % batch_size == 0:
                time.sleep(0.2)   # respiro breve para não travar o LLM

        if verbose:
            print(f"\n {total} summaries gerados.")
        if total:
            indexer.generate_vault_map()

        return total


# 
def main():
    ap = argparse.ArgumentParser(
        description="Vault Summarizer + verificador de tree-sitter"
    )
    ap.add_argument("workspace",  nargs="?", default=".")
    ap.add_argument("--url",      default="http://localhost:8080/v1")
    ap.add_argument("--all",      action="store_true",
                    help="Re-gera summaries mesmo para arquivos que já têm")
    ap.add_argument("--file",     help="Sumariza apenas um arquivo específico")
    ap.add_argument("--check-ts", action="store_true",
                    help="Mostra quais linguagens têm tree-sitter instalado")
    args = ap.parse_args()

    if args.check_ts:
        print(f"tree-sitter core : {'' if _TS_CORE else '  pip install tree-sitter'}")
        for ext, pkg in _TS_PKG.items():
            try:
                import importlib
                importlib.import_module(pkg)
                status = ""
            except ImportError:
                pkg_name = pkg.split(".")[0].replace("_", "-")
                status   = f"  pip install {pkg_name}"
            print(f"  {ext:6s}  {pkg:45s}  {status}")
        return

    try:
        from vault_indexer_v2 import VaultIndexer
    except ImportError:
        from vault_indexer import VaultIndexer

    idx = VaultIndexer(args.workspace)
    sm  = ileSummarizer(base_url=args.url)

    if args.file:
        fp      = idx.workspace / args.file
        content = fp.read_text(encoding="utf-8", errors="replace")
        symbols = idx._symbols(content, fp.suffix)
        summary = sm.summarize(args.file, content, symbols)
        print(f"\n {args.file}\n→  {summary}\n")

        import sqlite3
        with sqlite3.connect(idx.db_path) as db:
            db.execute(
                "UPDATE files SET summary=? WHERE rel_path=?",
                (summary, args.file)
            )
        idx.generate_vault_map()
    else:
        sm.summarize_vault(idx, only_missing=not args.all)


if __name__ == "__main__":
    main()
