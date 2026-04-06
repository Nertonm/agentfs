#!/usr/bin/env python3
"""
vault_watcher.py - Daemon de reindexação reativa

Monitora o workspace e reindexe arquivos modificados em tempo real.
  • watchdog instalado → event-driven (inotify / SEvents / ReadDirectoryChanges)
  • sem watchdog      → polling por hash a cada N segundos (stdlib puro)

Uso:
  python vault_watcher.py [workspace] [--interval 1.5] [--poll 5]
  python vault_watcher.py --config config.toml
"""

import sys, time, threading, argparse, signal, hashlib
from pathlib import Path

try:
    from vault_indexer_v2 import VaultIndexer, TEXT_EXTS, IGNORED_DIRS
except ImportError:
    from vault_indexer import VaultIndexer, TEXT_EXTS, IGNORED_DIRS

# 
try:
    from watchdog.observers import Observer
    from watchdog.events import (
        ileSystemEventHandler,
        ileCreatedEvent, ileModifiedEvent,
        ileDeletedEvent, ileMovedEvent,
    )
    _WATCHDOG = True
except ImportError:
    _WATCHDOG = False


# 
class Debouncer:
    """
    Acumula eventos e dispara o flush após DEBOUNCE_S segundos de silêncio.
    Evita reindexar 30× enquanto o editor salva em múltiplos writes.
    """
    def __init__(self, flush_fn, debounce_s: float = 1.5):
        self._fn      = flush_fn
        self._delay   = debounce_s
        self._pending: set[str] = set()
        self._lock    = threading.Lock()
        self._timer   = None

    def push(self, rel_path: str):
        with self._lock:
            self._pending.add(rel_path)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self._delay, self._flush)
            self._timer.start()

    def _flush(self):
        with self._lock:
            batch       = list(self._pending)
            self._pending.clear()
            self._timer = None
        if batch:
            self._fn(batch)

    def flush_now(self):
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            batch = list(self._pending)
            self._pending.clear()
        if batch:
            self._fn(batch)


# 
if _WATCHDOG:
    class _VaultHandler(ileSystemEventHandler):
        def __init__(self, indexer: VaultIndexer, debouncer: Debouncer):
            self._idx = indexer
            self._deb = debouncer

        def _rel(self, path: str):
            p = Path(path)
            if p.suffix not in TEXT_EXTS:
                return None
            if any(d in p.parts for d in IGNORED_DIRS):
                return None
            try:
                return str(p.relative_to(self._idx.workspace))
            except ValueError:
                return None

        def on_created(self, event):
            if not event.is_directory:
                r = self._rel(event.src_path)
                if r: self._deb.push(r)

        def on_modified(self, event):
            if not event.is_directory:
                r = self._rel(event.src_path)
                if r: self._deb.push(r)

        def on_deleted(self, event):
            if not event.is_directory:
                r = self._rel(event.src_path)
                if r: self._deb.push(r)

        def on_moved(self, event):
            if not event.is_directory:
                for path in (event.src_path, event.dest_path):
                    r = self._rel(path)
                    if r: self._deb.push(r)


# 
class VaultWatcher:
    def __init__(
        self,
        workspace:     str,
        debounce_s:    float = 1.5,
        poll_interval: int   = 5,
        embed_url:     str   = "",
        load_st:       bool  = True,
        verbose:       bool  = True,
    ):
        self.verbose = verbose
        self._stop   = threading.Event()

        self.idx = VaultIndexer(
            workspace,
            embed_url     = embed_url,
            load_st_model = load_st,
        )
        self.deb = Debouncer(self._process_batch, debounce_s)
        self._poll_interval = poll_interval
        self._observer      = None

        if _WATCHDOG:
            self._log("Modo reativo (watchdog) habilitado.")
        else:
            self._log(f" Utilizando verificação por polling a cada {poll_interval}s")
            self._log("   pip install watchdog  para modo event-driven")

    def _log(self, msg: str):
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _process_batch(self, rel_paths: list[str]):
        changed = []
        for rp in rel_paths:
            try:
                if self.idx.reindex_file(rp):
                    changed.append(rp)
            except Exception as e:
                self._log(f" Falha ao processar arquivo {rp}: {e}")

        if changed:
            self.idx.generate_vault_map()
            self._log(f" {len(changed)}arquivo(s) atualizado(s) no índice:")
            for rp in changed:
                self._log(f"   • {rp}")
            # Re-embede apenas os chunks novos dos arquivos alterados
            if self.idx._st_model is not None:
                self.idx.embed_vault(verbose=False)

    def start(self, initial_scan: bool = True):
        if initial_scan:
            self._log(" Iniciando verificação do diretório...")
            self.idx.scan(verbose=self.verbose)
            if self.idx._st_model is not None:
                self.idx.embed_vault(verbose=self.verbose)

        if _WATCHDOG:
            self._start_watchdog()
        else:
            self._start_polling()

    def _start_watchdog(self):
        handler = _VaultHandler(self.idx, self.deb)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.idx.workspace), recursive=True)
        self._observer.start()
        self._log(f" Aguardando alterações no diretório: {self.idx.workspace}")
        self._log("  Pressione Ctrl+C para interromper.")
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.deb.flush_now()
            self._observer.stop()
            self._observer.join()
            self._log("Monitoramento finalizado de forma limpa.")

    def _start_polling(self):
        """Fallback: compara hashes a cada poll_interval segundos."""
        self._log(f"⏱ Verificando alterações no diretório: {self.idx.workspace}")
        self._log("  Pressione Ctrl+C para interromper.")

        def _hash(fp: Path) -> str:
            try:
                return hashlib.md5(fp.read_bytes()).hexdigest()
            except Exception:
                return ""

        # Inicializa snapshot
        last: dict[str, str] = {}
        for fp in self.idx.workspace.rglob("*"):
            if fp.is_file() and fp.suffix in TEXT_EXTS:
                if not any(d in fp.parts for d in IGNORED_DIRS):
                    rel = str(fp.relative_to(self.idx.workspace))
                    last[rel] = _hash(fp)

        try:
            while not self._stop.is_set():
                time.sleep(self._poll_interval)
                current: dict[str, str] = {}

                for fp in self.idx.workspace.rglob("*"):
                    if fp.is_file() and fp.suffix in TEXT_EXTS:
                        if not any(d in fp.parts for d in IGNORED_DIRS):
                            rel = str(fp.relative_to(self.idx.workspace))
                            current[rel] = _hash(fp)

                changed = set()
                for rel, h in current.items():
                    if last.get(rel) != h:
                        changed.add(rel)
                for rel in set(last) - set(current):
                    changed.add(rel)   # deletados

                if changed:
                    self._process_batch(list(changed))

                last = current

        except KeyboardInterrupt:
            pass
        finally:
            self._log("Monitoramento finalizado de forma limpa.")

    def stop(self):
        self._stop.set()


# 
def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        if sys.version_info >= (3, 11):
            import tomllib
            return tomllib.loads(p.read_text(encoding="utf-8"))
        else:
            import tomli
            return tomli.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f" Falha ao carregar configuração: {e}")
        return {}


# 
def main():
    ap = argparse.ArgumentParser(description="Vault Watcher - reindexação reativa")
    ap.add_argument("workspace",    nargs="?", default=".")
    ap.add_argument("--config",     default="config.toml")
    ap.add_argument("--interval",   type=float, default=1.5,
                    help="Debounce em segundos (padrão: 1.5)")
    ap.add_argument("--poll",       type=int,   default=5,
                    help="Intervalo polling sem watchdog (padrão: 5s)")
    ap.add_argument("--embed-url",  default="",
                    help="URL llama-server para embeddings")
    ap.add_argument("--no-st",      action="store_true",
                    help="Não carregar sentence-transformers")
    ap.add_argument("--no-initial-scan", action="store_true")
    args = ap.parse_args()

    cfg        = _load_config(args.config)
    workspace  = cfg.get("workspace", {}).get("path",    args.workspace)
    embed_url  = cfg.get("llama",     {}).get("url",     args.embed_url).rstrip("/v1")
    debounce   = cfg.get("indexer",   {}).get("watch_debounce", args.interval)
    poll       = cfg.get("indexer",   {}).get("poll_interval",  args.poll)
    use_st     = (not args.no_st and
                  cfg.get("indexer", {}).get("use_sentence_transformers", True))

    watcher = VaultWatcher(
        workspace     = workspace,
        debounce_s    = debounce,
        poll_interval = poll,
        embed_url     = embed_url,
        load_st       = use_st,
    )

    signal.signal(signal.SIGTERM, lambda *_: watcher.stop())
    watcher.start(initial_scan=not args.no_initial_scan)


if __name__ == "__main__":
    main()
