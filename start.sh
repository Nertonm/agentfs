#!/usr/bin/env bash
# start.sh - Inicia o stack completo do SmartContext Agent
#
# Uso:
#   ./start.sh [workspace] [modelo.gguf]              → chat interativo
#   ./start.sh [workspace] [modelo.gguf] "tarefa..."  → execução autônoma
#
# Exemplos:
#   ./start.sh . ~/models/mistral-7b-q4_k_m.gguf
#   ./start.sh ./projeto ~/models/mistral-7b-q4_k_m.gguf "documentar as classes públicas"

set -e

WORKSPACE="${1:-.}"
MODEL="${2:-model.gguf}"
TASK="${3:-}"
LLAMA_PORT=8080
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/config.toml"

# ── Cores ─────────────────────────────────────────────────────────────────────
GRN='\033[0;32m'; YLW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GRN}[start]${NC} $*"; }
warn() { echo -e "${YLW}[start]${NC} $*"; }
err()  { echo -e "${RED}[start]${NC} $*" >&2; }

log "SmartContext Agent"
log "Workspace  : $WORKSPACE"
log "Modelo     : $MODEL"
[ -n "$TASK" ] && log "Tarefa     : $TASK"
echo ""

# ── 1. Dependências Python ────────────────────────────────────────────────────
log "Verificando dependências Python..."
python3 -c "import requests"              2>/dev/null || { warn "Instalando requests...";    pip install -q requests; }
python3 -c "import numpy"                 2>/dev/null || { warn "Instalando numpy...";       pip install -q numpy; }
python3 -c "import sentence_transformers" 2>/dev/null || warn "sentence-transformers ausente - busca semântica desativada. (pip install sentence-transformers)"
python3 -c "import watchdog"              2>/dev/null || warn "watchdog ausente - usando polling. (pip install watchdog)"

# ── 2. llama-server ───────────────────────────────────────────────────────────
if curl -sf "http://localhost:$LLAMA_PORT/health" | grep -q "ok" 2>/dev/null; then
    log "llama-server já rodando em :$LLAMA_PORT "
else
    if command -v llama-server &>/dev/null; then
        log "Iniciando llama-server ($MODEL)..."
        llama-server -m "$MODEL" -ngl 99 -c 4096 \
            --host 0.0.0.0 --port "$LLAMA_PORT" \
            --log-disable > /tmp/llama_server.log 2>&1 &
        LLAMA_PID=$!
        echo $LLAMA_PID > /tmp/smartctx_llama.pid
        log "Aguardando llama-server iniciar (PID $LLAMA_PID)..."
        for i in $(seq 1 40); do
            sleep 1
            if curl -sf "http://localhost:$LLAMA_PORT/health" | grep -q "ok" 2>/dev/null; then
                log "llama-server pronto "
                break
            fi
            [ $i -eq 40 ] && { err "Timeout - llama-server não iniciou. Log: /tmp/llama_server.log"; exit 1; }
        done
    else
        err "llama-server não encontrado no PATH."
        err "Inicie manualmente: llama-server -m modelo.gguf -ngl 99 -c 4096 --port 8080"
        exit 1
    fi
fi

# ── 3. Vault Watcher (background) ────────────────────────────────────────────
log "Iniciando vault watcher em background..."
python3 "$SCRIPT_DIR/vault_watcher.py" "$WORKSPACE" \
    --config "$CONFIG" \
    --no-initial-scan \
    > /tmp/smartctx_watcher.log 2>&1 &
WATCHER_PID=$!
echo $WATCHER_PID > /tmp/smartctx_watcher.pid
log "Watcher PID $WATCHER_PID   (log: /tmp/smartctx_watcher.log)"

# ── 4. Scan + embed inicial ───────────────────────────────────────────────────
log "Indexando workspace..."
python3 "$SCRIPT_DIR/vault_indexer.py" "$WORKSPACE" \
    --embed-all 2>/dev/null || \
python3 "$SCRIPT_DIR/vault_indexer.py" "$WORKSPACE"
log "Índice pronto "
echo ""

# ── 5. Agente ─────────────────────────────────────────────────────────────────
if [ -n "$TASK" ]; then
    log "Modo autônomo: $TASK"
    python3 "$SCRIPT_DIR/multi_agent.py" "$TASK" \
        --workspace "$WORKSPACE" \
        --url "http://localhost:$LLAMA_PORT/v1"
else
    log "Modo interativo (chat)"
    python3 "$SCRIPT_DIR/agent.py" \
        --workspace "$WORKSPACE" \
        --url "http://localhost:$LLAMA_PORT/v1" \
        --ctx 4096
fi

# ── Cleanup ───────────────────────────────────────────────────────────────────
log "Encerrando watcher (PID $WATCHER_PID)..."
kill "$WATCHER_PID" 2>/dev/null || true
rm -f /tmp/smartctx_watcher.pid /tmp/smartctx_llama.pid
log "Encerrado. Até logo!"
