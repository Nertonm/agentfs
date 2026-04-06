# SmartContext Agent

Agente de software local com memória persistente, navegação de vault e gerenciamento automático de contexto.
Projetado para rodar com modelos 7B em GPU com 8GB VRAM via `llama-server` (llama.cpp + Vulkan).

---

## Índice

1. [Visão geral](#1-visão-geral)
2. [Requisitos](#2-requisitos)
3. [Instalação](#3-instalação)
4. [Início rápido](#4-início-rápido)
5. [Arquitetura](#5-arquitetura)
6. [Módulos](#6-módulos)
7. [Ferramentas do agente](#7-ferramentas-do-agente)
8. [Modo autônomo (multi-agente)](#8-modo-autônomo-multi-agente)
9. [Configuração](#9-configuração)
10. [Upgrades opcionais](#10-upgrades-opcionais)
11. [Estrutura criada no workspace](#11-estrutura-criada-no-workspace)
12. [Perguntas frequentes](#12-perguntas-frequentes)

---

## 1. Visão geral

SmartContext Agent é um sistema de agente local que resolve o principal gargalo de modelos pequenos:
**o contexto curto (4k–8k tokens) não cabe em projetos reais**.

A solução usa quatro estratégias combinadas:

| Estratégia | Implementação | Benefício |
|------------|---------------|-----------|
| Mapa estrutural compacto | `VAULT_MAP.md` sempre no system prompt | Modelo sabe o que existe sem carregar tudo |
| Memória externa persistente | `.agent-notes/STATE.md` + notas de sessão | Sobrevive a reinicializações e compressões |
| Cache LRU de chunks | `LRUFileCache` no agente interativo | Evita recarregar os mesmos arquivos |
| Contexto isolado por step | `ExecutorAgent` no modo autônomo | Cada step parte com contexto limpo |

### Modos de uso

```
Modo interativo  → agent.py  (chat, exploração, perguntas)
Modo autônomo    → multi_agent.py           (tarefas complexas, refatoração)
```

---

## 2. Requisitos

### Hardware
- GPU com ≥ 8 GB VRAM (NVIDIA ou AMD via Vulkan)
- RAM: ≥ 8 GB
- Disco: ≥ 2 GB livres para o modelo

### Software
- Python 3.10+
- `llama-server` do projeto [llama.cpp](https://github.com/ggerganov/llama.cpp)
- `curl` (verificação de saúde do servidor)

### Modelos recomendados (7B Q4_K_M)
- Mistral 7B Instruct v0.3
- Qwen2.5 7B Instruct
- LLaMA 3.1 8B Instruct
- Gemma 3 9B Instruct

---

## 3. Instalação

### Mínima (BM25 + grafo, sem embeddings semânticos)
```bash
pip install requests
```

### Recomendada (busca semântica + watch reativo)
```bash
pip install requests numpy sentence-transformers watchdog
```

### Completa (+ chunking por AST real)
```bash
pip install requests numpy sentence-transformers watchdog
pip install tree-sitter
pip install tree-sitter-python tree-sitter-javascript tree-sitter-typescript
pip install tree-sitter-rust tree-sitter-go
```

Para verificar quais linguagens têm suporte AST:
```bash
python vault_summarizer.py --check-ts
```

### llama-server (llama.cpp com Vulkan)
```bash
# Ubuntu/Debian
sudo apt install build-essential cmake libvulkan-dev
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)
sudo cp build/bin/llama-server /usr/local/bin/
```

---

## 4. Início rápido

### Linux / macOS
```bash
# Torna o launcher executável
chmod +x start.sh

# Chat interativo
./start.sh ./meu_projeto ~/models/mistral-7b-q4_k_m.gguf

# Tarefa autônoma
./start.sh ./meu_projeto ~/models/mistral-7b-q4_k_m.gguf \
    "documentar todas as classes públicas em docs/"
```

### Windows (PowerShell)
```powershell
# Chat interativo
.\start.ps1 .\meu_projeto C:\models\mistral-7b-q4_k_m.gguf

# Tarefa autônoma
.\start.ps1 .\meu_projeto C:\models\mistral-7b-q4_k_m.gguf \
    "adicionar type hints em todos os arquivos Python"
```

### Sem o launcher (manual)
```bash
# 1. Inicia o llama-server
llama-server -m modelo.gguf -ngl 99 -c 4096 --port 8080 &

# 2. Indexa o workspace
python vault_indexer.py ./projeto --embed-all

# 3. Inicia o watcher em background
python vault_watcher.py ./projeto &

# 4a. Chat interativo
python agent.py --workspace ./projeto

# 4b. OU tarefa autônoma
python multi_agent.py "refatorar os handlers" --workspace ./projeto
```

---

## 5. Arquitetura

```
start.sh / start.ps1
    │
    ├── llama-server  (llama.cpp + Vulkan, porta 8080)
    │       └── modelo GGUF carregado em VRAM
    │
    ├── vault_watcher.py  (processo background)
    │       ├── watchdog (inotify/FSEvents) → evento por arquivo salvo
    │       ├── Debouncer (1.5s) → evita reindexação em cascata
    │       └── vault_indexer_v2.reindex_file() → atualiza SQLite + re-embede
    │
    └── [modo escolhido]
            │
            ├── agent.py  (chat interativo)
            │       ├── system prompt: VAULT_MAP + STATE + tools doc
            │       ├── LRUFileCache (8 slots) → chunks de arquivo
            │       ├── compressão automática a 85% do contexto
            │       └── 14 ferramentas via JSON
            │
            └── multi_agent.py  (execução autônoma)
                    ├── PlannerAgent  (~1k tokens) → JSON com 3-8 steps
                    └── ExecutorAgent × N  (contexto isolado por step)
                            ├── ferramentas filtradas por tipo de step
                            ├── histórico = N × ~30 tokens (1 linha/step)
                            └── checkpoint em PLAN.md + step_results/

Camada de dados (workspace/.vault-index/index.sqlite):
    ┌─ files      ← metadados, symbols, wikilinks, summaries
    ├─ chunks     ← fatias de texto dos arquivos
    ├─ embeddings ← vetores float32 (384 dims, MiniLM-L6)
    ├─ backlinks  ← grafo de [[wikilinks]]
    ├─ files_fts  ← índice FTS5 (busca lexical)
    └─ chunks_fts ← índice FTS5 de chunks
```


### Zonas de pressão da memória (Eviction)

| Zona | Limite | Ação |
|---|---|---|
| Normal | < 70% | Nenhuma ação |
| Advertência | 70-85% | Remove silenciosamente LRU chunks de baixa prioridade |
| Crítica | > 85% | Comprime a metade mais antiga do histórico do agente |

### Pipeline de retrieval híbrido

Cada query passa por três canais independentes que são fundidos com
Reciprocal Rank Fusion (RRF):

```
query
  ├── BM25  (FTS5) ──────────────────┐
  ├── Semântica (cosine, MiniLM-L6) ─┼── RRF → top-K resultados
  └── Grafo (backlinks + wikilinks) ─┘
```

Fórmula RRF: `score(d) = Σ 1 / (60 + rank(d, lista_i))`

Degradação graciosa:
- Sem numpy/sentence-transformers → BM25 + Grafo
- Sem watchdog → polling por hash a cada 5s
- Sem tree-sitter → chunking por regex

---

## 6. Módulos

### `llm_client.py`
Cliente HTTP compartilhado para o `llama-server`.
Usada por `agent.py`, `multi_agent.py` e `vault_summarizer.py`.

```python
from llm_client import call_llm, tok

resposta = call_llm(
    messages    = [{"role": "user", "content": "Olá!"}],
    system      = "Você é um assistente técnico.",
    max_tokens  = 200,
    temperature = 0.1,
    base_url    = "http://localhost:8080/v1",
)
```

---

### `vault_indexer.py`
Núcleo do sistema. Escaneia, indexa e busca no workspace.

```python
from vault_indexer_v2 import VaultIndexer

idx = VaultIndexer("./projeto")
idx.scan()                              # scan completo
idx.embed_vault()                       # embede todos os chunks

# Busca híbrida
resultados = idx.retrieve_hybrid("como funciona o treino", k=8)
for r in resultados:
    print(r["rel_path"], r["sources"], r["snippet"])

# Reindexação de arquivo único (chamada pelo watcher)
idx.reindex_file("src/trainer.py")

# Gerar/ler vault map
mapa = idx.read_vault_map()             # string compacta ≤ 2800 chars
```

**CLI:**
```bash
python vault_indexer.py ./projeto
python vault_indexer.py ./projeto --embed-all
python vault_indexer.py ./projeto --query "função de validação"
python vault_indexer.py ./projeto --watch     # re-scan a cada 30s (fallback)
```

---

### `agent_notebook.py`
Memória persistente do agente em Markdown.

```python
from agent_notebook import AgentNotebook

nb = AgentNotebook("./projeto")

# Lê/escreve o estado atual
print(nb.read_state())
nb.update_state_section("Objetivo atual", "Refatorar o módulo de autenticação")

# Cria nota de sessão
nb.add_note(
    title          = "Refatoração auth",
    content        = "Separei handlers em auth/handlers.py",
    files_touched  = ["src/auth.py", "src/auth/handlers.py"],
    decisions      = ["Usar dataclasses em vez de dicts"],
    open_questions = ["Migrar testes para pytest?"],
)

# Lê notas recentes (compacto para o contexto)
print(nb.read_recent_notes(n=3))
print(nb.search_notes("dataclass"))
```

---

### `agent.py`
Agente interativo com gerenciamento automático de contexto.

```bash
python agent.py --workspace ./projeto --ctx 4096

# Comandos especiais no chat:
#   status   → barra de uso do contexto + status do cache LRU
#   index    → reindexação manual do workspace
#   notas    → lista todas as notas do caderno
#   limpar   → limpa o histórico de mensagens
#   sair     → encerra
```

---

### `vault_watcher.py`
Daemon de reindexação reativa.

```bash
# Com watchdog (recomendado)
python vault_watcher.py ./projeto

# Sem watchdog (polling a cada 3s)
python vault_watcher.py ./projeto --poll 3

# Via config.toml
python vault_watcher.py --config config.toml

# Sem scan inicial (útil quando start.sh já fez o scan)
python vault_watcher.py ./projeto --no-initial-scan
```

---

### `vault_summarizer.py`
Gera summaries de 1-2 frases por arquivo via LLM local.
Armazenados em `index.sqlite` e exibidos no `VAULT_MAP.md`.

```bash
# Sumariza arquivos sem summary
python vault_summarizer.py ./projeto

# Re-sumariza todos (inclusive os que já têm)
python vault_summarizer.py ./projeto --all

# Sumariza um arquivo específico
python vault_summarizer.py ./projeto --file src/trainer.py

# Verifica suporte de tree-sitter
python vault_summarizer.py --check-ts
```

---

### `multi_agent.py`
Orchestrador para tarefas complexas com múltiplos passos.

```bash
# Tarefa direta
python multi_agent.py "adicionar validação de schema nos endpoints POST" \
    --workspace ./projeto

# Modo interativo (pede a tarefa)
python multi_agent.py --workspace ./projeto -i

# Apenas gera o plano (sem executar)
python multi_agent.py "refatorar auth module" --workspace ./projeto --plan-only
```

**Quando usar `multi_agent.py` vs `agent.py`:**

| Situação | Use |
|----------|-----|
| Perguntas, exploração, dúvidas pontuais | `agent.py` |
| Tarefas com > 3 arquivos envolvidos | `multi_agent.py` |
| Refatoração, geração de código em lote | `multi_agent.py` |
| Análise profunda de uma base de código | `multi_agent.py` |
| Debugging rápido | `agent.py` |

---

## 7. Ferramentas do agente

O agente interativo responde com JSON para chamar ferramentas:

| Ferramenta | Descrição |
|-----------|-----------|
| `list_dir` | Lista diretório com árvore visual |
| `read_file` | Lê arquivo por faixas de linhas (paginado) |
| `search` | Busca lexical rápida (BM25) |
| `retrieve` | Busca híbrida BM25 + semântica + grafo |
| `write_file` | Cria ou sobrescreve arquivo + reindexação automática |
| `write_note` | Cria nota datada no caderno do agente |
| `update_state` | Atualiza seção do STATE.md |
| `read_notes` | Lê N notas mais recentes |
| `search_notes` | Busca texto nas notas |
| `summarize` | Gera summary de um arquivo específico |
| `summarize_vault` | Gera summaries para toda a vault |
| `embed_vault` | Embede chunks pendentes |
| `pin` | Fixa/desfixa arquivo no VAULT_MAP |
| `backlinks` | Lista arquivos que linkam para o arquivo dado |

---

## 8. Modo autônomo (multi-agente)

### Tipos de step e ferramentas disponíveis

| Tipo | Ferramentas |
|------|-------------|
| `explore` | `list_dir`, `read_file`, `backlinks` |
| `search` | `retrieve`, `search`, `read_file` |
| `analyze` | `read_file`, `retrieve`, `read_notes` |
| `write` | `write_file`, `write_note`, `update_state` |
| `refactor` | `read_file`, `write_file`, `search` |
| `review` | `read_file`, `retrieve`, `read_notes`, `write_note` |

### Orçamento de contexto por step (4096 tokens)

```
System prompt fixo    ≈  800 tokens  (tarefa + plano + ferramentas + vault map)
Histórico (N steps)   ≈  N × 30 t   (apenas 1 linha resumida por step anterior)
Resposta do modelo    ≈  500 tokens
Disponível para work  ≈  2500 tokens
```

Com 10 steps: overhead histórico = 300 tokens. O modelo ainda tem ~2200 tokens livres.

### Checkpointing

Ao executar, o orchestrador mantém:
```
.agent-notes/
  PLAN.md                 ← plano completo com status em tempo real
  step_results/
    step_001.md           ← resultado detalhado do step 1
    step_002.md           ← resultado detalhado do step 2
  STATE.md                ← atualizado ao final
  YYYY-MM-DD_tarefa.md    ← nota final da execução
```

Se a execução for interrompida, o `PLAN.md` preserva o progresso.

---

## 9. Configuração

Edite `config.toml` conforme seu ambiente:

```toml
[workspace]
path = "."              # diretório do projeto

[llama]
url            = "http://localhost:8080/v1"
context_tokens = 4096
model          = "local"

[indexer]
use_sentence_transformers = true   # false → BM25+Grafo apenas
embed_on_start            = true   # embede ao iniciar o watcher
watch_debounce            = 1.5    # segundos de debounce
poll_interval             = 5      # segundos entre polls (sem watchdog)

[agent]
lru_slots   = 8     # arquivos simultâneos no LRU cache
soft_limit  = 0.70  # 70% → aviso de contexto
hard_limit  = 0.85  # 85% → compressão automática
```

---

## 10. Upgrades opcionais

### sentence-transformers (busca semântica)
```bash
pip install sentence-transformers numpy
```
Ativa o canal semântico na busca híbrida usando `all-MiniLM-L6-v2` (22 MB, CPU).

### watchdog (reindexação event-driven)
```bash
pip install watchdog
```
Substitui o polling por hash por notificações do SO (inotify no Linux,
FSEvents no macOS, ReadDirectoryChanges no Windows). Latência: ~1.5s vs ~5s.

### tree-sitter (chunking por AST)
```bash
pip install tree-sitter tree-sitter-python tree-sitter-javascript
pip install tree-sitter-typescript tree-sitter-rust tree-sitter-go
```
Chunking por limites reais de função/classe em vez de regex.
Melhora a qualidade dos embeddings e da busca semântica.

### sqlite-vec (vetores nativos no SQLite)
```bash
pip install sqlite-vec
```
Substitui o loop Python de cosine similarity por busca vetorial nativa no SQLite.
Impacto notável apenas em vaults com > 10.000 chunks.

---

## 11. Estrutura criada no workspace

```
seu_projeto/
  .vault-index/
    index.sqlite          ← banco principal (arquivos, chunks, embeddings, backlinks)
    VAULT_MAP.md          ← mapa compacto sempre no system prompt
  .agent-notes/
    STATE.md              ← estado atual do agente (sempre no contexto)
    PLAN.md               ← plano atual (modo autônomo)
    step_results/
      step_001.md
      step_002.md
      ...
    2026-04-05_tarefa.md  ← notas de sessão indexadas por data
```

Todos os arquivos `.vault-index/` e `.agent-notes/` podem ser adicionados ao `.gitignore`:
```
.vault-index/
.agent-notes/
```

Ou versionados junto ao projeto para preservar o histórico do agente entre sessões.

---

## 12. Perguntas frequentes

**O agente funciona sem GPU?**
Sim. O `llama-server` roda em CPU, mas fica muito lento para modelos 7B.
Para uso em CPU, prefira modelos 3B (ex.: Phi-3 Mini, Qwen2.5 3B).

**Posso usar com Ollama em vez de llama-server?**
Sim. Ollama expõe `/v1/chat/completions` compatível com OpenAI.
Basta apontar `--url http://localhost:11434/v1`.

**O vault_watcher.py precisa rodar o tempo todo?**
Não. Sem o watcher, o índice fica desatualizado até você rodar
`python vault_indexer.py ./projeto` manualmente (ou digitar `index` no chat).

**Como limitar quais arquivos são indexados?**
Edite `IGNORED_DIRS` e `TEXT_EXTS` no início de `vault_indexer.py`.

**O modelo fica em loop de tool calls?**
O agente interativo limita a 12 chamadas por turno; o executor a 8.
Se o modelo não emitir `CONCLUÍDO:`, o step é encerrado automaticamente.

**Como ver o que o watcher está fazendo?**
```bash
tail -f /tmp/smartctx_watcher.log
```

**Como resetar o índice?**
```bash
rm -rf .vault-index/
python vault_indexer.py ./projeto --embed-all
```
