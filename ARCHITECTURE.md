# Fundamentos Teóricos e Arquitetura - SmartContext Agent

_Versão 1.0 · Abril 2026_

---

## Índice

1. [Enquadramento do problema](#1-enquadramento-do-problema)
2. [O gargalo físico: KV cache e orçamento de contexto](#2-o-gargalo-físico-kv-cache-e-orçamento-de-contexto)
3. [Memória hierárquica - da RAM virtual ao MemGPT](#3-memória-hierárquica--da-ram-virtual-ao-memgpt)
4. [Navegação agentica de filesystem](#4-navegação-agêntica-de-filesystem)
5. [RAG clássico e suas limitações](#5-rag-clássico-e-suas-limitações)
6. [Retrieval híbrido - BM25, semântica e grafo](#6-retrieval-híbrido--bm25-semântica-e-grafo)
7. [Vaults tipo Obsidian como grafo de conhecimento](#7-vaults-tipo-obsidian-como-grafo-de-conhecimento)
8. [Políticas de eviction e gestão do contexto](#8-políticas-de-eviction-e-gestão-do-contexto)
9. [Modelos pequenos e autonomia controlada](#9-modelos-pequenos-e-autonomia-controlada)
10. [Mapa de decisões da arquitetura](#10-mapa-de-decisões-da-arquitetura)
11. [Lacunas identificadas na literatura](#11-lacunas-identificadas-na-literatura)
12. [Referências](#12-referências)

---

## 1. Enquadramento do problema

O objetivo central do SmartContext Agent é: **um agente local com memória
externa persistente, capaz de navegar dinamicamente um diretório/vault e
operar bem com contextos pequenos (2k–8k tokens)**. Esse perfil de requisitos
aparece no ecossistema em peças relativamente maduras, mas raramente como um
sistema completo e _local-first_ end-to-end [REPORT].

O espaço de soluções se divide em três famílias que frequentemente são
confundidas:

### 1.1 Memória hierárquica / virtual context management

Sistemas que tratam o contexto como RAM e usam armazenamento externo (disco,
DB vetorial, grafo, logs) como memória virtual. Aqui entram MemGPT/Letta [1][2]
e, mais recentemente, ByteRover Context Tree [3][4].

### 1.2 Agentes que navegam filesystem e repositórios

Frameworks de engenharia de software que dão ao agente ferramentas para listar,
buscar e editar arquivos, com interfaces que reduzem ruído e custo de tokens.
SWE-agent e OpenHands são referências; Aider é um caso híbrido muito relevante
para orçamento de contexto [5][6][7].

### 1.3 RAG e Agentic/Active RAG

Pipelines que fazem retrieve-inject passivo, ou retrieval sob demanda/iterativo
ativo. Isso resolve perguntas sobre uma base, mas não resolve sozinho um agente
que precisa trabalhar num projeto longo com histórico, decisões e estado [8][9].

A distinção é importante porque confundir as três famílias leva a soluções
subótimas: RAG puro não tem memória operacional; navegação pura de filesystem
não tem retrieval semântico; e sistemas de memória hierárquica sem ACI
(Agent-Computer Interface) expõem o modelo a ruído excessivo.

---

## 2. O gargalo físico: KV cache e orçamento de contexto

### 2.1 KV cache como recurso gerenciado

O gargalo físico que domina o perfil de hardware alvo (GPU com 8 GB VRAM +
llama.cpp/Vulkan) é o custo de VRAM que cresce com o contexto, principalmente
pelo **KV cache** [REPORT]. O paper PagedAttention/vLLM [10] trata o KV cache
como memória paginada para reduzir fragmentação e melhorar throughput de
serving, propondo blocos e um gerenciador para KV. Embora o objetivo do vLLM
seja throughput de serving, a ideia conceitual é diretamente relevante:
_a memória cresce com o contexto e deve ser tratada como recurso gerenciado_.

Para a arquitetura do SmartContext Agent, isso se traduz em:

- **rolling buffer cache** - análogo ao sliding window proposto em discussões
  práticas de llama.cpp [REPORT], evita que KV cache vire o limitador de VRAM
  em sessões longas com contextos de 8k tokens
- separação clara entre o que vive **dentro** do contexto (tokens) e o que
  vive **fora** (SQLite, Markdown, arquivos)

### 2.2 O efeito "lost in the middle"

Mesmo quando é possível encher o contexto com muito conteúdo, o fenômeno
descrito em Liu et al. (2023) [11] mostra que modelos tendem a performar melhor
quando a evidência relevante está no começo/fim do prompt e pior quando está no
meio. Isso implica que carregar tudo é duplamente ruim:

1. custa tokens/VRAM
2. pode reduzir a capacidade do modelo de usar a evidência correta

Essa evidência empírica fundamenta a decisão arquitetural de manter o
**VAULT_MAP.md e STATE.md sempre no início do system prompt**, enquanto chunks
de arquivo são carregados sob demanda.

### 2.3 Implicações para 4k–8k tokens

```
Orçamento típico (4096 tokens):

  System prompt fixo   ≈  600–800 t  (mapa + estado + tools doc)
  Histórico de chat    ≈  800–1200 t  (últimas N mensagens)
  Chunk de arquivo     ≈  300–600 t  (um arquivo médio)
  Resposta do modelo   ≈  400–700 t
  ─────────────────────────────────
  Margem de segurança  ≈  800–1000 t
```

Qualquer arquitetura que ignore esse orçamento e tente carregar o projeto
inteiro no contexto se tornará inoperante em projetos com mais de ~30 arquivos.

---

## 3. Memória hierárquica - da RAM virtual ao MemGPT

### 3.1 MemGPT e a metáfora de SO

O paper MemGPT: Towards LLMs as Operating Systems [1] introduz formalmente a
analogia entre contexto de LLM e RAM de computador. A arquitetura propõe três
camadas:

| Camada | Analogia | Implementação |
|--------|----------|---------------|
| Main context | RAM | Janela de contexto ativa |
| External storage (recall) | Disco/swap | Histórico e memória episódica |
| Archival storage | HD/tape | Conhecimento de longo prazo |

O movimento entre camadas (paging) é decidido pelo próprio modelo via tool
calls especiais (`recall_memory`, `archival_memory_search`). O SmartContext
Agent herda essa metáfora, mas aplica uma versão **parcialmente determinística**
- fundamentada na evidência do MemTool [14] de que modelos 7B têm eficiência
autonômica inferior a modelos grandes.

### 3.2 Letta (ex-MemGPT)

A evolução prática do MemGPT, o framework Letta [2], generaliza os blocos de
memória (`core_memory`, `recall`, `archival`) para agentes stateful com
persistência de sessão. É conectável a endpoints OpenAI-compatíveis, incluindo
llama-server local [REPORT]. A decisão de não usar Letta diretamente no
SmartContext Agent é de **controle e auditabilidade**: o sistema completo
precisa ser depurável sem um framework adicional de alto nível.

### 3.3 ByteRover Context Tree

O ByteRover [3][4] introduz uma das ideias mais alinhadas ao problema: um
sistema de retrieval em 5 tiers progressivos, onde as camadas mais baratas
(sem chamada ao LLM) resolvem a maior parte das queries. O princípio é
análogo ao cache hierárquico de CPUs:

```
Tier 1  → Exact match (zero custo)
Tier 2  → BM25 / fuzzy lexical  (custo baixo)
Tier 3  → Embeddings / semântico (custo médio)
Tier 4  → Grafo de contexto     (custo médio)
Tier 5  → LLM fallback          (custo alto)
```

O SmartContext Agent implementa os Tiers 2–4 de forma unificada via Reciprocal
Rank Fusion, sem adicionar a dependência do framework ByteRover (licença Elastic
2.0, não permissiva).

### 3.4 AMEM - Memória Zettelkasten para agentes

O AMEM (Agentic Memory) [15] propõe memória inspirada no método Zettelkasten:
ao criar uma nova memória, o sistema gera uma nota com atributos (descrição,
keywords, tags) e decide conexões com memórias existentes. O resultado é uma
rede navegável em vez de um índice top-k vetorial.

O módulo `agent_notebook.py` do SmartContext Agent implementa o espírito do
AMEM de forma deliberadamente simples: notas Markdown datadas com frontmatter
implícito (seções padronizadas), sem exigir que o LLM gerencie o grafo de
conexões - o grafo emerge naturalmente dos wikilinks nas notas.

---

## 4. Navegação agêntica de filesystem

### 4.1 Agent-Computer Interface (ACI) - SWE-agent

O SWE-agent [5] formaliza o conceito de **Agent-Computer Interface (ACI)**: um
conjunto pequeno de ações (ver, buscar, editar) com feedback conciso e
guardrails, para permitir navegação de repositórios inteiros e edição
incremental. O paper enfatiza uma diferença crítica:

> "Para LMs, conteúdo distrativo tem custo fixo (memória/compute) e pode
> degradar performance."

Isso é diretamente análogo à metáfora de SO: o ACI é a camada de syscall que
impede o agente de mapear o disco inteiro na RAM automaticamente. O SmartContext
Agent adota esse princípio na definição das 14 ferramentas: cada ferramenta
retorna outputs compactos com indicadores de paginação explícitos
(` N linhas restantes → use read_file com start_line: X`).

### 4.2 OpenHands - context window management explícito

O OpenHands SDK [6] descreve explicitamente um componente de **Context Window
Management** dentro da arquitetura, além de workspace local/remoto e estado
event-sourced. Um detalhe prático valioso para setups locais: o SDK menciona
fallback para prompt-based function calling quando o modelo não tem tool calling
nativo - relevante porque muitos servidores locais não têm tool calling robusto.
O SmartContext Agent adota esse fallback: ferramentas são chamadas via JSON
puro no texto de resposta, não via function_call da API.

### 4.3 Aider - repo map como contexto estrutural compacto

O Aider [7] é importante por um motivo simples: **ele assume que o repo todo
não cabe no contexto**, então cria um _repo map_ (árvore + símbolos) para
orientar o modelo em repos maiores. Esse é o padrão replicado no
`VAULT_MAP.md` do SmartContext Agent - manter um resumo estrutural (árvore,
símbolos, arquivos quentes) sempre carregado, e trazer apenas trechos quando
necessário.

A diferença é que o SmartContext Agent adiciona a dimensão de **grafo de
wikilinks/backlinks** ao mapa estrutural, indo além da metáfora puramente de
código para abranger também vaults de conhecimento no estilo Obsidian.

---

## 5. RAG clássico e suas limitações

O RAG clássico (embed-retrieve-inject) [8] falha em pelo menos quatro pontos
para o caso de uso de agente local em projeto longo [REPORT]:

### 5.1 Tarefa stateful vs. conhecimento estático

O agente precisa lembrar **decisões**, TODOs, hipóteses, mudanças no código -
isso é memória de trabalho, não recuperação de conhecimento estático. RAG puro
não tem noção de estado entre sessões.

### 5.2 Lost in the middle e ruído

Injetar chunks irrelevantes consome orçamento e pode piorar o uso de evidência
[11]. Em contextos de 4k tokens, um chunk errado pode consumir 15% do orçamento.

### 5.3 Context poisoning

Trabalhos recentes mostram que dados contaminados no índice podem ser
explorados para degradar respostas do modelo [REPORT]. Mesmo sem modelo de
ameaça adversarial, o princípio se aplica: o índice não é verdade, e curadoria
e filtros são necessários. O SmartContext Agent mitiga isso via hashing de
conteúdo (`content_hash` em `index.sqlite`) e reindexação incremental.

### 5.4 Embeddings estáticos para código

Código muda frequentemente; chunks estáticos ficam desatualizados. Além disso,
código requer sinais estruturais (símbolos, imports, chamadas) além de
semântica textual. O `vault_indexer.py` combina embeddings com extração
explícita de símbolos via regex (com upgrade opcional para tree-sitter).

### 5.5 SelfRAG e FLARE - Agentic RAG

SelfRAG [12] treina o modelo para decidir *quando* recuperar e produzir
reflection tokens para criticar o que recuperou/gerou. FLARE [13] usa previsão
de sentença e baixa confiança para disparar retrieval forward-looking. Ambos
melhoram a eficiência do retrieval, mas dependem de fine-tuning ou de modelos
com capacidade de auto-avaliação alta - o que é instável em modelos 7B. A
decisão arquitetural do SmartContext Agent é usar retrieval mais barato e
determinístico primeiro (regras, grafo, BM25) e deixar o LLM decidir apenas
quando há ambiguidade real.

---

## 6. Retrieval híbrido - BM25, semântica e grafo

### 6.1 BM25 - Okapi BM25

O BM25 (Best Match 25) [Robertson & Zaragoza, 2009] é uma função de ranking
probabilístico baseada em TF-IDF com normalização por tamanho de documento.
A implementação no SmartContext Agent usa **FTS5 do SQLite**, que implementa
internamente uma variante de BM25 otimizada para buscas rápidas em texto puro.

Pontos fortes para o caso de uso:
- Excelente em nomes de funções, símbolos, identificadores
- Zero dependência externa, funciona em CPU
- Atualização incremental O(1) via triggers FTS5

### 6.2 Dense Retrieval - MiniLM-L6-v2

O modelo `all-MiniLM-L6-v2` (22 MB, 384 dimensões) é treinado para similaridade
semântica de sentenças via sentence-transformers [Reimers & Gurevych, 2019].
Para o contexto do SmartContext Agent, ele cobre casos que BM25 não resolve:
"como funciona o treino" encontra `trainer.py` mesmo que a query não use as
palavras exatas do arquivo.

A busca semântica usa similaridade de cosseno:

  sim(q, d) = (q · d) / (‖q‖ · ‖d‖)

onde q e d são os vetores de embedding da query e do chunk, respectivamente.
Os vetores são armazenados como BLOB float32 em `index.sqlite` - sem dependência
de servidor de vetores externo.

### 6.3 Graph Expansion - wikilinks e backlinks

O terceiro canal de retrieval expande os resultados iniciais via grafo de
wikilinks/backlinks. Se um arquivo relevante é encontrado pelos canais
anteriores, seus vizinhos no grafo (arquivos que o referenciam ou que ele
referencia) são adicionados ao ranking como candidatos adicionais.

Essa abordagem é análoga ao **PageRank como prior probabilístico** sobre o
corpus: arquivos muito referenciados tendem a ser mais centrais ao projeto.

### 6.4 Reciprocal Rank Fusion (RRF)

A fusão dos três rankings usa **Reciprocal Rank Fusion** [Cormack et al., 2009]:

  RRF_score(d) = Σ_i  1 / (k + rank_i(d))

onde k = 60 (constante de suavização), rank_i(d) é a posição do documento d
no ranking i, e a soma é sobre os três rankings (BM25, semântico, grafo).

O RRF tem propriedades importantes para este sistema:
- **Robusto a rankings parciais**: um documento ausente de um ranking recebe
  penalidade implícita, não causa erro
- **Sem calibração de scores**: não exige normalização entre sistemas
  heterogêneos (pontuação FTS5 vs. cosine similarity)
- **Empiricamente superior** à combinação linear ponderada em benchmarks de
  information retrieval quando os rankings têm escalas diferentes

### 6.5 Degradação graciosa

O sistema funciona com qualquer subconjunto dos três canais:

```
Dependências    Canal ativo          Qualidade
─────────────────────────────────────────────────
stdlib apenas   BM25 + Grafo         Boa (busca lexical + estrutura)
+ numpy+ST      BM25 + SEM + Grafo   Melhor (busca semântica)
+ sqlite-vec    idem, nativo SQL      Melhor (10k+ chunks eficiente)
+ tree-sitter   Chunking por AST      Melhor ainda (qualidade dos chunks)
```

---

## 7. Vaults tipo Obsidian como grafo de conhecimento

### 7.1 Estrutura formal

Um diretório Markdown genérico se torna um vault do Obsidian quando adicionamos
sinais de estrutura de grafo:

- **wikilinks** - `[[Nota]]` criando arestas explícitas entre documentos
- **backlinks** - arestas reversas computadas (documentos que apontam para X)
- **tags** e convenções de frontmatter - metadata estruturada

A documentação do plugin de Backlinks do Obsidian define formalmente backlinks
como "links de outras notas apontando para a nota ativa" [REPORT]. No
SmartContext Agent, o grafo é armazenado em `index.sqlite` (tabela `backlinks`)
e reconstruído a cada scan incremental.

### 7.2 Navegação por grafo vs. RAG puro

Um agente com contexto curto se beneficia do grafo porque ele permite um padrão
barato [REPORT]:

1. carregue a nota pivô (a que você abriu/encontrou)
2. carregue apenas vizinhos do grafo (links diretos, backlinks) sob orçamento
3. só então use busca vetorial/textual para expandir

Isso é diferente do RAG puro (top-k por embeddings): o grafo dá **prior
probabilístico e caminhos de navegação** que são interpretáveis e auditáveis.
Um arquivo sem backlinks é provavelmente periférico; um arquivo com 10 backlinks
é provavelmente central ao projeto.

### 7.3 LlamaIndex ObsidianReader

O ObsidianReader do LlamaIndex [REPORT] faz algo diretamente alinhado ao
objetivo de vault-aware: percorre a vault, carrega Markdown e adiciona metadata
incluindo wikilinks e backlinks, construindo um `backlinks_map`. O SmartContext
Agent reimplementa essa extração em stdlib puro via regex, sem depender do
LlamaIndex, mas o conceito é idêntico.

### 7.4 Smart Connections e Khoj

O plugin Smart Connections afirma offline-by-default e embarca um modelo local
de embeddings para fazer similaridade dentro da vault [REPORT]. O Khoj oferece
self-host com integração Obsidian e suporte explícito a llama-cpp-server e
Ollama como backends [REPORT]. Ambos resolvem _encontre notas parecidas_, mas
não resolvem navegação agêntica (sequência de decisões com ferramentas) nem
paging de contexto durante trabalho longo - que são o foco do SmartContext Agent.

---

## 8. Políticas de eviction e gestão do contexto

### 8.1 Estado da arte em eviction de contexto

Ainda há pouca padronização de engenharia para eviction de conteúdo no prompt,
mas aparecem propostas explícitas de demand paging e pressão de memória,
criticando o status quo de "acumula até dar crise" [REPORT]. Uma linha
complementar propõe pruning por relevância em vez de LRU puro, para preservar
contexto essencial em diálogos longos.

### 8.2 LRU + importância - a política adotada

Para 7B, a combinação de heurísticas simples com noção de importância é mais
estável do que tentar descobrir tudo via auto-reflexão do modelo [REPORT].
A política do SmartContext Agent:

```
Prioridade de eviction (menor → fica; maior → sai primeiro):

  NUNCA evict  → objetivo atual, constraints, VAULT_MAP, STATE.md
  Baixa prio   → chunks de arquivos lidos uma vez (LRU puro)
  Média prio   → resultados de busca sem uso posterior
  Alta prio    → duplicatas, conteúdo sem referência no plano atual

  Ao evictar → escrever resumo de 5-10 linhas no caderno do agente
```

Esse padrão aparece explicitamente na arquitetura recomendada do relatório de
pesquisa base [REPORT] e está alinhado ao princípio de demand paging da
literatura recente.

### 8.3 Compressão automática vs. eviction seletiva

A compressão automática do histórico (implementada no `agent.py`
ao atingir 85% do contexto) é uma forma de **compactação de memória** análoga
ao garbage collection: em vez de decidir item a item o que descartar, o sistema
pede ao LLM um resumo da metade mais antiga do histórico e substitui os
tokens originais pelo resumo (3–5 frases, ~80 tokens vs. os ~800 originais).
O resumo é salvo no caderno antes de ser descartado do contexto.

---

## 9. Modelos pequenos e autonomia controlada

### 9.1 MemTool e o custo da autonomia

O paper MemTool [14] faz paging de *ferramentas* (em vez de conteúdo): ele
remove/insere descrições de tools para caber no orçamento de contexto, e mede
a eficiência de remoção em três modos: Autônomo, Workflow e Híbrido.

O resultado central é crítico para a arquitetura do SmartContext Agent:

> "No modo Autônomo, modelos reasoning grandes conseguem alta eficiência de
> remoção de tools, enquanto modelos médios ficam muito abaixo.
> Workflow/Híbrido tende a ser mais consistente."

Isso fundamenta a decisão de usar **ferramentas filtradas por tipo de step**
no `multi_agent.py`: em vez de expor todas as 14 ferramentas ao executor, cada
tipo de step recebe apenas o subconjunto relevante (3–4 ferramentas), reduzindo
o espaço de decisão do modelo.

### 9.2 ReAct - raciocínio e ação intercalados

O paradigma ReAct (Reason+Act) [Yao et al., 2022] propõe intercalar geração de
raciocínio (pensamento em linguagem natural) com ações (chamadas de ferramentas),
o que melhora a capacidade de planejamento e reduz alucinações em modelos de
tamanho intermediário. O SmartContext Agent usa uma versão simplificada: o modelo
pode emitir texto livre como raciocínio antes de emitir o JSON de ferramenta,
sem necessidade de marcadores especiais.

### 9.3 Planner/Executor - isolamento de contexto por step

O `multi_agent.py` implementa a separação Planner/Executor inspirada em
arquiteturas de multi-agente como SWE-agent (Coding Agent + Triage Agent) e
no princípio de **isolamento de contexto por step**: cada Executor recebe um
contexto limpo, com apenas o histórico compacto dos steps anteriores (~30 tokens
por step), o que mantém o orçamento controlado mesmo em tarefas com 10+ steps.

Tabela de orçamento por step (4096 tokens):

```
Componente                 Tokens estimados
─────────────────────────────────────────────
System prompt (task+plan)  ≈ 250
Ferramentas do step        ≈ 150
VAULT_MAP compacto         ≈ 200
Histórico N steps × 30t    ≈ N × 30
Resposta do modelo         ≈ 500
─────────────────────────────────────────────
Disponível para trabalho   ≈ 4096 − 600 − N×30
Com N=10 steps             ≈ 3196 tokens livres
```

---

## 10. Mapa de decisões da arquitetura

A tabela abaixo conecta cada decisão de design a sua base na literatura:

| Decisão | Alternativa descartada | Base na literatura |
|---------|------------------------|-------------------|
| VAULT_MAP sempre no system prompt | Carregar estrutura sob demanda | Aider repo map [7]; "lost in the middle" [11] |
| STATE.md sempre no contexto | Memória só em DB | MemGPT working memory [1]; AMEM [15] |
| SQLite puro (sem servidor de vetores) | Qdrant / Chroma | sqlite-vec [REPORT]; auditabilidade |
| BM25 + semântica + grafo (RRF) | Top-k vetorial puro | ByteRover 5-tier [3]; RRF [Cormack 2009] |
| Ferramentas filtradas por tipo de step | Todas as tools sempre | MemTool Workflow mode [14] |
| Compressão com resumo salvo | Truncar sem registrar | MemGPT archival write [1] |
| JSON no texto (não function_call) | API function calling | OpenHands fallback [6] |
| Watcher com debounce 1.5s | Re-scan periódico | Engenharia de sistemas; UX |
| Chunking por limites de função | Janela fixa de linhas | Aider ACI [7]; tree-sitter |
| Caderno em Markdown puro | DB estruturado | AMEM Zettelkasten [15]; auditabilidade |

---

## 11. Lacunas identificadas na literatura

O relatório de pesquisa base [REPORT] identifica como lacuna recorrente a
**combinação simultânea** de:

1. índice de arquivos como grafo (links/backlinks/símbolos)
2. paging dinâmico sob orçamento rígido (4k–8k)
3. agente com padrão de tomar notas em Markdown persistente
4. funcionamento 100% local com performance aceitável em 7B/8GB

Peças individuais existem: ByteRover aproxima a ideia de Context Tree; Letta
traz a metáfora de OS; LlamaIndex ObsidianReader extrai backlinks; Aider mostra
repo map. Mas a integração _vault navigation + paging + note-taking_ ainda tende
a ser artesanal.

O SmartContext Agent é uma solução end-to-end para essa lacuna específica,
priorizando: (a) mínimo de dependências, (b) auditabilidade (tudo em SQLite e
Markdown), e (c) degradação graciosa em hardware modesto.

Questões em aberto para trabalho futuro:

1. **Avaliação quantitativa do VAULT_MAP**: como medir se o mapa estrutural
   está ajudando ou virando ruído? Métricas candidatas: recall@k antes/depois
   do mapa, tempo até primeira ação correta, taxa de re-leitura de arquivos [REPORT].

2. **Chunking ótimo para código em contexto curto**: janela por função/classe
   (AST), janela de linhas, ou blocos semânticos (imports + símbolos usados)?
   Como isso afeta a qualidade dos embeddings? [REPORT]

3. **Quando usar sliding window attention vs. paging externo**: para modelos
   com Mistral/Gemma sliding window, qual o ponto de virada onde paging externo
   tem mais retorno do que janela longa? [REPORT]

4. **Curadoria contra context poisoning**: filtros por origem, hashes de
   conteúdo (já implementado), verificação cruzada por grafo, ou listas de
   confiança? [REPORT]

5. **Coordenação multi-agente em hardware limitado**: quando sistemas de
   caching/coordenação tipo Pancake [REPORT] fazem sentido mesmo localmente?

---

## 12. Referências

[1] Packer, C. et al. **MemGPT: Towards LLMs as Operating Systems**.
    arXiv:2310.08560, 2023. Base conceitual para contexto como RAM e memória
    em camadas.

[2] **Letta** (ex-MemGPT). Implementação prática de agentes stateful com
    memory blocks. https://github.com/letta-ai/letta (Apache 2.0)

[3] **ByteRover Context Tree**. 5-tier progressive retrieval, Adaptive
    Knowledge Lifecycle. Paper e documentação do projeto ByteRover.

[4] **ByteRover CLI** (`brv`). Materialização prática do Context Tree com
    ferramentas e integração com múltiplos provedores.
    https://github.com/byterover/byterover (Elastic License 2.0)

[5] Yang, J. et al. **SWE-agent: Agent-Computer Interfaces Enable Automated
    Software Engineering**. arXiv:2405.15793, 2024. ACI e custo de contexto
    em navegação de repositórios.

[6] **OpenHands Software Agent SDK**. Runtime modular com Context Window
    Management, workspace e fallback para tool calling.
    https://github.com/All-Hands-AI/OpenHands (MIT core)

[7] **Aider AI**. Repo map para orientar modelos em repos maiores que o
    contexto. https://aider.chat / https://github.com/paul-gauthier/aider

[8] Lewis, P. et al. **Retrieval-Augmented Generation for Knowledge-Intensive
    NLP Tasks**. NeurIPS 2020. Referência fundacional de RAG.

[9] Asai, A. et al. **Self-RAG: Learning to Retrieve, Generate, and Critique
    through Self-Reflection**. arXiv:2310.11511, 2023.

[10] Kwon, W. et al. **Efficient Memory Management for Large Language Model
     Serving with PagedAttention**. SOSP 2023. KV cache como memória paginada.

[11] Liu, N. F. et al. **Lost in the Middle: How Language Models Use Long
     Contexts**. TACL 2024. Evidência empírica de degradação no meio do prompt.

[12] Asai, A. et al. **Self-RAG** (ver [9]).

[13] Jiang, Z. et al. **Active Retrieval Augmented Generation** (FLARE).
     EMNLP 2023. Retrieval forward-looking baseado em baixa confiança.

[14] **MemTool**. Paging de ferramentas para tool context em conversas multi-turn.
     Modos Autônomo/Workflow/Híbrido e benchmark ScaleMCP. arXiv, 2024.

[15] **AMEM - Agentic Memory**. Memória inspirada em Zettelkasten com notas
     estruturadas e links dinâmicos. Repo público para reprodução.

[Robertson & Zaragoza, 2009] Robertson, S., Zaragoza, H. **The Probabilistic
     Relevance Framework: BM25 and Beyond**. Foundations and Trends in
     Information Retrieval, 3(4), 2009.

[Reimers & Gurevych, 2019] Reimers, N., Gurevych, I. **Sentence-BERT: Sentence
     Embeddings using Siamese BERT-Networks**. EMNLP 2019. Base do
     sentence-transformers / all-MiniLM-L6-v2.

[Cormack et al., 2009] Cormack, G. V., Clarke, C. L. A., Buettcher, S.
     **Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank
     Learning Methods**. SIGIR 2009. Fundamentação do RRF usado na fusão
     de rankings.

[Yao et al., 2022] Yao, S. et al. **ReAct: Synergizing Reasoning and Acting
     in Language Models**. ICLR 2023. Intercalação de raciocínio e ações.

[REPORT] **Agente Local com Memória Externa, Vault e Paging - Estado da Arte**.
     Relatório de pesquisa base deste projeto, Abril 2026. Disponível em
     `deep-research-report.md`.
