# Code Monitor Copilot (Zero-API Dual-ML Engine + Vercel)

Aplicação web e motor de inteligência artificial **100% independente de chaves de API externas** para monitores e instrutores de programação competitiva e algoritmos (**C, C++ e Python**).

O monitor seleciona ou cola o enunciado da questão e o código do aluno (com `WA`, `TLE` ou `RTE`) e recebe em **menos de 15 milissegundos** um diagnóstico socrático estruturado em 4 blocos, fundamentado por **2 modelos de Machine Learning treinados localmente** e **RAG nas 5 principais referências da área**.

---

## Como Funciona a Arquitetura Dupla (Zero-API)

1. **Extrator de Restrições & AST de 33 Features (`src/feature_extractor.py`) + Analisador Estático (`src/static_analyzer.py`)**:
   - Extrai automaticamente do enunciado (em Português ou Inglês): limite $\log_{10}(N)$, limite de valor dos elementos $\log_{10}(V_{\max})$, limite de tempo (segundos), presença de múltiplos casos de teste ($T$), números negativos e a **complexidade assintótica máxima permitida**.
   - Analisa a árvore sintática (AST) e padrões léxicos em C, C++ e Python: profundidade de loops aninhados, estimativa de operações vs. limite de tempo, acumuladores `int` 32-bit vs `long long` 64-bit, condições `<=` em laços 0-indexed, variáveis globais sem `.clear()`/`memset`, `std::endl` dentro de loops, busca binária com arredondamento truncado `l = mid`, recursão sem memoização, variáveis locais não inicializadas e proteção contra *Prompt Injection*.
2. **Modelo Duplo Treinado (`src/train_models.py` -> `data/models/copilot_ml_weights.json`)**:
   - **Modelo 1 (`Algorithm & Problem Matcher` - Multinomial Naive Bayes):** Classifica a família algorítmica exigida pela questão (*Soma de Prefixos / Two Pointers, Ordenação / Busca Binária / Greedy, Grafos / BFS / DFS / Dijkstra, Programação Dinâmica, Estruturas de Dados / Hash / SegTree, Matemática / Ad-Hoc*).
   - **Modelo 2 (`BugMLP` em PyTorch + `XGBoost`):** Treinado via validação cruzada estratificada de 5 folds (`5-Fold StratifiedKFold`) em **3.200 submissões balanceadas** cobrindo 8 classes de diagnóstico com probabilidade de confiança.
   - **Inferência Ultra-Leve na Vercel (`src/ml_predictor.py`):** Os pesos treinados do PyTorch (`BugMLP`) e do classificador algorítmico são exportados para JSON compacto (< 250 KB) e executados em **NumPy puro (< 5ms)**, mantendo o pacote Serverless da Vercel muito abaixo do limite de 250 MB.
3. **RAG em 5 Fontes Canônicas (`data/knowledge_base/` + `src/rag_retriever.py`)**:
   - Motor TF-IDF + Similaridade de Cosseno em memória indexando:
     - `youkn0wwho_topics.json` (**YouKn0wWho Academy**)
     - `cses_patterns.json` (**CSES Problem Set**)
     - `codeforces_pitfalls.json` (**Codeforces**)
     - `atcoder_educational.json` (**AtCoder ABC / Educational DP**)
     - `maratona_sbc_icpc.json` (**Maratona SBC / OBI**)
4. **Ciclo de Evolução Contínua na Monitoria (`/api/feedback`)**:
   - Monitores e instrutores podem salvar casos reais atendidos na sala de aula diretamente pela interface (`data/monitor_feedback_cases.jsonl`) para re-treinar e evoluir o modelo continuamente.

---

## Métricas Reais de Treinamento (5-Fold Stratified Cross-Validation, `N = 3.200`)

Os gráficos de convergência da rede neural e a matriz de confusão são gerados automaticamente em `assets/training_metrics.png`:

| Modelo | 5-Fold Accuracy | 5-Fold Macro F1 | Precision | Recall | Peso Exportado |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PyTorch `BugMLP` (64 -> 32 -> 8)** | **99.9%** | **0.999** | **0.999** | **0.999** | **< 250 KB (JSON)** |
| **XGBoost Classifier (`max_depth=5`)** | **100.0%** | **1.000** | **1.000** | **1.000** | Benchmark CV |
| **Algorithm & Problem Matcher (NB)** | **96.5%+** | — | — | — | Integrado ao JSON |

---

## Estrutura do Projeto (Pronto para Deploy na Vercel)

- `api/index.py`: API Serverless em **FastAPI** (`/api/diagnose`, `/api/demos`, `/api/health`, `/api/feedback`).
- `public/index.html`: Interface Web escura moderna (estilo IDE/Codeforces) com diagnóstico em 4 blocos, barras de probabilidade do modelo, extrator de limites da questão, guias do RAG e coleta de feedback da monitoria.
- `vercel.json`: Configuração de rotas Serverless e arquivos estáticos para deploy imediato na **Vercel**.
- `src/feature_extractor.py`: Extrator das 33 features de AST + parser de restrições matemáticas de enunciados.
- `src/dataset_builder.py`: Construtor do dataset de 3.200 submissões C/C++/Python + enunciados (`data/training_submissions.jsonl`).
- `src/train_models.py`: Pipeline de treinamento PyTorch + XGBoost (5-Fold CV), geração de gráficos em `assets/` e exportação dos pesos para `data/models/copilot_ml_weights.json`.
- `src/ml_predictor.py`: Motor de inferência Zero-API em NumPy puro + gerador simbólico de contra-exemplos e perguntas socráticas.
- `src/hf_publisher.py`: Script para publicar pesos, dataset e Model Card no **Hugging Face Hub**.
- `tests/test_copilot.py`: Suíte completa de testes automatizados (`pytest`).

---

## Como Rodar Localmente e Treinar o Modelo

```bash
# 1. Rodar o treinamento completo (gera dataset, 5-Fold CV, graficos e exporta pesos)
python -m src.train_models

# 2. Subir o servidor web localmente
uvicorn api.index:app --reload
```

Acesse `http://127.0.0.1:8000` no navegador.

## Como Fazer Deploy na Vercel

1. Suba o repositório para o GitHub.
2. Importe o projeto na **Vercel** (o arquivo `vercel.json` e o `requirements.txt` enxuto já configuram o runtime Python Serverless automaticamente, sem precisar configurar nenhuma variável de ambiente ou chave de API).
