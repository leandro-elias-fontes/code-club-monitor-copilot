# Code Club Monitor Copilot

Aplicacao web enxuta para monitores de programacao competitiva (C, C++ e Python) nas aulas do UberHub Code Club / UFU. O monitor cola o enunciado e o codigo do aluno e recebe em poucos segundos um diagnostico fundamentado via RAG nas principais referencias da area, sem entregar o codigo pronto ao aluno.

## O Problema

Em turmas praticas com cerca de 30 alunos resolvendo exercicios ao mesmo tempo, varias submissoes retornam Wrong Answer (WA) ou Time Limit Exceeded (TLE) simultaneamente. Muitas vezes o aluno nao consegue explicar a logica que tentou escrever, e o monitor gasta varios minutos tentando encontrar um overflow de `int`, um acesso `i <= n` fora dos limites ou um `endl` dentro de um laco.

## Como Funciona

1. **Analise Estatica Deterministica (`src/static_analyzer.py`)**: roda em Python puro (0 tokens) antes da IA. Verifica profundidade de loops aninhados (risco de TLE), uso de `int` sem `long long`, condicao `<=` em lacos 0-indexed, falta de `.clear()`/`memset` em vetores globais com multiplos casos de teste, uso de `endl` em loop e tentativas de Prompt Injection nos comentarios.
2. **Base de Conhecimento RAG por Fontes (`data/knowledge_base/`) + Retriever (`src/rag_retriever.py`)**: motor TF-IDF + similaridade de cosseno 100% em memoria (compativel com Vercel a custo zero) indexando 5 fontes canonicas:
   - `youkn0wwho_topics.json` (YouKn0wWho Academy)
   - `cses_patterns.json` (CSES Problem Set)
   - `codeforces_pitfalls.json` (Codeforces)
   - `atcoder_educational.json` (AtCoder ABC / Educational DP)
   - `maratona_sbc_icpc.json` (Maratona SBC / OBI)
3. **Agente LangGraph em 3 Nos (`src/agent.py`)**: executa `static_check_node` -> `rag_retrieve_node` -> `llm_diagnosis_node` com saida estruturada em Pydantic usando `gemini-3.8-flash` (SDK `google-genai`), rotacao de 2 chaves gratuitas (`GEMINI_API_KEY_1` e `GEMINI_API_KEY_2`) e fallback automatico para `gemini-3.8-flash-lite` em erros 429/503.
4. **Economia de Tokens (`src/demo_cache.py`)**: 5 cenarios classicos pre-calculados em cache para demonstracao instantanea na Vercel gastando zero tokens.

## Estrutura do Projeto

- `data/knowledge_base/`: 5 arquivos JSON separados por fonte canonica.
- `src/rag_retriever.py`: indexador e buscador TF-IDF + cosseno com filtro opcional por fonte/tag.
- `src/static_analyzer.py`: analisador estatico em Python puro.
- `src/agent.py`: grafo LangGraph de 3 nos + cliente Gemini 3.8 Flash.
- `src/demo_cache.py`: 5 exemplos em cache (0 tokens).
- `api/index.py` e `frontend/index.html`: FastAPI + interface web em portugues pronta para Vercel (`vercel.json`).
- `tests/test_copilot.py`: testes automatizados com `pytest`.

## Como Rodar Localmente

```bash
python -m venv .venv
source .venv/bin/activate  # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload
```

Rodar lint e testes:

```bash
ruff check .
pytest -v
```
