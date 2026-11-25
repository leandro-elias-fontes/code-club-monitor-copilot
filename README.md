# Code Monitor Copilot

Copiloto de bancada para **instrutores e monitores** de programação competitiva e algoritmos (**C, C++ e Python**), focado em questões do **NEPS, Beecrowd, Codeforces, AtCoder e CSES**.

---

## Principais Recursos

1. **Sandbox Real com Compilador e Sanitizers (`src/sandbox_runner.py`)**:
   - Compila C/C++ com `-Wall -Wextra` e **ASan/UBSan** (quando suportado pelo toolchain Linux) e executa Python em subprocesso isolado (sem acesso às variáveis de ambiente do servidor e com limites de CPU/RAM).
   - Roda múltiplos casos de teste com diff lado a lado (**Saída do aluno vs. Saída esperada**), diferenciando `AC`, `PE` (*Presentation Error*), `WA`, `RTE`, `TLE` e `CE`.
2. **Detectores Determinísticos + ML Local (`src/platform_rules.py`, `src/micro_checker.py`, `src/ml_predictor.py`)**:
   - Mais de 25 padrões específicos para Beecrowd, NEPS, Codeforces e AtCoder (ex: falta de `fixed << setprecision(k)`, divisão inteira `int / int`, `cin >>` antes de `getline`, `1 << k` overflow, `memset` inválido, módulo negativo, cópia rasa `[[0]*m]*n` em Python).
   - Classificador `BugMLP` + `AlgoNB` em NumPy puro e RAG nas 5 principais referências da área (`YouKn0wWho`, `CSES`, `Codeforces`, `AtCoder`, `Maratona SBC`).
3. **Cascata Gratuita de IA (`src/free_llm_engine.py`, opcional)**:
   - Suporta provedores gratuitos em cascata automática (`GROQ_API_KEY` → `OPENROUTER_API_KEY` → `GITHUB_TOKEN` → `OLLAMA_BASE_URL` → motor local), fundamentando a resposta na saída real do compilador/sandbox.
4. **Interface de Bancada (`public/index.html`)**:
   - Editor **CodeMirror** com destaque de sintaxe e marcação automática na linha do erro.
   - Importação direta de enunciados e exemplos por **link** (`/api/problems/fetch-url`, com proteção anti-SSRF) ou por **print (`Ctrl+V` / OCR)**.
   - **Modo aluno (sem spoiler)** para exibir apenas contra-exemplo e pergunta socrática na bancada.
   - Chat socrático de acompanhamento (`/api/chat`), banco de questões da turma em SQLite (`/api/problems/custom`) e PIN opcional (`MONITOR_ACCESS_PIN`).

---

## Como Rodar Localmente

```bash
# Instalar dependências
pip install -r requirements.txt

# Subir o servidor web
uvicorn api.index:app --reload --port 8000
```

Acesse `http://127.0.0.1:8000`.

### Rodar Linters e Testes

```bash
ruff check .
pytest -v
```

---

## Como Subir Gratuitamente no Render (Docker Free)

O repositório já inclui [Dockerfile](Dockerfile) e [render.yaml](render.yaml) configurados para o plano gratuito (**512 MB RAM**, health check em `GET /health` e porta dinâmica `${PORT:-7860}`):

1. Crie uma conta gratuita em **https://render.com** e clique em **New + → Web Service**.
2. Conecte este repositório e selecione **Runtime: Docker** e **Instance Type: Free**.
3. Em **Environment Variables** (opcional, recomendado):
   - `GROQ_API_KEY`: chave gratuita criada em **https://console.groq.com** (sem cartão).
   - `MONITOR_ACCESS_PIN`: senha/PIN para restringir o acesso aos instrutores e monitores.
4. Clique em **Create Web Service**.
