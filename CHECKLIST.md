# Checklist do Projeto

Acompanhamento de funcionalidades, qualidade e próximos passos.

## Funcionalidades

### Diagnóstico
- [x] Análise estática e AST para C, C++ e Python
- [x] Extração de limites do enunciado (`N`, valores, tempo, múltiplos casos)
- [x] Detectores de erros de compilação, typos e variáveis não declaradas
- [x] 25+ detectores de armadilhas (Beecrowd, NEPS, Codeforces, AtCoder)
- [x] Modelo ML local (`BugMLP` + família algorítmica) em NumPy
- [x] RAG com referências da área
- [x] Cascata de IA gratuita (Groq → OpenRouter → GitHub Models → Ollama → local)
- [x] Classe de bug coerente quando a IA encontra erro lógico (sem banner verde falso)
- [x] Coerção de respostas JSON malformadas da IA

### Execução
- [x] Sandbox Python isolado
- [x] Compilação C/C++ com `-Wall -Wextra`
- [x] ASan/UBSan opcional (`SANDBOX_SANITIZERS=1`)
- [x] Comparação de saída com detecção de `PE`, `WA`, `RTE`, `TLE`, `CE`
- [x] Timeout de compilação não gera "Erro de Compilação" falso
- [x] `<bits/stdc++.h>` pré-compilado na imagem Docker

### Entrada de dados
- [x] Importação por link (NEPS, Beecrowd, AtCoder, CSES, Codeforces)
- [x] Catálogo offline com 3.500+ questões
- [x] OCR de prints (`Ctrl+V`)
- [x] Banco de questões da turma (SQLite)

### Interface
- [x] Editor CodeMirror com destaque da linha do erro
- [x] Fluxo enxuto em 2 passos (código → questão/link)
- [x] Modo aluno (sem spoiler)
- [x] Chat de acompanhamento
- [x] Histórico da sessão
- [x] Tela de PIN única (sem popups repetidos)
- [x] Layout responsivo para celular

## Qualidade e segurança
- [x] Lint (`ruff`) sem erros
- [x] 16 testes automatizados passando (`pytest`)
- [x] CI no GitHub Actions
- [x] Segredos isolados do processo do aluno
- [x] Limites de CPU, memória e arquivo no sandbox (Linux)
- [x] Proteção anti-SSRF na importação por link
- [x] Rate limit por IP e limite de payload
- [x] Headers de segurança HTTP
- [x] PIN de acesso opcional
- [x] Health check `GET /health`
- [x] Imagem Docker enxuta (< 512 MB de RAM)

## Desempenho
- [x] Checagem `-fsyntax-only` desativável (`GCC_SYNTAX_CHECK=0`)
- [x] Build rápido `-O0` com cabeçalho pré-compilado
- [x] Timeout menor por provedor de IA (15 s) para fallback mais rápido
- [ ] Medir tempo de resposta real em produção e ajustar timeouts
- [ ] Cache de diagnósticos para códigos idênticos

## Próximos passos
- [ ] Testes end-to-end da interface (Playwright) em desktop e celular
- [ ] Ampliar detectores com casos reais coletados pelo feedback dos monitores
- [ ] Re-treinar o modelo local com os casos de feedback
- [ ] Suporte a Java
- [ ] Painel simples com estatísticas de uso (bugs mais comuns da turma)
