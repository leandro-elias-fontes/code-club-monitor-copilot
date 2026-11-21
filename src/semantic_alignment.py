"""Semantic Alignment Analyzer: verifies if student code actually matches the problem's domain, I/O contract, and structure."""

from __future__ import annotations

import re

from src.static_analyzer import Language, _strip_cpp_comments_and_strings


class SemanticMismatchResult:
    """Result when the student's code does not make sense for the given problem statement."""

    def __init__(
        self,
        is_mismatch: bool,
        title: str = "",
        bug_location: str = "",
        counterexample: str = "",
        how_to_fix: str = "",
        monitor_tip: str = "",
        suspect_line: int | None = None,
    ) -> None:
        self.is_mismatch = is_mismatch
        self.title = title
        self.bug_location = bug_location
        self.counterexample = counterexample
        self.how_to_fix = how_to_fix
        self.monitor_tip = monitor_tip
        self.suspect_line = suspect_line


def check_problem_code_alignment(
    problem_statement: str,
    code: str,
    language: Language,
) -> SemanticMismatchResult | None:
    """Detect when the code solves a completely different problem or violates the problem's I/O & domain contract."""
    stmt = (problem_statement or "").strip()
    if len(stmt) < 25:
        return None

    low_stmt = stmt.lower()
    cleaned_code = _strip_cpp_comments_and_strings(code) if language in ("cpp", "c") else code
    low_code = cleaned_code.lower()

    # -------------------------------------------------------------------------
    # 1. Domain Mismatch: Text / String / Cipher / Palindrome / Characters vs Pure Integer Math Code
    # -------------------------------------------------------------------------
    is_string_or_cipher_problem = bool(
        re.search(
            r"\b(string|strings|palavra|palavras|letra|letras|caractere|caracteres|texto|mensagem|"
            r"criptograf|decifra|cifra|cesar|alfabeto|maiuscul|minuscul|palindrom|vogal|vogais|"
            r"substring|anagrama|ascii)\b",
            low_stmt,
        )
    ) and not bool(
        re.search(r"\b(imprima a soma de todos os elementos|soma do subarray|menor caminho)\b", low_stmt)
    )

    if language in ("cpp", "c"):
        code_handles_text = bool(
            re.search(
                r"\b(string|char|getline|isalpha|isdigit|tolower|toupper|substr|strlen|strcmp|fgets|putchar|getchar)\b|"
                r"%s|%c|'a'|'z'|'A'|'Z'",
                cleaned_code,
            )
        )
    else:
        code_handles_text = bool(
            re.search(
                r"\b(str|chr|ord|lower|upper|isalpha|isdigit|replace|join|strip)\b|'a'|'z'|\"a\"|\"z\"",
                cleaned_code,
            )
            or (
                "input()" in cleaned_code
                and "int(input())" not in cleaned_code
                and "map(int" not in cleaned_code
            )
        )

    if is_string_or_cipher_problem and not code_handles_text:
        return SemanticMismatchResult(
            is_mismatch=True,
            title="Incompatibilidade com o Enunciado: Questão de Texto/Strings vs Código Apenas Numérico",
            bug_location=(
                "O enunciado descreve um problema de manipulação de texto/caracteres (strings, letras, cifra ou mensagens), "
                "mas o código do aluno lê e processa apenas números inteiros (int / soma numérica) sem nenhuma variável de texto (string / char)."
            ),
            counterexample=(
                "Entrada da Questão: Texto / letras / strings (conforme o enunciado)\n"
                "Comportamento do Código: Tenta ler inteiros ('cin >> int' / 'int(input())'), falhando imediatamente na leitura da entrada!"
            ),
            how_to_fix=(
                "Substitua a leitura puramente numérica pela estrutura pedida no enunciado: use 'std::string' / 'getline(cin, s)' ou 'char' "
                "para processar os caracteres/letras da mensagem e verifique se o código colado realmente pertence a esta questão."
            ),
            monitor_tip=(
                "Pergunte ao aluno: 'O que a entrada desta questão fornece: números para somar ou palavras/letras para decifrar? "
                "Qual tipo de variável precisamos usar para ler textos e caracteres?'"
            ),
            suspect_line=5,
        )

    # -------------------------------------------------------------------------
    # 2. Domain Mismatch: Graph / Tree / Grid / Edges vs Simple 1D Scalar Loop
    # -------------------------------------------------------------------------
    is_graph_or_tree_problem = bool(
        re.search(
            r"\b(grafo|grafos|graph|vertice|vertices|vertex|aresta|arestas|edges|arvore|tree|"
            r"dijkstra|bfs|dfs|menor rota|shortest path|conexa|adjacencia|bipartido)\b",
            low_stmt,
        )
    )
    if language in ("cpp", "c"):
        code_has_graph_or_2d = bool(
            re.search(
                r"\b(adj|grafo|graph|vis|visited|dist|fila|queue|priority_queue|dfs|bfs|dijkstra|pair|push_back)\b|"
                r"\[[^\]]+\]\s*\[[^\]]+\]",
                low_code,
            )
        )
        # Count how many variables are read inside the main loop
        reads_edges = bool(re.search(r"cin\s*>>\s*\w+\s*>>\s*\w+|scanf\s*\(\s*\"%d\s*%d\"", low_code))
    else:
        code_has_graph_or_2d = bool(
            re.search(r"\b(adj|graph|vis|visited|dist|deque|heapq|dfs|bfs|append)\b", low_code)
        )
        reads_edges = "split()" in low_code

    if is_graph_or_tree_problem and not code_has_graph_or_2d and not reads_edges:
        return SemanticMismatchResult(
            is_mismatch=True,
            title="Incompatibilidade com o Enunciado: Questão de Grafos/Árvores vs Código Escalares 1D",
            bug_location=(
                "O enunciado exige representar e percorrer um Grafo/Árvore (vértices e arestas), mas o código do aluno "
                "faz apenas uma leitura linear de números isolados sem construir lista de adjacência, matriz ou leitura de pares (u, v)."
            ),
            counterexample=(
                "Entrada da Questão: N vértices, M arestas e pares de conexões (u, v)\n"
                "Comportamento do Código: Lê apenas uma lista 1D simples e ignora a topologia do grafo."
            ),
            how_to_fix=(
                "Construa uma lista de adjacência ('vector<int> adj[N]' em C++ ou 'adj = [[] for _ in range(n+1)]' em Python), "
                "leia os pares de arestas (u, v) e aplique a travessia exigida (BFS, DFS ou Dijkstra)."
            ),
            monitor_tip=(
                "Pergunte ao aluno: 'Como o seu código está guardando quais vértices estão conectados pelas arestas do grafo?'"
            ),
            suspect_line=5,
        )

    # -------------------------------------------------------------------------
    # 3. Domain Mismatch: Matrix / 2D Grid (N x M) vs Single 1D variable reading
    # -------------------------------------------------------------------------
    is_grid_or_matrix_problem = bool(
        re.search(r"\b(matriz|matrix|grid|tabuleiro|linhas e m colunas|n x m|h x w)\b", low_stmt)
    )
    code_reads_single_int_and_sum = bool(
        re.search(r"soma\s*\+=\s*x", low_code) and not re.search(r"vector|\[.*\]|\bsplit\b", low_code)
    )
    if is_grid_or_matrix_problem and code_reads_single_int_and_sum:
        return SemanticMismatchResult(
            is_mismatch=True,
            title="Incompatibilidade com o Enunciado: Questão de Matriz/Grid 2D vs Soma 1D Simples",
            bug_location=(
                "O enunciado trabalha com uma Matriz / Grid 2D (linhas e colunas), mas o código atual lê apenas um único inteiro N "
                "e acumula uma soma simples em 1D."
            ),
            counterexample=(
                "Entrada da Questão: Dimensões (N, M) seguidas por uma grade 2D de células\n"
                "Comportamento do Código: Lê apenas N elementos escalares e encerra antes de ler o restante da matriz."
            ),
            how_to_fix=(
                "Leia as duas dimensões da grade (ex: 'cin >> n >> m;') e utilize uma estrutura 2D ou dois laços aninhados "
                "(linha i de 0 a n-1, coluna j de 0 a m-1) para processar o grid."
            ),
            monitor_tip=(
                "Pergunte ao aluno: 'Quantas dimensões tem a entrada dessa questão? Um único laço até N consegue ler uma tabela N por M?'"
            ),
            suspect_line=5,
        )

    # -------------------------------------------------------------------------
    # 4. Multi-Testcase / EOF Contract Mismatch: Statement explicitly demands per-testcase loop / blank line
    # -------------------------------------------------------------------------
    demands_multi_cases = bool(
        re.search(
            r"\b(cada caso de teste|each test case|multiplos casos de teste|varios casos de teste|"
            r"ate o final do arquivo|until eof|termina com eof)\b",
            low_stmt,
        )
    )
    demands_blank_line_between_cases = bool(
        re.search(r"\b(linha em branco|blank line|empty line)\b", low_stmt)
    )

    has_outer_testcase_loop = bool(
        re.search(
            r"while\s*\(\s*(?:t\s*--|cin\s*>>|scanf|true|1)|for\s*\([^;]*;\s*\w+\s*<=?\s*t\b|sys\.stdin",
            low_code,
        )
    )

    if demands_multi_cases and not has_outer_testcase_loop:
        extra_blank = (
            " Além disso, o enunciado exige imprimir uma linha em branco após cada caso de teste."
            if demands_blank_line_between_cases
            else ""
        )
        return SemanticMismatchResult(
            is_mismatch=True,
            title="Incompatibilidade de Entrada/Saída: Questão com Múltiplos Casos de Teste vs Código de Caso Único",
            bug_location=(
                "O enunciado especifica que a entrada possui múltiplos casos de teste, mas o código do aluno processa "
                f"apenas um único caso e encerra ('return 0').{extra_blank}"
            ),
            counterexample=(
                "Entrada da Questão: Caso de Teste #1 seguido pelo Caso de Teste #2...\n"
                "Comportamento do Código: Processa apenas o primeiro bloco, ignora o restante dos casos de teste e toma Wrong Answer."
            ),
            how_to_fix=(
                "Envolva a leitura e resolução dentro de um laço de casos de teste: 'while (t--)' (se houver inteiro T no início) "
                "ou 'while (cin >> ...)' (se a leitura for até EOF)."
                + (" Não esqueça de imprimir a linha em branco extra ('\\n') após cada caso!" if demands_blank_line_between_cases else "")
            ),
            monitor_tip=(
                "Pergunte ao aluno: 'O enunciado diz que o programa vai rodar para apenas um caso ou para vários casos de teste em sequência? "
                "O que acontece com o segundo caso de teste quando o seu código chega no return 0?'"
            ),
            suspect_line=5,
        )

    # -------------------------------------------------------------------------
    # 5. Output Type Mismatch: Statement demands YES/NO or SIM/NAO boolean decision, code only prints a raw sum
    # -------------------------------------------------------------------------
    demands_yes_no = bool(
        re.search(r"imprima\s+[\"']?(sim|nao|yes|no|possivel|impossivel)[\"']?|print\s+[\"']?(yes|no)[\"']?", low_stmt)
    )
    code_prints_yes_no = bool(
        re.search(r"\"(sim|nao|yes|no|possivel|impossivel)\"|'(sim|nao|yes|no)'", low_code)
    )
    if demands_yes_no and not code_prints_yes_no:
        return SemanticMismatchResult(
            is_mismatch=True,
            title="Incompatibilidade de Saída: Enunciado Pede Decisão (SIM/NÃO ou YES/NO) vs Código Imprime Número",
            bug_location=(
                "O enunciado pede para imprimir uma resposta textual de decisão (como 'SIM'/'NAO' ou 'YES'/'NO'), "
                "mas o código do aluno imprime apenas um valor numérico bruto."
            ),
            counterexample=(
                "Saída Esperada pelo Juiz: 'SIM' ou 'NAO' (ou 'YES' / 'NO')\n"
                "Saída do Código do Aluno: Um número inteiro (Wrong Answer imediato na primeira linha)."
            ),
            how_to_fix=(
                "Adicione a condição lógica pedida pelo problema e imprima exatamente as strings exigidas pelo enunciado "
                "(verifique letras maiúsculas/minúsculas)."
            ),
            monitor_tip=(
                "Pergunte ao aluno: 'O que a seção de Saída (Output) do enunciado manda imprimir na tela: um número ou uma palavra como SIM/NÃO?'"
            ),
            suspect_line=13,
        )

    return None
