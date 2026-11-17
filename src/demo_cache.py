"""Pre-calculated zero-token demo scenarios for instant responses."""

from __future__ import annotations

import hashlib
from typing import Any

DEMO_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "demo_int_overflow",
        "title": "1. Overflow de int (Codeforces / CSES)",
        "language": "cpp",
        "problem_statement": "Dado N <= 10^5 e elementos ate 10^9, imprima a soma de todos os elementos.",
        "code": (
            "#include <iostream>\nusing namespace std;\n\nint main() {\n"
            "    int n;\n    cin >> n;\n    int soma = 0;\n"
            "    for (int i = 0; i < n; i++) {\n        int x;\n        cin >> x;\n"
            "        soma += x;\n    }\n    cout << soma << '\\n';\n    return 0;\n}"
        ),
        "diagnosis": {
            "student_logic": "O aluno leu N e somou todos os elementos na variavel 'soma' em um laco simples O(N).",
            "bug_location": "Linha 7 e 11: 'soma' foi declarada como 'int' (limite ~2.14e9). Com N = 10^5 e x = 10^9, a soma chega a 10^14 e causa Integer Overflow.",
            "counterexample": "Entrada: N = 3, valores = [1000000000, 1000000000, 1000000000] | Saida do Aluno: -1294967296 | Saida Esperada: 3000000000",
            "monitor_tip": "Qual e o maior numero que cabe em um int de 32 bits e quanto da a soma de tres valores iguais a 10^9?",
            "rag_references": [
                {
                    "id": "cf_int_overflow",
                    "source": "Codeforces",
                    "title": "Codeforces: Overflow de int 32-bit vs long long",
                    "category": "Armadilhas C++",
                    "technical_summary": "Multiplicacoes ou somas acumuladas ate 10^18 exigem long long.",
                    "common_mistakes": "Usar int em acumuladores quando valores chegam a 10^9.",
                    "url": "https://codeforces.com/blog/entry/15643",
                    "score": 0.95,
                },
                {
                    "id": "cses_intro_missing_weird",
                    "source": "CSES",
                    "title": "CSES Introductory Problems (Weird Algorithm, Missing Number)",
                    "category": "Introductory Problems",
                    "technical_summary": "Valores intermediarios frequentemente ultrapassam 2^31 - 1.",
                    "common_mistakes": "Declarar variaveis de soma/produto como int 32-bit.",
                    "url": "https://cses.fi/problemset/task/1068",
                    "score": 0.72,
                },
            ],
        },
    },
    {
        "id": "demo_out_of_bounds",
        "title": "2. Out-of-Bounds i <= n (C++)",
        "language": "cpp",
        "problem_statement": "Leia N (1 <= N <= 1000) inteiros em um vetor e imprima o maior elemento.",
        "code": (
            "#include <iostream>\n#include <vector>\nusing namespace std;\n\nint main() {\n"
            "    int n;\n    cin >> n;\n    vector<int> v(n);\n"
            "    for (int i = 0; i <= n; i++) cin >> v[i];\n"
            "    int mx = v[0];\n    for (int i = 1; i <= n; i++) if (v[i] > mx) mx = v[i];\n"
            "    cout << mx << '\\n';\n    return 0;\n}"
        ),
        "diagnosis": {
            "student_logic": "O aluno alocou um vector de tamanho N, leu os dados e buscou o maior valor percorrendo o vetor.",
            "bug_location": "Linhas 9 e 11: os lacos usam 'i <= n' comecando em 0. Como o vector e 0-indexed (0..n-1), acessar v[n] invade memoria fora do vetor.",
            "counterexample": "Entrada: N = 3, valores = [4, 7, 2] | Saida do Aluno: tenta ler v[3] (lixo / Segfault) | Saida Esperada: 7",
            "monitor_tip": "Se um vetor tem tamanho 3 e comeca no indice 0, quais sao os indices validos dele?",
            "rag_references": [
                {
                    "id": "cf_edge_cases_n1",
                    "source": "Codeforces",
                    "title": "Codeforces: Casos de Borda (N=1, Array Vazio, Todos Iguais)",
                    "category": "Casos de Borda",
                    "technical_summary": "Acessos fora de 0..n-1 causam Undefined Behavior ou Runtime Error.",
                    "common_mistakes": "Usar i <= n em laco 0-indexed acessando v[n].",
                    "url": "https://codeforces.com/blog/entry/11121",
                    "score": 0.91,
                }
            ],
        },
    },
    {
        "id": "demo_tle_on2",
        "title": "3. TLE O(N^2) para N=10^5 (YouKn0wWho)",
        "language": "cpp",
        "problem_statement": "Dado um array de N inteiros (N <= 10^5) e um alvo K, diga se existe par com soma K.",
        "code": (
            "#include <iostream>\n#include <vector>\nusing namespace std;\n\nint main() {\n"
            "    int n, k;\n    cin >> n >> k;\n    vector<int> a(n);\n"
            "    for (int i = 0; i < n; i++) cin >> a[i];\n"
            "    for (int i = 0; i < n; i++) {\n"
            "        for (int j = i + 1; j < n; j++) {\n"
            "            if (a[i] + a[j] == k) { cout << \"SIM\\n\"; return 0; }\n"
            "        }\n    }\n    cout << \"NAO\\n\";\n    return 0;\n}"
        ),
        "diagnosis": {
            "student_logic": "O aluno testou todos os pares (i, j) com dois lacos aninhados para verificar se algum par soma K.",
            "bug_location": "Linhas 10-14: dois lacos aninhados resultam em O(N^2). Para N = 10^5, sao ~5*10^9 operacoes, estourando o limite de 1s (TLE).",
            "counterexample": "Entrada: N = 100000, K = -1, array = [1..100000] | Saida do Aluno: TLE (>1s) | Saida Esperada: NAO (em O(N log N) ou O(N))",
            "monitor_tip": "Se voce fixar a[i], qual valor exato precisa procurar no resto do vetor sem usar um segundo for?",
            "rag_references": [
                {
                    "id": "ykw_complexity_limits",
                    "source": "YouKn0wWho",
                    "title": "Complexidade Big-O vs Limites de N em 1 segundo",
                    "category": "Complexidade",
                    "technical_summary": "N <= 10^5 exige O(N) ou O(N log N); O(N^2) excede 10^8 ops.",
                    "common_mistakes": "Dois lacos for aninhados para buscar pares quando N >= 10^4.",
                    "url": "https://youkn0wwho.academy/topic-list/time_complexity",
                    "score": 0.94,
                },
                {
                    "id": "cses_subarray_sums",
                    "source": "CSES",
                    "title": "CSES Sorting & Searching: Subarray Sums I e II",
                    "category": "Sorting and Searching",
                    "technical_summary": "Uso de Two Pointers ou map para evitar busca quadratica.",
                    "common_mistakes": "Usar unordered_map sem custom hash tomando TLE.",
                    "url": "https://cses.fi/problemset/task/1661",
                    "score": 0.68,
                },
            ],
        },
    },
    {
        "id": "demo_binary_search_loop",
        "title": "4. Busca Binaria Loop Infinito (Python)",
        "language": "python",
        "problem_statement": "Dado um array ordenado, encontre o maior indice i tal que arr[i] <= x.",
        "code": (
            "def upper_bound_idx(arr, x):\n"
            "    l, r = 0, len(arr) - 1\n"
            "    ans = -1\n"
            "    while l < r:\n"
            "        mid = (l + r) // 2\n"
            "        if arr[mid] <= x:\n"
            "            ans = mid\n"
            "            l = mid\n"
            "        else:\n"
            "            r = mid - 1\n"
            "    return ans"
        ),
        "diagnosis": {
            "student_logic": "O aluno implementou busca binaria salvando a melhor resposta em 'ans' sempre que arr[mid] <= x.",
            "bug_location": "Linhas 5 e 8: mid = (l + r) // 2 arredonda para baixo. Quando r == l + 1 e arr[mid] <= x, mid vale l e 'l = mid' nao altera o intervalo (loop infinito).",
            "counterexample": "Entrada: arr = [1, 3], x = 5 | Saida do Aluno: Loop infinito (l=0, r=1, mid=0 -> l=0) | Saida Esperada: 1",
            "monitor_tip": "Simule no papel com l = 0 e r = 1: quanto vale (0 + 1) // 2? O intervalo [l, r] diminuiu?",
            "rag_references": [
                {
                    "id": "ykw_binary_search_answer",
                    "source": "YouKn0wWho",
                    "title": "Binary Search e Binary Search on Answer",
                    "category": "Busca Binaria",
                    "technical_summary": "Atribuir l = mid exige mid arredondado para cima ou usar l = mid + 1.",
                    "common_mistakes": "Loop infinito quando mid = (l + r) / 2 e l = mid com r = l + 1.",
                    "url": "https://youkn0wwho.academy/topic-list/binary_search",
                    "score": 0.93,
                }
            ],
        },
    },
    {
        "id": "demo_sbc_globals_endl",
        "title": "5. Globais sem Clear + endl (Maratona SBC)",
        "language": "cpp",
        "problem_statement": "Para T <= 10^4 casos de teste, leia um grafo e imprima o grau de cada vertice.",
        "code": (
            "#include <iostream>\n#include <vector>\nusing namespace std;\n"
            "vector<int> adj[100005];\nint main() {\n    int t; cin >> t;\n"
            "    while (t--) {\n        int n, m; cin >> n >> m;\n"
            "        for (int i = 0; i < m; i++) { int u, v; cin >> u >> v; adj[u].push_back(v); }\n"
            "        for (int i = 1; i <= n; i++) cout << adj[i].size() << endl;\n    }\n}"
        ),
        "diagnosis": {
            "student_logic": "O aluno declarou lista de adjacencia global adj[] e leu T casos de teste imprimindo o tamanho de adj[i].",
            "bug_location": "Linhas 4, 7 e 10: 'adj' global nao e limpo entre casos de teste (contamina casos seguintes -> WA) e 'endl' dentro do laco forca flush dando TLE.",
            "counterexample": "Entrada: T = 2, caso 1 com aresta (1,2), caso 2 sem arestas (n=2, m=0) | Saida do Aluno no caso 2: 1 | Saida Esperada: 0",
            "monitor_tip": "O que acontece com o conteudo de adj[1] quando o segundo caso de teste comeca?",
            "rag_references": [
                {
                    "id": "sbc_global_reset",
                    "source": "Maratona SBC",
                    "title": "Maratona SBC: Multiplos Casos de Teste sem Limpar Globais",
                    "category": "Casos de Teste",
                    "technical_summary": "Limpe apenas os indices 0..n usados no caso atual com adj[i].clear().",
                    "common_mistakes": "Esquecer .clear() entre casos de teste ou dar memset em MAXN dentro de T casos.",
                    "url": "https://maratona.sbc.org.br/sobre/",
                    "score": 0.92,
                },
                {
                    "id": "cf_endl_tle",
                    "source": "Codeforces",
                    "title": "Codeforces: std::endl vs '\\n' causando TLE",
                    "category": "I/O e Performance",
                    "technical_summary": "Use '\\n' dentro de loops para evitar flushes excessivos.",
                    "common_mistakes": "Usar cout << ans << endl dentro de loops com muitas linhas.",
                    "url": "https://codeforces.com/blog/entry/5217",
                    "score": 0.78,
                },
            ],
        },
    },
]

_DEMO_BY_ID: dict[str, dict[str, Any]] = {d["id"]: d for d in DEMO_SCENARIOS}


def _hash_payload(language: str, problem_statement: str, code: str) -> str:
    norm = f"{language}::{problem_statement.strip()}::{code.strip()}"
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


_DEMO_BY_HASH: dict[str, dict[str, Any]] = {
    _hash_payload(d["language"], d["problem_statement"], d["code"]): d for d in DEMO_SCENARIOS
}


def lookup_demo(
    demo_id: str | None, language: str, problem_statement: str, code: str
) -> dict[str, Any] | None:
    """Return pre-calculated diagnosis in <1ms and 0 tokens if request matches a demo."""
    if demo_id and demo_id in _DEMO_BY_ID:
        return _DEMO_BY_ID[demo_id]["diagnosis"]
    h = _hash_payload(language, problem_statement, code)
    if h in _DEMO_BY_HASH:
        return _DEMO_BY_HASH[h]["diagnosis"]
    return None
