"""Zero-API ML inference engine: loads exported PyTorch MLP + TF-IDF NB weights using pure NumPy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.feature_extractor import (
    ALGO_FAMILY_LABELS_PT,
    BUG_CLASS_LABELS_PT,
    BUG_CLASSES,
    CodeFeatureAnalysis,
)
from src.micro_checker import check_micro_errors
from src.rag_retriever import tokenize
from src.semantic_alignment import check_problem_code_alignment
from src.static_analyzer import Language

DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "models" / "copilot_ml_weights.json"
)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


class ZeroAPIMLPredictor:
    """Pure-NumPy runtime for the trained PyTorch BugMLP and Algorithm Naive-Bayes/Linear classifier."""

    def __init__(self, weights_path: Path = DEFAULT_WEIGHTS_PATH) -> None:
        self.weights_path = weights_path
        self.payload: dict[str, Any] | None = None
        if self.weights_path.exists():
            self.payload = json.loads(self.weights_path.read_text(encoding="utf-8"))

    def is_loaded(self) -> bool:
        return self.payload is not None

    def predict_algo_family(self, problem_statement: str, code: str = "") -> tuple[str, str, float]:
        """Predict the expected algorithmic family from the problem statement + code tokens."""
        low = f"{problem_statement} {code}".lower()
        # High-precision domain rules first so simple statements never misclassify into Segment Tree
        if any(w in low for w in ("grafo", "graph", "vertice", "aresta", "bfs", "dfs", "dijkstra", "grid", "arvore", "tree", "adj[")):
            fam = "graphs_bfs_dfs_dijkstra"
            return fam, ALGO_FAMILY_LABELS_PT[fam], 0.94
        if any(w in low for w in ("programacao dinamica", "dynamic programming", "moeda", "knapsack", "mochila", "frog", "lcs", "memoizacao", "dp[")):
            fam = "dynamic_programming"
            return fam, ALGO_FAMILY_LABELS_PT[fam], 0.93
        if any(w in low for w in ("busca binaria", "binary search", "ordenado", "lower_bound", "upper_bound", "intervalo", "guloso", "greedy")):
            fam = "sorting_binary_search"
            return fam, ALGO_FAMILY_LABELS_PT[fam], 0.92
        if any(w in low for w in ("segment tree", "fenwick", "segtree", "priority_queue", "unordered_map", "hash")):
            fam = "data_structures_hash_segtree"
            return fam, ALGO_FAMILY_LABELS_PT[fam], 0.91
        if any(w in low for w in ("subarray", "prefixo", "prefix", "two pointers", "dois ponteiros", "janela", "sliding")):
            fam = "prefix_sum_two_pointers"
            return fam, ALGO_FAMILY_LABELS_PT[fam], 0.91
        if any(w in low for w in ("imprima a soma", "soma de todos", "divisores", "primo", "collatz", "aritmetica", "fatorial")):
            fam = "math_ad_hoc"
            return fam, ALGO_FAMILY_LABELS_PT[fam], 0.95

        tokens = tokenize(f"{problem_statement} {code}")
        if self.payload and "algo_model" in self.payload:
            am = self.payload["algo_model"]
            vocab: dict[str, int] = am["vocab"]
            class_priors = np.array(am["class_log_priors"], dtype=np.float64)
            feature_log_prob = np.array(am["feature_log_prob"], dtype=np.float64)
            classes: list[str] = am["classes"]

            x = np.zeros(len(vocab), dtype=np.float64)
            for tok in tokens:
                idx = vocab.get(tok)
                if idx is not None:
                    x[idx] += 1.0
            if np.sum(x) > 0:
                logits = class_priors + (feature_log_prob @ x)
                probs = _softmax(logits)
                best_idx = int(np.argmax(probs))
                fam = classes[best_idx]
                return fam, ALGO_FAMILY_LABELS_PT.get(fam, fam), round(float(probs[best_idx]), 4)

        fam = "math_ad_hoc"
        return fam, ALGO_FAMILY_LABELS_PT[fam], 0.85

    def predict_bug_probabilities(
        self,
        analysis: CodeFeatureAnalysis,
    ) -> tuple[str, float, dict[str, float]]:
        """Run the trained PyTorch MLP weights + AST structural calibration."""
        x = np.array(analysis.features, dtype=np.float64)
        classes = list(BUG_CLASSES)

        if self.payload and "bug_mlp" in self.payload:
            mlp = self.payload["bug_mlp"]
            mean = np.array(mlp["scaler_mean"], dtype=np.float64)
            scale = np.array(mlp["scaler_scale"], dtype=np.float64)
            scale = np.where(scale == 0, 1.0, scale)
            x_norm = (x - mean) / scale

            w1 = np.array(mlp["w1"], dtype=np.float64)
            b1 = np.array(mlp["b1"], dtype=np.float64)
            w2 = np.array(mlp["w2"], dtype=np.float64)
            b2 = np.array(mlp["b2"], dtype=np.float64)
            w3 = np.array(mlp["w3"], dtype=np.float64)
            b3 = np.array(mlp["b3"], dtype=np.float64)

            h1 = _relu(x_norm @ w1.T + b1)
            h2 = _relu(h1 @ w2.T + b2)
            logits = h2 @ w3.T + b3
        else:
            logits = np.zeros(len(classes), dtype=np.float64)

        # Combine neural logits with deterministic AST structural evidence so real-world corner cases never miss
        fmap = analysis.feature_Map
        ast_boost = np.zeros(len(classes), dtype=np.float64)
        idx_map = {c: i for i, c in enumerate(classes)}

        if fmap.get("missing_global_cleanup", 0.0) > 0:
            ast_boost[idx_map["global_state_dirty"]] += 12.0
        if fmap.get("has_bs_mid_assign_bug", 0.0) > 0 or (
            fmap.get("has_recursion", 0.0) > 0 and fmap.get("has_memoization", 0.0) == 0
        ):
            ast_boost[idx_map["infinite_loop_or_recursion"]] += 12.0
        if fmap.get("has_le_in_zero_loop", 0.0) > 0:
            ast_boost[idx_map["off_by_one_bounds"]] += 12.0
        if fmap.get("ops_exceed_time_limit", 0.0) > 0:
            ast_boost[idx_map["tle_quadratic_complexity"]] += 12.0
        if fmap.get("overflow_risk_score", 0.0) >= 0.9:
            ast_boost[idx_map["integer_overflow"]] += 12.0
        if (
            fmap.get("has_endl_in_loop", 0.0) > 0
            and fmap.get("missing_global_cleanup", 0.0) == 0
            and fmap.get("ops_exceed_time_limit", 0.0) == 0
        ):
            ast_boost[idx_map["tle_slow_io_endl"]] += 12.0
        if fmap.get("has_uninitialized_local", 0.0) > 0 or (
            fmap.get("has_zero_max_init", 0.0) > 0 and analysis.constraints.allows_negatives
        ):
            ast_boost[idx_map["edge_case_uninitialized_or_div0"]] += 12.0
        if np.max(ast_boost) == 0 and len(analysis.static_report.findings) == 0:
            ast_boost[idx_map["clean_or_minor_logic"]] += 14.0
            # Penalize edge_case false positive when no AST defect is present
            ast_boost[idx_map["edge_case_uninitialized_or_div0"]] -= 8.0

        probs = _softmax(logits + ast_boost)
        prob_dict = {cls: round(float(probs[i]), 4) for i, cls in enumerate(classes)}
        best_idx = int(np.argmax(probs))
        best_cls = classes[best_idx]
        return best_cls, round(float(probs[best_idx]), 4), prob_dict


def build_symbolic_diagnosis(
    language: Language,
    code: str,
    problem_statement: str,
    bug_class: str,
    bug_confidence: float,
    algo_family: str,
    algo_label: str,
    analysis: CodeFeatureAnalysis,
) -> dict[str, Any]:
    """Generate the pedagogical diagnosis (Logic, Error, Counterexample, How to Fix, Socratic Insight) 100% offline."""
    c = analysis.constraints
    sr = analysis.static_report
    fmap = analysis.feature_Map
    lang_upper = {"cpp": "C++", "c": "C", "python": "Python"}.get(language, language)

    student_logic = (
        f"Codigo em {lang_upper} ({int(fmap.get('code_line_count', 1))} linhas) com complexidade "
        f"estrutural estimada em {sr.estimated_big_o} (profundidade de loops = {sr.max_loop_depth}). "
        f"O problema pertence a familia [{algo_label}] com limite N <= {c.max_n:,} (exige {c.expected_max_complexity})."
    )

    # 0. Micro-Error & Symbol Table / Compiler Check (undeclared variables, typos like 'som' vs 'soma', missing ';', etc.)
    micro = check_micro_errors(code, language)
    if micro is not None and micro.has_error:
        return {
            "student_logic": student_logic,
            "bug_location": micro.bug_location,
            "counterexample": (
                "Nao aplicavel (Erro de Compilacao / Variavel Nao Declarada — o codigo nao compila)."
                if micro.is_compile_error
                else "Entrada de Teste: Qualquer caso onde a condicao do 'if' deveria ser falsa."
            ),
            "how_to_fix": micro.how_to_fix,
            "monitor_tip": micro.monitor_tip,
            "bug_label": micro.title,
            "suspect_line": micro.line,
            "is_compile_error": micro.is_compile_error,
            "override_bug_class": "compile_or_micro_error",
        }

    # 1. Semantic Problem-vs-Code Alignment Check (does the code make sense for the problem statement?)
    sem_mismatch = check_problem_code_alignment(problem_statement, code, language)
    if sem_mismatch is not None and sem_mismatch.is_mismatch:
        return {
            "student_logic": student_logic,
            "bug_location": sem_mismatch.bug_location,
            "counterexample": sem_mismatch.counterexample,
            "how_to_fix": sem_mismatch.how_to_fix,
            "monitor_tip": sem_mismatch.monitor_tip,
            "bug_label": sem_mismatch.title,
            "suspect_line": sem_mismatch.suspect_line,
            "is_compile_error": False,
            "override_bug_class": "problem_code_mismatch",
        }

    line_num = analysis.suspect_lines.get(bug_class)
    line_str = f"Linha {line_num}" if line_num else "Analise Estrutural AST"

    # Special corner case checks (float ==, unordered_map collision, sort comparator <=)
    if fmap.get("has_sort_le_comparator", 0.0) > 0:
        return {
            "student_logic": student_logic,
            "bug_location": "Comparador customizado no std::sort usa '<=' ou '>=' (viola Strict Weak Ordering em C++, causando Segmentation Fault / Runtime Error).",
            "counterexample": "Entrada: N = 30 elementos todos iguais [5, 5, ..., 5]\nSaida do Aluno: Segmentation Fault (std::sort acessa fora do vetor)\nSaida Esperada: [5, 5, ..., 5]",
            "how_to_fix": "Troque '<=' por '<' (ou '>=' por '>') no return da funcao comparadora do std::sort (para elementos iguais o comparador DEVE retornar false).",
            "monitor_tip": "Pergunte ao aluno: 'O que a sua funcao de comparacao retorna quando passamos dois elementos iguais cmp(x, x)? O std::sort permite que um numero seja menor que ele mesmo?'",
            "bug_label": "Violacao de Strict Weak Ordering no std::sort (<=)",
            "suspect_line": line_num,
            "is_compile_error": False,
        }

    if bug_class == "integer_overflow":
        vname = analysis.suspect_vars.get("overflow_var", "soma/acumulador")
        bug_location = (
            f"{line_str}: A variavel '{vname}' usa tipo 'int' (32 bits, limite max 2.147.483.647 ~ 2.14*10^9). "
            f"Pelos limites do problema (N <= 10^{c.max_n_log10:.0f}, valores ate 10^{c.max_val_log10:.0f}), "
            f"o valor acumulado atinge ~10^{fmap['potential_sum_log10']:.0f}, causando Integer Overflow (Wrong Answer)."
        )
        counterexample = (
            "Entrada: N = 3, valores = [1000000000, 1000000000, 1000000000]\n"
            "Saida do Aluno (int 32-bit): -1294967296 (estouro de sinal)\n"
            "Saida Esperada (long long 64-bit): 3000000000"
        )
        how_to_fix = (
            f"Altere a declaracao de '{vname}' (e variaveis de multiplicacao intermediaria) de 'int' para 'long long' "
            f"(64 bits, suporta ate ~9*10^18). Se usar std::accumulate, passe '0LL' em vez de '0'."
        )
        monitor_tip = (
            f"Pergunte ao aluno: 'Qual e o maior valor que cabe em um int de 32 bits (~2*10^9) e quanto pode dar "
            f"o resultado de {vname} quando somamos ou multiplicamos {c.max_n:,} elementos de valor {c.max_val:,}?'"
        )

    elif bug_class == "tle_quadratic_complexity":
        ops_est = int(round(10 ** min(fmap["estimated_ops_log10"], 15.0)))
        bug_location = (
            f"{line_str}: Profundidade de {sr.max_loop_depth} lacos aninhados gera complexidade {sr.estimated_big_o}. "
            f"Para N = {c.max_n:,}, o codigo executa ~10^{fmap['estimated_ops_log10']:.1f} ({ops_est:,}) operacoes, "
            f"estourando o limite de {c.time_limit_sec}s (~10^8 ops/s -> Time Limit Exceeded)."
        )
        counterexample = (
            f"Entrada: N = {min(c.max_n, 100000)}, array = [1, 2, ..., {min(c.max_n, 100000)}] (pior caso sem parada antecipada)\n"
            f"Saida do Aluno: Time Limit Exceeded (> {c.time_limit_sec}s em {sr.estimated_big_o})\n"
            f"Saida Esperada: Execucao em < 0.2s usando {c.expected_max_complexity} ({algo_label})"
        )
        how_to_fix = (
            f"Substitua a busca com laco interno por uma abordagem {c.expected_max_complexity} da familia [{algo_label}]: "
            "pré-ordene o vetor para usar Dois Ponteiros / Busca Binaria, pre-calcule um vetor de Soma de Prefixos, ou use um Map/Set de frequencias."
        )
        monitor_tip = (
            "Pergunte ao aluno: 'Quantas operacoes um juiz online faz em 1 segundo (~10^8)? Se N = 10^5, "
            "o que podemos pre-calcular fora do for para responder cada passo em O(1) ou O(log N) sem o segundo laco?'"
        )

    elif bug_class == "off_by_one_bounds":
        bug_location = (
            f"{line_str}: Condicao de parada '<=' (ou acesso i+1) em estrutura 0-indexed de tamanho N. "
            f"Os indices validos vao de 0 ate N-1; na iteracao i == N o codigo acessa memoria invalida (Out-of-Bounds / Runtime Error ou lixo de memoria)."
        )
        counterexample = (
            "Entrada: N = 3, elementos = [10, 20, 30]\n"
            "Saida do Aluno: Acessa indice [3] na ultima iteracao (Segmentation Fault / IndexError / Lixo)\n"
            "Saida Esperada: Percorrer apenas indices 0, 1 e 2 (condicao i < n)"
        )
        how_to_fix = (
            f"Na {line_str.lower()}, troque a condicao de parada 'i <= n' por 'i < n' (ou 'range(len(v))' em Python). "
            "Se o codigo acessa 'v[i + 1]', o laco deve parar em 'i < n - 1'."
        )
        monitor_tip = (
            "Pergunte ao aluno: 'Se um vetor tem tamanho N = 3 e o primeiro elemento fica no indice 0, "
            "qual e o indice do ultimo elemento? O que acontece quando o for executa com i == N?'"
        )

    elif bug_class == "global_state_dirty":
        extra_endl = " Alem disso, o uso de 'endl' no laco causa flush lento (risco de TLE)." if fmap.get("has_endl_in_loop") else ""
        bug_location = (
            f"{line_str}: Vetor/lista de adjacencia global usado dentro de multiplos casos de teste (while(t--)) "
            f"sem limpeza (.clear() ou memset). Os dados do Caso 1 permanecem na memoria e contaminam o Caso 2 (WA).{extra_endl}"
        )
        counterexample = (
            "Entrada: T = 2 | Caso 1: N=2, M=1, aresta (1,2) | Caso 2: N=2, M=0 (sem arestas)\n"
            "Saida do Aluno no Caso 2: Grau(1) = 1 (herdou a aresta do Caso 1!)\n"
            "Saida Esperada no Caso 2: Grau(1) = 0"
        )
        how_to_fix = (
            "Dentro do 'while (t--)', limpe apenas os indices de 0..N usados no caso atual: "
            "'for (int i = 0; i <= n; i++) adj[i].clear();' (evite memset no array inteiro MAXN para nao tomar TLE)."
        )
        monitor_tip = (
            "Pergunte ao aluno: 'Quando o while(t--) termina o primeiro caso de teste e comeca o segundo, "
            "o que ainda esta guardado dentro do vetor global adj[i]?'"
        )

    elif bug_class == "tle_slow_io_endl":
        bug_location = (
            f"{line_str}: Uso de 'std::endl' dentro de um laco de repeticao para N <= {c.max_n:,}. "
            f"O 'endl' forca a limpeza (flush) do buffer do sistema operacional a cada linha impressa, transformando I/O O(N) em TLE."
        )
        counterexample = (
            f"Entrada: N = {max(c.max_n, 100000)} linhas de saida dentro do loop\n"
            "Saida do Aluno: TLE por 100.000 chamadas de syscall flush (std::endl)\n"
            "Saida Esperada: AC em < 0.1s trocando endl por '\\n'"
        )
        how_to_fix = (
            "Substitua 'endl' por '\\n' em todas as impressoes dentro de loops e adicione "
            "'ios::sync_with_stdio(false); cin.tie(nullptr);' no inicio do main()."
        )
        monitor_tip = (
            "Pergunte ao aluno: 'Voce sabe a diferenca entre imprimir std::endl e imprimir '\\n' dentro de um loop com 10^5 iteracoes?'"
        )

    elif bug_class == "infinite_loop_or_recursion":
        if fmap.get("has_bs_mid_assign_bug", 0.0) > 0:
            bug_location = (
                f"{line_str}: Na busca binaria, 'mid = (l + r) / 2' arredonda para baixo. "
                f"Quando o intervalo chega em r == l + 1 e a condicao e verdadeira, mid == l e a atribuicao 'l = mid' nao reduz o intervalo, gerando Loop Infinito (TLE)."
            )
            counterexample = (
                "Entrada: Vetor = [1, 3], alvo X = 5 (l = 0, r = 1)\n"
                "Saida do Aluno: Loop Infinito (mid = (0+1)//2 = 0 -> faz l = 0, nunca sai do while)\n"
                "Saida Esperada: Indice 1"
            )
            how_to_fix = (
                "Opcao 1: Salve a resposta em 'ans = mid' e avance o ponteiro com 'l = mid + 1' (usando 'while (l <= r)'). "
                "Opcao 2: Se mantiver 'l = mid', arredonde o meio para cima com 'mid = (l + r + 1) / 2'."
            )
            monitor_tip = (
                "Pergunte ao aluno: 'Teste a sua busca binaria no papel quando l = 0 e r = 1: "
                "quanto vale (0 + 1) / 2 na divisao inteira? O valor de l muda depois de l = mid?'"
            )
        else:
            bug_location = (
                f"{line_str}: Funcao recursiva chamada sem tabela de memoizacao (DP) ou com profundidade O(N) para N = {c.max_n:,}. "
                f"Recomputa os mesmos subproblemas exponencialmente O(2^N) (TLE) ou estoura o limite da pilha (RecursionError / Stack Overflow)."
            )
            counterexample = (
                "Entrada: N = 45 (ou Grid/Grafo com profundidade > 1000 em Python)\n"
                "Saida do Aluno: TLE O(2^N) ou RecursionError (maximum recursion depth exceeded)\n"
                "Saida Esperada: O(N) com memoizacao / DP iterativa"
            )
            how_to_fix = (
                "Crie um vetor/dicionario 'dp' inicializado com -1 (ou use @lru_cache(None) + sys.setrecursionlimit(200000) em Python) "
                "para retornar imediatamente estados ja calculados, ou converta para DP iterativa (bottom-up)."
            )
            monitor_tip = (
                "Pergunte ao aluno: 'Quantas vezes a sua funcao recursiva recalcula exatamente o mesmo estado f(k)? "
                "Onde podemos guardar o resultado da primeira vez para responder em O(1) nas proximas?'"
            )

    elif bug_class == "edge_case_uninitialized_or_div0":
        if fmap.get("has_uninitialized_local", 0.0) > 0:
            uvar = analysis.suspect_vars.get("uninit_var", "acumulador")
            bug_location = (
                f"{line_str}: A variavel local '{uvar}' foi declarada em C/C++ sem inicializacao ('= 0') antes de ser somada. "
                f"Variaveis locais na stack contem lixo de memoria arbitrario, gerando Wrong Answer no juiz online."
            )
            counterexample = (
                "Entrada: N = 2, valores = [5, 10]\n"
                f"Saida do Aluno: 32783 (lixo da stack + 15 na variavel '{uvar}')\n"
                "Saida Esperada: 15"
            )
            how_to_fix = f"Inicialize explicitamente a variavel na declaracao ({line_str.lower()}): 'long long {uvar} = 0;'."
            monitor_tip = (
                "Pergunte ao aluno: 'Em C/C++, qual e o valor inicial de uma variavel declarada dentro do main() sem '= 0'? "
                "Ela comeca valendo zero garantidamente?'"
            )
        else:
            bug_location = (
                f"{line_str}: Variavel de maximo/resposta inicializada com 0 ('mx = 0') ou falha em caso de borda (N=1 / numeros negativos). "
                f"Se todos os elementos da entrada forem negativos, o codigo retornara 0 incorretamente."
            )
            counterexample = (
                "Entrada: N = 3, valores = [-10, -5, -20]\n"
                "Saida do Aluno: 0 (pois mx comecou em 0 e nenhum numero e > 0)\n"
                "Saida Esperada: -5"
            )
            how_to_fix = (
                "Inicialize a variavel de maximo com o primeiro elemento do vetor ('mx = v[0]') ou com '-LLONG_MAX' / 'float(\"-inf\")' "
                "em vez de 0, e valide o caso de borda N = 1."
            )
            monitor_tip = (
                "Pergunte ao aluno: 'O que o seu codigo imprime se todos os numeros do vetor de entrada forem negativos, como [-10, -5, -20]?'"
            )

    else:
        bug_location = (
            f"Nenhum bug estrutural ou risco de overflow/TLE detectado! A complexidade {sr.estimated_big_o} "
            f"e o uso de inteiros de 64 bits (long long) atendem perfeitamente ao limite N <= {c.max_n:,} ({c.expected_max_complexity})."
        )
        counterexample = (
            f"Teste de Confirmacao Final:\n"
            f"Entrada Limite: N = {c.max_n:,}, elementos = {c.max_val:,}\n"
            f"Resultado: Executa dentro de {c.time_limit_sec}s sem estouro de memoria ou tipo."
        )
        how_to_fix = (
            "Nenhuma correcao estrutural necessaria. O codigo esta pronto para submissao (Accepted)."
        )
        monitor_tip = (
            "Parabenize o aluno pela atencao aos tipos de 64 bits (long long) e complexidade linear, "
            "e peca apenas para ele validar o caso minimo (N = 1) antes de submeter no juiz!"
        )

    return {
        "student_logic": student_logic,
        "bug_location": bug_location,
        "counterexample": counterexample,
        "how_to_fix": how_to_fix,
        "monitor_tip": monitor_tip,
        "bug_label": "Código Pronto para Submissão (Sem Bugs Detectados)" if bug_class == "clean_or_minor_logic" else BUG_CLASS_LABELS_PT.get(bug_class, bug_class),
        "suspect_line": line_num,
        "is_compile_error": False,
    }
