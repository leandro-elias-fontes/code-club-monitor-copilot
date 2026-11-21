"""Deep AST, problem constraint, and code-vs-problem feature extractor (0 external deps)."""

from __future__ import annotations

import ast
import math
import re
from typing import Literal

from pydantic import BaseModel, Field

from src.static_analyzer import (
    Language,
    StaticReport,
    _strip_cpp_comments_and_strings,
    analyze_code,
)

AlgoFamily = Literal[
    "math_ad_hoc",
    "prefix_sum_two_pointers",
    "sorting_binary_search",
    "graphs_bfs_dfs_dijkstra",
    "dynamic_programming",
    "data_structures_hash_segtree",
]

ALGO_FAMILIES: list[AlgoFamily] = [
    "math_ad_hoc",
    "prefix_sum_two_pointers",
    "sorting_binary_search",
    "graphs_bfs_dfs_dijkstra",
    "dynamic_programming",
    "data_structures_hash_segtree",
]

ALGO_FAMILY_LABELS_PT: dict[str, str] = {
    "math_ad_hoc": "Matematica / Ad-Hoc / Simulacao",
    "prefix_sum_two_pointers": "Soma de Prefixos / Two Pointers / Sliding Window",
    "sorting_binary_search": "Ordenacao / Busca Binaria / Greedy",
    "graphs_bfs_dfs_dijkstra": "Grafos / Grid (BFS, DFS, Dijkstra, Arvores)",
    "dynamic_programming": "Programacao Dinamica (DP / Memoizacao)",
    "data_structures_hash_segtree": "Estruturas de Dados (Map/Hash, Heap, SegTree/BIT)",
}

BugClass = Literal[
    "integer_overflow",
    "tle_quadratic_complexity",
    "off_by_one_bounds",
    "global_state_dirty",
    "tle_slow_io_endl",
    "infinite_loop_or_recursion",
    "edge_case_uninitialized_or_div0",
    "clean_or_minor_logic",
]

BUG_CLASSES: list[BugClass] = [
    "integer_overflow",
    "tle_quadratic_complexity",
    "off_by_one_bounds",
    "global_state_dirty",
    "tle_slow_io_endl",
    "infinite_loop_or_recursion",
    "edge_case_uninitialized_or_div0",
    "clean_or_minor_logic",
]

BUG_CLASS_LABELS_PT: dict[str, str] = {
    "integer_overflow": "Overflow de Inteiro 32-bits (int vs long long)",
    "tle_quadratic_complexity": "Time Limit Exceeded (Complexidade O(N^2)/O(N^3) vs Limite de N)",
    "off_by_one_bounds": "Out-of-Bounds / Off-by-One (i <= n ou indice fora da borda)",
    "global_state_dirty": "Contaminacao de Estado Global entre Casos de Teste (falta .clear()/memset)",
    "tle_slow_io_endl": "Time Limit Exceeded por I/O Lento (std::endl / flush constante em loop)",
    "infinite_loop_or_recursion": "Loop Infinito (Busca Binaria / While) ou Estouro de Recursao",
    "edge_case_uninitialized_or_div0": "Variavel Nao Inicializada / Caso de Borda (N=1, Negativos, Divisao/Modulo)",
    "clean_or_minor_logic": "Estrutura Compativel com os Limites (Sem Bug Estrutural Critico)",
}

FEATURE_NAMES: list[str] = [
    "lang_is_cpp",
    "lang_is_c",
    "lang_is_python",
    "max_n_log10",
    "max_val_log10",
    "has_multi_testcase_stmt",
    "time_limit_sec",
    "max_loop_depth",
    "estimated_ops_log10",
    "ops_exceed_time_limit",
    "uses_int_32",
    "uses_64bit_int",
    "has_accumulator_op",
    "potential_sum_log10",
    "overflow_risk_score",
    "has_le_in_zero_loop",
    "has_index_plus_one",
    "has_global_container",
    "has_multi_test_loop",
    "missing_global_cleanup",
    "has_endl_in_loop",
    "has_fast_io",
    "has_binary_search_pattern",
    "has_bs_mid_assign_bug",
    "has_recursion",
    "has_memoization",
    "has_uninitialized_local",
    "has_zero_max_init",
    "has_float_equality",
    "has_unordered_map",
    "has_sort_le_comparator",
    "has_mod_arithmetic",
    "code_line_count",
]


class ProblemConstraints(BaseModel):
    """Extracted mathematical constraints and expected complexity from problem statement."""

    max_n: int = 1000
    max_n_log10: float = 3.0
    max_val: int = 1000
    max_val_log10: float = 3.0
    time_limit_sec: float = 1.0
    has_multi_testcases: bool = False
    allows_negatives: bool = False
    expected_max_complexity: str = "O(N^2)"
    predicted_algo_family: str = "math_ad_hoc"
    predicted_algo_label: str = ALGO_FAMILY_LABELS_PT["math_ad_hoc"]
    algo_confidence: float = 0.0


def parse_problem_constraints(problem_statement: str) -> ProblemConstraints:
    """Extract N limit, element value limit, time limit, and negative/multi-testcase flags."""
    text = problem_statement or ""
    low = text.lower()

    # Remove common modulo constants and contest/problem headers like 'beecrowd | 2502' or 'CF-1350'
    cleaned = re.sub(r"10\s*[\^**]+\s*9\s*\+\s*7", " MOD_CONST ", low)
    cleaned = re.sub(r"1000000007|998244353", " MOD_CONST ", cleaned)
    cleaned = re.sub(r"\b(?:beecrowd|uri|cf|codeforces|atcoder|cses|problema|problem|questao)\s*[\|:\-#]?\s*\d{3,6}\b", " ", cleaned)

    powers_n: list[float] = []
    powers_val: list[float] = []

    # Explicit N/M/Q scientific notation e.g. N <= 2 * 10^5, n <= 1e5
    for m in re.finditer(
        r"\b([nmqk]|tamanho|vertices|arestas)\s*(?:<=|<|=|ate)\s*(?:(\d+(?:\.\d+)?)\s*[*x]\s*)?10\s*[\^**]+\s*(\d+)",
        cleaned,
    ):
        mult = float(m.group(2)) if m.group(2) else 1.0
        exp = float(m.group(3))
        powers_n.append(math.log10(max(mult, 1.0)) + exp)

    for m in re.finditer(
        r"\b([nmqk])\b[^.\n,;]{0,20}?(\d+(?:\.\d+)?)e(\d+)",
        cleaned,
    ):
        mult = float(m.group(2))
        exp = float(m.group(3))
        powers_n.append(math.log10(max(mult, 1.0)) + exp)

    # Explicit value limits e.g. a_i <= 10^9, valores ate 10^9, x <= 1e9
    for m in re.finditer(
        r"(?:a\[?i\]?|v\[?i\]?|x_?i|valor(?:es)?|elemento(?:s)?|peso(?:s)?|ate)\b[^.\n,;]{0,25}?(?:(\d+(?:\.\d+)?)\s*[*x]\s*)?10\s*[\^**]+\s*(\d+)",
        cleaned,
    ):
        mult = float(m.group(1)) if m.group(1) else 1.0
        exp = float(m.group(2))
        powers_val.append(math.log10(max(mult, 1.0)) + exp)

    # Generic powers of 10 in statement
    all_sci: list[float] = []
    for m in re.finditer(r"(?:(\d+(?:\.\d+)?)\s*[*x]\s*)?10\s*[\^**]+\s*(\d+)", cleaned):
        mult = float(m.group(1)) if m.group(1) else 1.0
        exp = float(m.group(2))
        all_sci.append(math.log10(max(mult, 1.0)) + exp)

    for m in re.finditer(r"\b(\d+(?:\.\d+)?)e(\d+)\b", cleaned):
        mult = float(m.group(1))
        exp = float(m.group(2))
        all_sci.append(math.log10(max(mult, 1.0)) + exp)

    # Plain integers like 100000, 200000, 1000000000
    for m in re.finditer(r"\b(\d{3,18})\b", cleaned):
        val = int(m.group(1))
        if val not in (1000000007, 998244353) and val >= 100:
            lg = math.log10(val)
            if lg <= 6.5:
                powers_n.append(lg)
            else:
                powers_val.append(lg)

    if not powers_n and all_sci:
        n_cands = [p for p in all_sci if p <= 7.0]
        powers_n = n_cands if n_cands else [min(all_sci)]

    if not powers_val and all_sci:
        v_cands = [p for p in all_sci if p >= 6.0]
        powers_val = v_cands if v_cands else all_sci

    raw_max_n = max(powers_n) if powers_n else 5.0
    raw_max_val = max(powers_val) if powers_val else raw_max_n

    max_n = int(round(10**raw_max_n))
    max_val = int(round(10**raw_max_val))
    max_n_log10 = round(raw_max_n, 2)
    max_val_log10 = round(raw_max_val, 2)

    tl_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|seg|sec|segundos)\b", low)
    time_limit_sec = float(tl_match.group(1)) if tl_match else 1.0

    has_multi_tc = bool(
        re.search(r"\bt\s*<=\s*|\bcasos de teste\b|\btest cases\b|\bmultiplos casos\b", low)
    )
    allows_neg = bool(re.search(r"negativ|-\s*10\s*\^|<=\s*0\b|-\d{3,}", low))

    if max_n_log10 >= 5.8:
        expected_comp = "O(N) ou O(log N)"
    elif max_n_log10 >= 4.3:
        expected_comp = "O(N log N) ou O(N)"
    elif max_n_log10 >= 3.0:
        expected_comp = "O(N^2) no maximo"
    elif max_n_log10 >= 2.0:
        expected_comp = "O(N^3) no maximo"
    else:
        expected_comp = "O(2^N) / Backtracking / DP"

    return ProblemConstraints(
        max_n=max_n,
        max_n_log10=max_n_log10,
        max_val=max_val,
        max_val_log10=max_val_log10,
        time_limit_sec=time_limit_sec,
        has_multi_testcases=has_multi_tc,
        allows_negatives=allows_neg,
        expected_max_complexity=expected_comp,
    )


def _detect_uninitialized_cpp_local(cleaned: str) -> tuple[bool, int | None, str | None]:
    """Check if C/C++ code declares local int/long long var without init and uses += on it."""
    lines = cleaned.splitlines()
    uninit_vars: dict[str, int] = {}
    inside_main_or_fn = False
    for idx, line in enumerate(lines, start=1):
        if re.search(r"\b(main|solve)\s*\(", line):
            inside_main_or_fn = True
        if inside_main_or_fn:
            m = re.search(r"^\s*(?:long\s+long|int)\s+([a-zA-Z_]\w*)\s*;", line)
            if m:
                vname = m.group(1)
                if vname not in ("n", "m", "t", "k", "q", "x", "u", "v", "a", "b", "c", "i", "j"):
                    uninit_vars[vname] = idx
            for vname, decl_line in list(uninit_vars.items()):
                if re.search(rf"(?:cin\s*>>[^;]*\b{vname}\b|scanf\s*\([^)]*&\s*{vname}\b|\b{vname}\s*=(?!=))", line):
                    uninit_vars.pop(vname, None)
                elif re.search(rf"\b{vname}\s*(\+=|\+\+|-=|\*=)", line):
                    return True, decl_line, vname
    return False, None, None


def _detect_bs_infinite_loop(code: str, language: Language) -> tuple[bool, int | None]:
    """Detect binary search pattern where mid rounds down and l = mid causes infinite loop."""
    lines = code.splitlines()
    has_while_lr = False
    mid_floor = False
    l_eq_mid_line: int | None = None
    for idx, line in enumerate(lines, start=1):
        if re.search(r"\bwhile\s*[\(\s].*\b(l|left|ini|low)\s*<=\s*(r|right|fim|high)|\bwhile\s*[\(\s].*\b(l|left|ini|low)\s*<\s*(r|right|fim|high)", line):
            has_while_lr = True
        if re.search(r"\bmid\s*=\s*\(\s*(l|left|ini|low)\s*\+\s*(r|right|fim|high)\s*\)\s*(//|/)\s*2", line):
            mid_floor = True
        if has_while_lr and re.search(r"^\s*(l|left|ini|low)\s*=\s*mid\s*(?:;|$)", line):
            l_eq_mid_line = idx
    return (has_while_lr and mid_floor and l_eq_mid_line is not None), l_eq_mid_line


def _detect_recursion_without_memo(code: str, language: Language) -> tuple[bool, bool]:
    """Return (has_recursion, has_memoization)."""
    has_memo = any(
        tok in code
        for tok in ("lru_cache", "cache", "dp[", "memo[", "visited[", "vis[", "dist[")
    )
    if language == "python":
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    fn_name = node.name
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                            if child.func.id == fn_name:
                                return True, has_memo
        except SyntaxError:
            pass
        return False, has_memo

    cleaned = _strip_cpp_comments_and_strings(code)
    for m in re.finditer(r"\b(?:int|long\s+long|void|bool)\s+([a-zA-Z_]\w*)\s*\([^)]*\)\s*\{", cleaned):
        fn_name = m.group(1)
        if fn_name == "main":
            continue
        body_start = m.end()
        body_slice = cleaned[body_start : body_start + 600]
        if re.search(rf"\b{fn_name}\s*\(", body_slice):
            return True, has_memo
    return False, has_memo


def _check_compile_or_syntax_error(code: str, language: Language) -> tuple[bool, int | None, str | None]:
    """Detect Python syntax errors or C/C++ compile errors (such as '#define int long long' + 'int main()')."""
    if language == "python":
        try:
            ast.parse(code)
            return False, None, None
        except SyntaxError as exc:
            ln = exc.lineno or 1
            return True, ln, f"Erro de Sintaxe Python na linha {ln}: {exc.msg}"

    # Classic C++ competitive programming compile error: #define int long long with int main()
    has_define_int_ll = False
    for idx, raw_line in enumerate(code.splitlines(), start=1):
        if re.search(r"#\s*define\s+int\s+long\s+long", raw_line):
            has_define_int_ll = True
        if has_define_int_ll and re.search(r"\bint\s+main\s*\(", raw_line):
            return (
                True,
                idx,
                f"Linha {idx}: Erro de Compilacao C++ ('::main must return int'). Com '#define int long long' ativo, "
                "'int main()' vira 'long long main()', que o compilador G++ rejeita (use 'signed main()' ou 'int32_t main()').",
            )

    cleaned = _strip_cpp_comments_and_strings(code)
    brace_stack: list[int] = []
    paren_stack: list[int] = []
    for idx, line in enumerate(cleaned.splitlines(), start=1):
        for ch in line:
            if ch == "{":
                brace_stack.append(idx)
            elif ch == "}":
                if not brace_stack:
                    return True, idx, f"Erro de Compilacao C/C++ na linha {idx}: chave '}}' fechada sem abertura correspondente."
                brace_stack.pop()
            elif ch == "(":
                paren_stack.append(idx)
            elif ch == ")":
                if not paren_stack:
                    return True, idx, f"Erro de Compilacao C/C++ na linha {idx}: parentese ')' fechado sem abertura correspondente."
                paren_stack.pop()
    if brace_stack:
        ln = brace_stack[-1]
        return True, ln, f"Erro de Compilacao C/C++ na linha {ln}: bloco '{{' aberto e nao fechado com '}}'."
    if paren_stack:
        ln = paren_stack[-1]
        return True, ln, f"Erro de Compilacao C/C++ na linha {ln}: parentese '(' aberto e nao fechado com ')'."
    return False, None, None


class CodeFeatureAnalysis(BaseModel):
    """Rich static + AST + constraint alignment analysis for ML inference and explanation."""

    constraints: ProblemConstraints
    static_report: StaticReport
    features: list[float] = Field(default_factory=list)
    feature_Map: dict[str, float] = Field(default_factory=dict)
    suspect_lines: dict[str, int] = Field(default_factory=dict)
    suspect_vars: dict[str, str] = Field(default_factory=dict)
    has_compile_error: bool = False
    compile_error_line: int | None = None
    compile_error_msg: str | None = None


def extract_all_features(
    code: str,
    language: Language,
    problem_statement: str = "",
) -> CodeFeatureAnalysis:
    """Extract the 33-dimensional numeric feature vector + suspect lines/variables."""
    constraints = parse_problem_constraints(problem_statement)
    static_rep = analyze_code(code, language, problem_statement)
    cleaned = _strip_cpp_comments_and_strings(code) if language in ("cpp", "c") else code
    lines = cleaned.splitlines()

    lang_is_cpp = 1.0 if language == "cpp" else 0.0
    lang_is_c = 1.0 if language == "c" else 0.0
    lang_is_python = 1.0 if language == "python" else 0.0

    max_depth = float(static_rep.max_loop_depth)
    est_ops_log10 = constraints.max_n_log10 * max(max_depth, 1.0)
    # Python is ~10x slower per op than C++
    ops_threshold = 7.2 if language == "python" else 8.2
    ops_exceed = 1.0 if (max_depth >= 2 and est_ops_log10 > ops_threshold) else 0.0

    # Integer overflow detection (C/C++)
    uses_int_32 = 0.0
    uses_64bit = 0.0
    has_acc_op = 0.0
    suspect_lines: dict[str, int] = {}
    suspect_vars: dict[str, str] = {}

    if language in ("cpp", "c"):
        has_define_int_ll = bool(re.search(r"#\s*define\s+int\s+long\s+long", cleaned))
        if "long long" in cleaned or "int64_t" in cleaned or has_define_int_ll:
            uses_64bit = 1.0
        for idx, line in enumerate(lines, start=1):
            m_decl = re.search(r"(?<!long\s)\bint\s+([a-zA-Z_]\w*)\s*=\s*0", line)
            if m_decl and not has_define_int_ll:
                uses_int_32 = 1.0
                suspect_vars.setdefault("overflow_var", m_decl.group(1))
                suspect_lines.setdefault("integer_overflow", idx)
            if re.search(r"(\+=|\*=|\baccumulate\b|\*\s*[a-zA-Z_]\w*)", line):
                has_acc_op = 1.0
                if "integer_overflow" not in suspect_lines and uses_int_32 > 0:
                    suspect_lines["integer_overflow"] = idx
        if not uses_int_32 and not has_define_int_ll and re.search(r"(?<!long\s)\bint\s+\w+", cleaned):
            uses_int_32 = 1.0

    potential_sum_log10 = constraints.max_n_log10 + constraints.max_val_log10
    if has_acc_op and language in ("cpp", "c") and uses_int_32 and not uses_64bit:
        overflow_risk = 1.0 if (potential_sum_log10 >= 9.3 or constraints.max_val_log10 >= 9.0) else 0.4
    else:
        overflow_risk = 0.0

    # Off-by-one / Out-of-bounds
    has_le_zero = 0.0
    has_idx_plus_one = 0.0
    for idx, line in enumerate(lines, start=1):
        if language in ("cpp", "c") and re.search(r"for\s*\([^;]*=\s*0\s*;[^;]*<=\s*\w+", line):
            has_le_zero = 1.0
            suspect_lines.setdefault("off_by_one_bounds", idx)
        elif language == "python" and (
            re.search(r"range\s*\([^)]*len\([^)]+\)\s*\+\s*1\)", line)
            or re.search(r"<=\s*len\(", line)
        ):
            has_le_zero = 1.0
            suspect_lines.setdefault("off_by_one_bounds", idx)
        if re.search(r"\[\s*[i|j]\s*\+\s*1\s*\]|\[\s*[i|j]\s*-\s*1\s*\]", line):
            has_idx_plus_one = 1.0
            suspect_lines.setdefault("off_by_one_bounds", idx)

    # Nested loop line tracking
    for idx, line in enumerate(lines, start=1):
        if re.search(r"\b(for|while)\b", line):
            suspect_lines.setdefault("tle_quadratic_complexity", idx)

    # Global container + multi testcase
    has_global_cont = 1.0 if bool(
        re.search(r"^\s*(?:vector\s*<[^>]+>|int|long\s+long)\s+\w+\s*\[", cleaned, re.MULTILINE)
    ) else 0.0
    has_multi_loop = 1.0 if bool(
        re.search(r"while\s*\(\s*t\s*--\s*\)|for\s*\([^)]*;\s*\w+\s*<\s*t\s*;", cleaned, re.IGNORECASE)
    ) else 0.0
    has_cleanup = ".clear()" in cleaned or "memset(" in cleaned
    missing_cleanup = 1.0 if (has_global_cont and (has_multi_loop or constraints.has_multi_testcases) and not has_cleanup) else 0.0
    if missing_cleanup:
        for idx, line in enumerate(lines, start=1):
            if re.search(r"vector\s*<|while\s*\(\s*t\s*--", line):
                suspect_lines.setdefault("global_state_dirty", idx)

    # Slow I/O endl inside loop
    has_endl_loop = 0.0
    in_loop = False
    for idx, line in enumerate(lines, start=1):
        if re.search(r"\b(for|while)\b", line):
            in_loop = True
        if in_loop and "endl" in line:
            has_endl_loop = 1.0
            suspect_lines.setdefault("tle_slow_io_endl", idx)
            break

    has_fast_io = 1.0 if ("sync_with_stdio" in cleaned or "sys.stdin" in cleaned) else 0.0

    # Binary search / infinite loop / recursion
    has_bs = 1.0 if bool(re.search(r"\bmid\b|\blower_bound\b|\bupper_bound\b|\bbisect\b", cleaned)) else 0.0
    bs_bug, bs_line = _detect_bs_infinite_loop(cleaned, language)
    if bs_bug and bs_line:
        suspect_lines.setdefault("infinite_loop_or_recursion", bs_line)

    has_rec, has_memo = _detect_recursion_without_memo(code, language)
    if has_rec and not has_memo:
        for idx, line in enumerate(lines, start=1):
            if re.search(r"\b(def|void|int|long\s+long)\b", line):
                suspect_lines.setdefault("infinite_loop_or_recursion", idx)

    # Uninitialized local or max_init = 0 with negatives or float ==
    uninit_flag, uninit_line, uninit_var = _detect_uninitialized_cpp_local(cleaned) if language in ("cpp", "c") else (False, None, None)
    if uninit_flag and uninit_line:
        suspect_lines.setdefault("edge_case_uninitialized_or_div0", uninit_line)
        if uninit_var:
            suspect_vars["uninit_var"] = uninit_var

    has_zero_max = 0.0
    has_max_comparison = bool(
        re.search(r"\b(max|std::max)\s*\(|\bif\s*\([^)]*>\s*(mx|max_val|maior|ans|best)\b", cleaned)
    )
    if has_max_comparison:
        for idx, line in enumerate(lines, start=1):
            if re.search(r"\b(mx|max_val|maior|ans|best)\s*=\s*0\b", line):
                has_zero_max = 1.0
                suspect_lines.setdefault("edge_case_uninitialized_or_div0", idx)

    has_float_eq = 1.0 if bool(re.search(r"\b(double|float)\b", cleaned) and re.search(r"==\s*\d", cleaned)) else 0.0
    has_umap = 1.0 if "unordered_map" in cleaned else 0.0
    has_sort_le = 1.0 if bool(re.search(r"sort\s*\(", cleaned) and re.search(r"return\s+[^;]*<=", cleaned)) else 0.0
    has_mod = 1.0 if ("%" in cleaned or "1000000007" in cleaned or "998244353" in cleaned) else 0.0

    fmap: dict[str, float] = {
        "lang_is_cpp": lang_is_cpp,
        "lang_is_c": lang_is_c,
        "lang_is_python": lang_is_python,
        "max_n_log10": constraints.max_n_log10,
        "max_val_log10": constraints.max_val_log10,
        "has_multi_testcase_stmt": 1.0 if constraints.has_multi_testcases else 0.0,
        "time_limit_sec": constraints.time_limit_sec,
        "max_loop_depth": max_depth,
        "estimated_ops_log10": round(est_ops_log10, 2),
        "ops_exceed_time_limit": ops_exceed,
        "uses_int_32": uses_int_32,
        "uses_64bit_int": uses_64bit,
        "has_accumulator_op": has_acc_op,
        "potential_sum_log10": round(potential_sum_log10, 2),
        "overflow_risk_score": overflow_risk,
        "has_le_in_zero_loop": has_le_zero,
        "has_index_plus_one": has_idx_plus_one,
        "has_global_container": has_global_cont,
        "has_multi_test_loop": has_multi_loop,
        "missing_global_cleanup": missing_cleanup,
        "has_endl_in_loop": has_endl_loop,
        "has_fast_io": has_fast_io,
        "has_binary_search_pattern": has_bs,
        "has_bs_mid_assign_bug": 1.0 if bs_bug else 0.0,
        "has_recursion": 1.0 if has_rec else 0.0,
        "has_memoization": 1.0 if has_memo else 0.0,
        "has_uninitialized_local": 1.0 if uninit_flag else 0.0,
        "has_zero_max_init": has_zero_max,
        "has_float_equality": has_float_eq,
        "has_unordered_map": has_umap,
        "has_sort_le_comparator": has_sort_le,
        "has_mod_arithmetic": has_mod,
        "code_line_count": float(max(len(lines), 1)),
    }

    vec = [fmap[k] for k in FEATURE_NAMES]
    has_comp_err, comp_ln, comp_msg = _check_compile_or_syntax_error(code, language)
    if has_comp_err and comp_ln:
        suspect_lines["compile_or_syntax_error"] = comp_ln

    return CodeFeatureAnalysis(
        constraints=constraints,
        static_report=static_rep,
        features=vec,
        feature_Map=fmap,
        suspect_lines=suspect_lines,
        suspect_vars=suspect_vars,
        has_compile_error=has_comp_err,
        compile_error_line=comp_ln,
        compile_error_msg=comp_msg,
    )
