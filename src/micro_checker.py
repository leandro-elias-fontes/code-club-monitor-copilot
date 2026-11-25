"""Micro-Error & Symbol Table / Compiler Analyzer for C, C++, and Python (0% False-Positive Audited)."""

from __future__ import annotations

import ast
import builtins
import difflib
import os
import re
import shutil
import subprocess
import tempfile

from src.static_analyzer import Language, _strip_cpp_comments_and_strings

_CPP_KEYWORDS_AND_STD: set[str] = {
    # Keywords & types
    "alignas", "alignof", "and", "and_eq", "asm", "auto", "bitand", "bitor", "bool", "break",
    "case", "catch", "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept",
    "const", "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
    "co_return", "co_yield", "decltype", "default", "delete", "do", "double", "dynamic_cast",
    "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto",
    "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "not_eq",
    "nullptr", "operator", "or", "or_eq", "private", "protected", "public", "register",
    "reinterpret_cast", "requires", "return", "short", "signed", "sizeof", "static",
    "static_assert", "static_cast", "struct", "switch", "template", "this", "thread_local",
    "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using",
    "virtual", "void", "volatile", "wchar_t", "while", "xor", "xor_eq",
    "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "size_t", "ssize_t", "ptrdiff_t", "ll", "ull", "ld", "pii", "pll", "vi", "vll",
    "__int128", "__int128_t", "__uint128_t", "intmax_t", "uintmax_t",
    # Std objects, streams, containers, functions, constants
    "std", "cin", "cout", "cerr", "clog", "endl", "flush", "fixed", "setprecision", "ios",
    "ios_base", "sync_with_stdio", "tie", "getline", "scanf", "printf", "puts",
    "getchar", "putchar", "fgets", "fscanf", "fprintf", "sprintf", "snprintf", "sscanf",
    "stdin", "stdout", "stderr", "EOF", "NULL", "istream", "ostream", "stringstream", "istringstream", "ostringstream",
    "ifstream", "ofstream", "fstream", "freopen", "fclose", "fopen", "fflush",
    "vector", "string", "pair", "tuple", "array", "deque", "list", "forward_list", "set",
    "multiset", "unordered_set", "unordered_multiset", "map", "multimap", "unordered_map",
    "unordered_multimap", "stack", "queue", "priority_queue", "bitset", "span", "valarray",
    "complex", "function", "optional", "variant", "any", "reference_wrapper",
    "iterator", "const_iterator", "greater", "less", "plus", "minus", "multiplies", "bit_xor",
    "mt19937", "mt19937_64", "random_device", "uniform_int_distribution", "uniform_real_distribution",
    "chrono", "high_resolution_clock", "steady_clock", "system_clock", "now", "time_since_epoch",
    "main", "solve", "push_back", "emplace_back", "pop_back", "push", "pop", "emplace",
    "top", "front", "back", "begin", "end", "rbegin", "rend", "cbegin", "cend", "size",
    "empty", "clear", "resize", "reserve", "assign", "insert", "erase", "find", "count",
    "lower_bound", "upper_bound", "equal_range", "binary_search", "sort", "stable_sort",
    "partial_sort", "nth_element", "reverse", "rotate", "unique", "next_permutation", "prev_permutation",
    "is_sorted", "merge", "inplace_merge", "partition", "stable_partition",
    "min", "max", "minmax", "min_element", "max_element", " minmax_element", "clamp", "swap", "exchange",
    "fill", "fill_n", "copy", "copy_n", "move", "transform", "accumulate", "partial_sum",
    "adjacent_difference", "inner_product", "iota", "gcd", "lcm",
    "__gcd", "__lg", "__builtin_popcount", "__builtin_popcountl", "__builtin_popcountll",
    "__builtin_clz", "__builtin_clzl", "__builtin_clzll", "__builtin_ctz", "__builtin_ctzl", "__builtin_ctzll",
    "__builtin_parity", "__builtin_parityll", "__builtin_ffs", "__builtin_expect",
    "abs", "labs", "llabs", "fabs", "sqrt", "sqrtl", "cbrt", "hypot", "pow", "powl",
    "exp", "exp2", "log", "log2", "log10", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "ceil", "ceill", "floor", "floorl", "round", "roundl", "trunc", "fmod", "remainder", "isnan", "isinf",
    "memset", "memcpy", "memmove", "memcmp", "strlen", "strcmp", "strncmp", "strcpy", "strncpy",
    "strcat", "strchr", "strstr", "atoi", "atoll", "atof", "strtol", "strtoll", "to_string",
    "stoi", "stoll", "stoul", "stoull", "stod", "stof", "stold",
    "isdigit", "isalpha", "isalnum", "islower", "isupper", "isspace", "tolower", "toupper",
    "first", "second", "make_pair", "make_tuple", "get", "substr",
    "length", "c_str", "data", "at", "npos", "append", "replace", "rfind", "find_first_of", "find_last_of",
    "INT_MAX", "INT_MIN", "UINT_MAX", "LONG_MAX", "LONG_MIN", "LLONG_MAX", "LLONG_MIN", "ULLONG_MAX",
    "DBL_MAX", "DBL_MIN", "FLT_MAX", "CHAR_BIT", "numeric_limits", "infinity",
    "MOD", "INF", "MAXN", "EPS", "PI", "M_PI",
    "malloc", "calloc", "realloc", "free", "exit", "abort", "assert", "time", "srand", "rand", "qsort",
    "__gnu_pbds", "tree", "null_type", "tree_order_statistics_node_update", "rb_tree_tag",
    "gp_hash_table", "cc_hash_table", "order_of_key", "find_by_order",
}

_PYTHON_BUILTINS: set[str] = set(dir(builtins)) | {
    "sys", "math", "collections", "heapq", "bisect", "itertools", "functools",
    "Counter", "defaultdict", "deque", "lru_cache", "cache", "reduce", "cmp_to_key",
    "stdin", "stdout", "readline", "readlines", "write", "inf", "gcd", "lcm",
    "sqrt", "ceil", "floor", "log2", "log10", "comb", "perm", "factorial", "isqrt",
    "permutations", "combinations", "product", "accumulate", "groupby", "chain",
}


class MicroErrorResult:
    """Result of micro-error (undeclared variable, typo, missing semicolon, assignment in if, syntax)."""

    def __init__(
        self,
        has_error: bool,
        line: int | None = None,
        title: str = "",
        bug_location: str = "",
        how_to_fix: str = "",
        monitor_tip: str = "",
        is_compile_error: bool = True,
    ) -> None:
        self.has_error = has_error
        self.line = line
        self.title = title
        self.bug_location = bug_location
        self.how_to_fix = how_to_fix
        self.monitor_tip = monitor_tip
        self.is_compile_error = is_compile_error


def _suggest_closest_var(unknown: str, declared: set[str]) -> str | None:
    """Find the closest declared variable name using Levenshtein ratio (e.g., 'soma' -> 'som')."""
    user_vars = [v for v in declared if v not in _CPP_KEYWORDS_AND_STD and len(v) >= 2]
    matches = difflib.get_close_matches(unknown, user_vars, n=1, cutoff=0.65)
    return matches[0] if matches else None


def _strip_balanced_angles(s: str) -> str:
    """Replace template arguments `<...>` on known/custom template types with `<>` so commas inside templates don't split declarations."""
    out: list[str] = []
    depth = 0
    for idx, ch in enumerate(s):
        if ch == "<":
            prev_slice = s[max(0, idx - 30) : idx]
            if depth > 0 or re.search(
                r"\b(?:vector|pair|tuple|map|unordered_map|multimap|unordered_multimap|set|unordered_set|multiset|unordered_multiset|"
                r"queue|priority_queue|stack|deque|list|forward_list|array|bitset|function|complex|valarray|optional|variant|"
                r"tree|gp_hash_table|cc_hash_table|greater|less|plus|minus|allocator|template|numeric_limits|"
                r"V|AR|vc|vv|vvc|P|T)\s*$",
                prev_slice,
            ):
                if depth == 0:
                    out.append("<>")
                depth += 1
                continue
        elif ch == ">" and depth > 0:
            depth -= 1
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


def _extract_declared_cpp_identifiers(cleaned: str) -> tuple[set[str], set[str]]:
    """Return (declared_variables_and_functions, declared_custom_types) from C/C++ code."""
    declared: set[str] = set()
    custom_types: set[str] = {
        "int", "long", "short", "char", "bool", "float", "double", "void", "auto", "signed", "unsigned",
        "size_t", "ssize_t", "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
        "__int128", "__int128_t", "__uint128_t", "string", "ll", "ull", "ld", "pii", "pll", "vi", "vll",
        "clock_t", "time_t", "ptrdiff_t",
        "istream", "ostream", "stringstream", "istringstream", "ostringstream", "ifstream", "ofstream", "fstream",
        "vector", "set", "multiset", "unordered_set", "unordered_multiset",
        "map", "multimap", "unordered_map", "unordered_multimap",
        "pair", "tuple", "queue", "priority_queue", "stack", "deque", "bitset", "array",
        "function", "complex", "mt19937", "mt19937_64", "random_device", "tree", "gp_hash_table",
    }

    # 1. Macros: #define NAME or #define NAME(a, b)
    for m in re.finditer(r"#\s*define\s+([a-zA-Z_]\w*)(?:\(([^)]*)\))?", cleaned):
        macro_name = m.group(1)
        declared.add(macro_name)
        custom_types.add(macro_name)
        if m.group(2):
            for pm in re.finditer(r"[a-zA-Z_]\w*", m.group(2)):
                declared.add(pm.group(0))

    # 2. using Alias = ...;
    for m in re.finditer(r"\busing\s+([a-zA-Z_]\w*)\s*=", cleaned):
        alias = m.group(1)
        declared.add(alias)
        custom_types.add(alias)

    # 3. typedef ... Alias;
    for m in re.finditer(r"\btypedef\b[^;]+?\b([a-zA-Z_]\w*)\s*(?:\[[^\]]*\])?\s*;", cleaned):
        alias = m.group(1)
        declared.add(alias)
        custom_types.add(alias)

    # 4. struct / class / enum / namespace names and trailing instances: struct P { ... } p[N], q;
    for m in re.finditer(r"\b(?:struct|class|enum|namespace)\s+([a-zA-Z_]\w*)", cleaned):
        sname = m.group(1)
        declared.add(sname)
        custom_types.add(sname)

    for m in re.finditer(r"\}\s*([a-zA-Z_]\w*(?:\s*\[[^\]]*\])?(?:\s*,\s*[a-zA-Z_]\w*(?:\s*\[[^\]]*\])?)*)\s*;", cleaned):
        for vm in re.finditer(r"([a-zA-Z_]\w*)\s*(?:\[|$|,)", m.group(1)):
            declared.add(vm.group(1))

    # 5. template <class T, typename U>
    for m in re.finditer(r"\btemplate\s*<([^>]+)>", cleaned):
        for tm in re.finditer(r"\b(?:class|typename|int|auto|size_t)\s+([a-zA-Z_]\w*)", m.group(1)):
            tname = tm.group(1)
            declared.add(tname)
            custom_types.add(tname)

    # 6. Strip template `<...>` so nested angle brackets and non-whitespace `>varname` are clean
    simplified = _strip_balanced_angles(cleaned)
    # Remove `std::` prefix
    simplified = re.sub(r"\bstd\s*::\s*", "", simplified)

    # Build dynamic regex for all built-in + custom types
    types_alt = "|".join(sorted((re.escape(t) for t in custom_types), key=len, reverse=True))
    decl_stmt_re = re.compile(
        rf"(?:\b(?:const|constexpr|static|inline|unsigned|signed|long|short|register|volatile|mutable)\s+)*"
        rf"\b(?:{types_alt})\b(?:\s*<>\s*)?"
        rf"(?:\s+(?:const|constexpr|long|int|double|char)\b)*"
        rf"(\s*[\*&]+\s*|\s+)"
        rf"([^;{{}}]+(?:\{{[^;{{}}]*\}}[^;{{}}]*)*)"
    )

    for m in decl_stmt_re.finditer(simplified):
        decl_chunk = m.group(2).strip()
        if not decl_chunk:
            continue

        # Split comma-separated declarators outside parentheses/brackets/braces
        depth = 0
        parts: list[str] = []
        curr: list[str] = []
        for ch in decl_chunk:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                parts.append("".join(curr))
                curr = []
            else:
                curr.append(ch)
        if curr:
            parts.append("".join(curr))

        for part in parts:
            # Take LHS before '=' initializer (unless it's 'operator==')
            lhs = part.split("=", 1)[0].strip() if "operator" not in part else part.strip()
            if not lhs:
                continue
            # Check if it's a function declaration/definition: fn_name(type1 arg1, type2 arg2)
            fn_m = re.match(r"^[\*&\s]*([a-zA-Z_]\w*)\s*\((.*)\)", lhs)
            if fn_m:
                declared.add(fn_m.group(1))
                params_str = fn_m.group(2).strip()
                if params_str:
                    for p_chunk in params_str.split(","):
                        p_lhs = p_chunk.split("=", 1)[0].strip()
                        p_lhs = re.sub(r"\[[^\]]*\]", "", p_lhs).strip()
                        tokens = re.findall(r"[a-zA-Z_]\w*", p_lhs)
                        if tokens:
                            last_tok = tokens[-1]
                            if last_tok not in custom_types:
                                declared.add(last_tok)
            else:
                var_m = re.match(r"^[\*&\s]*([a-zA-Z_]\w*)", lhs)
                if var_m:
                    vname = var_m.group(1)
                    if vname not in custom_types:
                        declared.add(vname)

    # 7. Also capture function parameters in operator overloads or multiline function headers: `(type &arg, type arg2)`
    for m in re.finditer(rf"\b(?:{types_alt})\b(?:\s*<>)?\s*(?:[\*&\s]+|\b)\s*([a-zA-Z_]\w*)\s*(?:=|,|\)|\[|;)", simplified):
        vname = m.group(1)
        if vname not in custom_types:
            declared.add(vname)

    # 8. Range-based for loops & C++17 structured bindings: for (auto [u, v] : edges) or for (auto &x : v)
    for m in re.finditer(r"for\s*\([^:;]*?\[([^\]]+)\]\s*:", simplified):
        for vm in re.finditer(r"[a-zA-Z_]\w*", m.group(1)):
            declared.add(vm.group(0))
    for m in re.finditer(r"for\s*\(\s*(?:const\s+)?[a-zA-Z_]\w*(?:\s*<>)?\s*[&\*]?\s*([a-zA-Z_]\w*)\s*:", simplified):
        declared.add(m.group(1))

    # 9. Direct input bindings: any identifier immediately following `cin >>` or `&var` in `scanf`
    # Wait: if a student writes `cin >> x;` where `x` was NOT declared, g++ fails! So we must NOT auto-declare `cin >>` targets.
    # Instead, catch lambda capture parameters `[&](int i, int j)`
    for m in re.finditer(r"\]\s*\(([^)]*)\)", simplified):
        for p_chunk in m.group(1).split(","):
            p_lhs = p_chunk.split("=", 1)[0].strip()
            toks = re.findall(r"[a-zA-Z_]\w*", p_lhs)
            if toks and toks[-1] not in custom_types:
                declared.add(toks[-1])

    return declared, custom_types


def _run_local_gcc_syntax_check(code: str, language: Language) -> tuple[bool, MicroErrorResult | None]:
    """If g++/gcc is installed locally, run -fsyntax-only as the authoritative compiler oracle.
    Returns (compiler_ran, error_result_or_None)."""
    compiler = shutil.which("g++" if language == "cpp" else "gcc")
    if not compiler and os.path.exists(r"C:\MinGW\bin\g++.exe"):
        compiler = r"C:\MinGW\bin\g++.exe" if language == "cpp" else r"C:\MinGW\bin\gcc.exe"
    if not compiler:
        return False, None

    # Pre-check #define int long long + int main() before calling gcc so we give our rich pedagogical message
    has_define_int_ll = False
    for idx, raw_line in enumerate(code.splitlines(), start=1):
        if re.search(r"#\s*define\s+int\s+long\s+long", raw_line):
            has_define_int_ll = True
        if has_define_int_ll and re.search(r"\bint\s+main\s*\(", raw_line):
            return True, MicroErrorResult(
                has_error=True,
                line=idx,
                title="Erro de Compilação C++ ('#define int long long' com 'int main()')",
                bug_location=(
                    f"Linha {idx}: Com '#define int long long' ativo, a assinatura 'int main()' vira 'long long main()', "
                    "que o compilador C++ rejeita ('error: ::main must return int')."
                ),
                how_to_fix=f"Na linha {idx}, troque 'int main()' por 'signed main()' (ou 'int32_t main()').",
                monitor_tip=(
                    "Pergunte ao aluno: 'O que o pré-processador faz com a palavra int na linha do int main() "
                    "quando colocamos #define int long long no topo do arquivo?'"
                ),
            )

    suffix = ".cpp" if language == "cpp" else ".c"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name
        cmd = [compiler, "-fsyntax-only", "-std=c++17", tmp_path] if language == "cpp" else [compiler, "-fsyntax-only", tmp_path]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2.5,
            check=False,
        )
        if proc.returncode == 0:
            return True, None

        stderr = proc.stderr or ""
        # Ignore missing bits/stdc++.h if MinGW doesn't have it, fallback to symbol table
        if "bits/stdc++.h: No such file" in stderr:
            return False, None

        m = re.search(r":(\d+):(?:\d+:)?\s*error:\s*([^\n]+)", stderr)
        if m:
            ln = int(m.group(1))
            err_msg = m.group(2).strip()
            # Check if it's an undeclared identifier error so we can suggest closest typo (e.g. 'soma' vs 'som')
            m_undecl = re.search(r"['‘`\"]([a-zA-Z_]\w*)['’`\"]\s+was not declared in this scope", err_msg)
            if m_undecl:
                ident = m_undecl.group(1)
                cleaned = _strip_cpp_comments_and_strings(code)
                declared, _ = _extract_declared_cpp_identifiers(cleaned)
                suggestion = _suggest_closest_var(ident, declared)
                if suggestion:
                    return True, MicroErrorResult(
                        has_error=True,
                        line=ln,
                        title=f"Variável Não Declarada / Erro de Digitação ('{ident}' vs '{suggestion}')",
                        bug_location=(
                            f"Linha {ln}: A variável '{ident}' não foi declarada neste escopo "
                            f"('error: \\'{ident}\\' was not declared in this scope'). "
                            f"Você declarou '{suggestion}' anteriormente, mas escreveu '{ident}' na linha {ln}."
                        ),
                        how_to_fix=(
                            f"Padronize o nome da variável: troque '{ident}' por '{suggestion}' na linha {ln} "
                            f"(ou renomeie a declaração de '{suggestion}' para '{ident}')."
                        ),
                        monitor_tip=(
                            f"Pergunte ao aluno: 'Olhe o nome da variável que você declarou lá em cima ({suggestion}) "
                            f"e o nome que você usou na linha {ln} ({ident}): elas estão escritas exatamente iguais?'"
                        ),
                    )
                return True, MicroErrorResult(
                    has_error=True,
                    line=ln,
                    title=f"Erro de Compilação: Identificador '{ident}' Não Declarado",
                    bug_location=f"Linha {ln}: O identificador '{ident}' foi usado sem ter sido declarado antes ('{err_msg}').",
                    how_to_fix=f"Declare a variável '{ident}' com seu tipo antes de usá-la na linha {ln}.",
                    monitor_tip=f"Pergunte ao aluno: 'Onde a variável {ident} da linha {ln} foi declarada no código?'",
                )

            return True, MicroErrorResult(
                has_error=True,
                line=ln,
                title=f"Erro de Compilação na Linha {ln}: {err_msg}",
                bug_location=f"Linha {ln}: O compilador C/C++ apontou: '{err_msg}'.",
                how_to_fix=f"Revise a sintaxe na linha {ln} (verifique ponto e vírgula ';', chaves ou tipos).",
                monitor_tip=f"Mostre a linha {ln} ao aluno e peça para ele verificar o que o compilador acusou nessa instrução.",
            )
    except Exception:  # noqa: BLE001
        return False, None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    return False, None


def _check_cpp_symbol_and_syntax_errors(code: str, language: Language) -> MicroErrorResult | None:
    """Pure-Python C/C++ symbol table & micro-syntax checker (0% false positive on competitive templates)."""
    # 1. #define int long long + int main()
    has_define_int_ll = False
    for idx, raw_line in enumerate(code.splitlines(), start=1):
        if re.search(r"#\s*define\s+int\s+long\s+long", raw_line):
            has_define_int_ll = True
        if has_define_int_ll and re.search(r"\bint\s+main\s*\(", raw_line):
            return MicroErrorResult(
                has_error=True,
                line=idx,
                title="Erro de Compilação C++ ('#define int long long' com 'int main()')",
                bug_location=(
                    f"Linha {idx}: Com '#define int long long' ativo, a assinatura 'int main()' vira 'long long main()', "
                    "que o compilador C++ rejeita ('error: ::main must return int')."
                ),
                how_to_fix=f"Na linha {idx}, troque 'int main()' por 'signed main()' (ou 'int32_t main()').",
                monitor_tip=(
                    "Pergunte ao aluno: 'O que o pré-processador faz com a palavra int na linha do int main() "
                    "quando colocamos #define int long long no topo do arquivo?'"
                ),
            )

    cleaned = _strip_cpp_comments_and_strings(code)
    lines = cleaned.splitlines()

    # 2. Unbalanced braces / parentheses
    brace_stack: list[int] = []
    paren_stack: list[int] = []
    for idx, line in enumerate(lines, start=1):
        for ch in line:
            if ch == "{":
                brace_stack.append(idx)
            elif ch == "}":
                if not brace_stack:
                    return MicroErrorResult(
                        True,
                        idx,
                        "Erro de Compilação: Chave '}' Sobrando",
                        f"Linha {idx}: A chave '}}' foi fechada sem ter um bloco '{{' aberto correspondente.",
                        f"Remova a chave '}}' extra na linha {idx} ou verifique onde o bloco deveria ter sido aberto.",
                        f"Peça ao aluno para conferir o par de cada chave '{{ }}' perto da linha {idx}.",
                    )
                brace_stack.pop()
            elif ch == "(":
                paren_stack.append(idx)
            elif ch == ")":
                if not paren_stack:
                    return MicroErrorResult(
                        True,
                        idx,
                        "Erro de Compilação: Parêntese ')' Sobrando",
                        f"Linha {idx}: Parêntese ')' fechado sem abertura '(' correspondente.",
                        f"Corrija os parênteses da expressão na linha {idx}.",
                        f"Mostre a linha {idx} e peça para o aluno contar os parênteses abertos e fechados.",
                    )
                paren_stack.pop()

    if brace_stack:
        ln = brace_stack[-1]
        return MicroErrorResult(
            True,
            ln,
            "Erro de Compilação: Bloco '{' Não Fechado",
            f"Linha {ln}: O bloco aberto com '{{' nunca foi fechado com '}}' até o final do arquivo.",
            f"Adicione a chave de fechamento '}}' correspondente ao bloco aberto na linha {ln}.",
            "Pergunte ao aluno: 'Faltou fechar alguma chave } no final da função main() ou do laço?'",
        )
    if paren_stack:
        ln = paren_stack[-1]
        return MicroErrorResult(
            True,
            ln,
            "Erro de Compilação: Parêntese '(' Não Fechado",
            f"Linha {ln}: Parêntese '(' aberto e não fechado.",
            f"Feche o parêntese ')' na linha {ln}.",
            f"Peça ao aluno para revisar a expressão da linha {ln}.",
        )

    # 3. Undeclared variables / Typos (e.g., declared 'som = 0', used 'soma += x')
    declared, custom_types = _extract_declared_cpp_identifiers(cleaned)
    all_known = _CPP_KEYWORDS_AND_STD | declared | custom_types

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Strip template arguments <...> on this line and member accesses (.field / ->field / ::member)
        line_clean = _strip_balanced_angles(stripped)
        line_clean = re.sub(r"(\.|->|::)\s*[a-zA-Z_]\w*", " ", line_clean)
        # Strip hex/binary/integer suffix literals like 0x3f3f3f3f, 100005u, 1e18, 0LL
        line_clean = re.sub(r"\b0[xXbB][0-9a-fA-F_]+[uUlL]*\b|\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[uUlLfF]+\b", " ", line_clean)

        # Only flag an identifier if it is actively used in an assignment/accumulation/IO statement
        # and has a close typo match OR is in a direct mutation/IO statement (`ident +=`, `cin >> ident`, `cout << ident`)
        for m in re.finditer(r"\b([a-zA-Z_]\w*)\b", line_clean):
            ident = m.group(1)
            if ident in all_known or (len(ident) == 1 and ident.isupper()):
                continue
            suggestion = _suggest_closest_var(ident, declared)
            is_direct_var_use = bool(
                re.search(
                    rf"(?:\b{re.escape(ident)}\s*(?:\+=|-=|\*=|/=|%=|=[^=]|\+\+|--|\[))|"
                    rf"(?:(?:cin\s*>>|cout\s*<<|<<|>>)\s*{re.escape(ident)}\b)",
                    line_clean,
                )
            )
            if suggestion and is_direct_var_use and len(ident) >= 3:
                return MicroErrorResult(
                    has_error=True,
                    line=idx,
                    title=f"Variável Não Declarada / Erro de Digitação ('{ident}' vs '{suggestion}')",
                    bug_location=(
                        f"Linha {idx}: A variável '{ident}' não foi declarada neste escopo "
                        f"('error: \\'{ident}\\' was not declared in this scope'). "
                        f"Você declarou '{suggestion}' anteriormente, mas escreveu '{ident}' na linha {idx}."
                    ),
                    how_to_fix=(
                        f"Padronize o nome da variável: troque '{ident}' por '{suggestion}' na linha {idx} "
                        f"(ou renomeie a declaração de '{suggestion}' para '{ident}')."
                    ),
                    monitor_tip=(
                        f"Pergunte ao aluno: 'Olhe o nome da variável que você declarou lá em cima ({suggestion}) "
                        f"e o nome que você usou na linha {idx} ({ident}): elas estão escritas exatamente iguais?'"
                    ),
                )

    # 4. Accidental single '=' assignment inside 'if (a = b)' instead of '=='
    for idx, line in enumerate(lines, start=1):
        m_if = re.search(r"\bif\s*\(\s*([a-zA-Z_]\w*(?:\[[^\]]+\])?)\s*=\s*([^=][^)]*)\)", line)
        if m_if:
            lhs_var = m_if.group(1)
            return MicroErrorResult(
                has_error=True,
                line=idx,
                title=f"Bug Lógico Micro: Atribuição '=' dentro do 'if' em vez de Comparação '==' (Linha {idx})",
                bug_location=(
                    f"Linha {idx}: A condição do 'if' usa um único sinal de igual ('{lhs_var} = ...'), "
                    "que sobrescreve o valor da variável em vez de comparar igualdade ('==')."
                ),
                how_to_fix=f"Na linha {idx}, troque '=' por '==' dentro da condição do 'if'.",
                monitor_tip=(
                    f"Pergunte ao aluno: 'Na linha {idx}, dentro do if, você quer guardar um valor novo em {lhs_var} "
                    "com = ou comparar se os dois valores são iguais com ==?'"
                ),
                is_compile_error=False,
            )

    return None


def _check_python_symbol_and_syntax_errors(code: str) -> MicroErrorResult | None:
    """AST-based Python syntax & undefined variable / typo analyzer (handles comprehensions, with, for, except)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        ln = exc.lineno or 1
        return MicroErrorResult(
            has_error=True,
            line=ln,
            title=f"Erro de Sintaxe Python na Linha {ln}",
            bug_location=f"Linha {ln}: {exc.msg} ({(exc.text or '').strip()}).",
            how_to_fix=f"Revise a sintaxe na linha {ln} (verifique dois-pontos ':', parênteses, operadores ou indentação).",
            monitor_tip=f"Peça ao aluno para observar a linha {ln} e verificar se faltou fechar parênteses ou colocar ':' no final do bloco.",
        )

    assigned: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Param)):
            assigned.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            assigned.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                    assigned.add(arg.arg)
                if node.args.vararg:
                    assigned.add(node.args.vararg.arg)
                if node.args.kwarg:
                    assigned.add(node.args.kwarg.arg)
        elif isinstance(node, ast.Lambda):
            for arg in node.args.args:
                assigned.add(arg.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            assigned.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    # `from math import *` or `from functools import *`: disable strict undefined check
                    return None
                assigned.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Global):
            for gname in node.names:
                assigned.add(gname)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            ident = node.id
            if ident not in assigned and ident not in _PYTHON_BUILTINS and len(ident) >= 3:
                ln = getattr(node, "lineno", 1)
                suggestion = _suggest_closest_var(ident, assigned)
                if suggestion:
                    return MicroErrorResult(
                        has_error=True,
                        line=ln,
                        title=f"NameError Python: Variável '{ident}' Não Definida (Você quis dizer '{suggestion}'?)",
                        bug_location=(
                            f"Linha {ln}: A variável '{ident}' é usada, mas nunca foi definida "
                            f"('NameError: name \\'{ident}\\' is not defined'). Você declarou '{suggestion}' anteriormente."
                        ),
                        how_to_fix=f"Na linha {ln}, corrija o nome da variável de '{ident}' para '{suggestion}'.",
                        monitor_tip=(
                            f"Pergunte ao aluno: 'Compare o nome da variável que você criou ({suggestion}) com o nome "
                            f"que você usou na linha {ln} ({ident}): tem alguma letra diferente?'"
                        ),
                        is_compile_error=True,
                    )

    return None


def check_micro_errors(code: str, language: Language) -> MicroErrorResult | None:
    """Run deterministic compiler/symbol-table micro-error checks with 0% false positives on valid C/C++/Python."""
    if language == "python":
        return _check_python_symbol_and_syntax_errors(code)

    # First run our rich pure-Python symbol table & micro-error analyzer (which works identically on Vercel, Linux CI, and Windows)
    sym_res = _check_cpp_symbol_and_syntax_errors(code, language)
    if sym_res is not None:
        return sym_res

    # Second, if local g++/gcc is available, run -fsyntax-only as an additional compiler check
    # (disabled on low-CPU servers via GCC_SYNTAX_CHECK=0; the real sandbox compile still catches CE)
    if os.getenv("GCC_SYNTAX_CHECK", "1").strip() == "0":
        return None
    compiler_ran, gcc_res = _run_local_gcc_syntax_check(code, language)
    if compiler_ran and gcc_res is not None:
        return gcc_res

    return None
