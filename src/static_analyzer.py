"""Deterministic pure-Python static analyzer (0 tokens) for C/C++/Python."""

from __future__ import annotations

import ast
import re
from typing import Literal

from pydantic import BaseModel, Field

Language = Literal["cpp", "c", "python"]

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(output|print|return|write|show)\s+(the\s+)?(full|complete)\s+(ac|accepted|solution|gabarito)\s*(code)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(reveal|print|show)\s+(your\s+)?(system\s+prompt|hidden\s+instructions)",
        re.IGNORECASE,
    ),
    re.compile(r"</student_code>|</problem_statement>|<\|im_start\|>", re.IGNORECASE),
]


class StaticReport(BaseModel):
    """Deterministic static analysis findings (0 tokens)."""

    language: Language
    max_loop_depth: int = 0
    estimated_big_o: str = "O(1)"
    findings: list[str] = Field(default_factory=list)
    rag_hints: list[str] = Field(default_factory=list)
    is_injection_attempt: bool = False


def _estimate_big_o(depth: int) -> str:
    if depth <= 0:
        return "O(1)"
    if depth == 1:
        return "O(N)"
    return f"O(N^{depth})"


def _detect_prompt_injection(text: str) -> bool:
    return any(pat.search(text) for pat in _INJECTION_PATTERNS)


def _strip_cpp_comments_and_strings(code: str) -> str:
    pattern = re.compile(r"//[^\n]*|/\*[\s\S]*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'")
    return pattern.sub(lambda m: "\n" * m.group(0).count("\n"), code)


def _python_loop_depth(code: str) -> tuple[int, str | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return 0, f"Erro de sintaxe Python na linha {exc.lineno}: {exc.msg}."

    max_depth = 0

    def _walk(node: ast.AST, depth: int) -> None:
        nonlocal max_depth
        is_loop = isinstance(node, (ast.For, ast.While, ast.AsyncFor))
        nxt = depth + 1 if is_loop else depth
        if nxt > max_depth:
            max_depth = nxt
        for child in ast.iter_child_nodes(node):
            _walk(child, nxt)

    _walk(tree, 0)
    return max_depth, None


def _cpp_loop_depth(cleaned_code: str) -> int:
    brace_balance = 0
    loop_depths: list[int] = []
    max_depth = 0
    pending_loop = False
    is_outer_testcase_loop = False
    loop_re = re.compile(r"\b(for|while)\s*\(")
    tc_loop_re = re.compile(
        r"\bwhile\s*\(\s*(?:t\s*--|tc\s*--|cases\s*--|q\s*--|cin\s*>>|scanf\s*\(|getline\s*\()",
        re.IGNORECASE,
    )

    for line in cleaned_code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        has_loop = bool(loop_re.search(stripped))
        if has_loop:
            is_tc = bool(tc_loop_re.search(stripped)) and len(loop_depths) == 0
            if "{" in stripped:
                if not is_tc:
                    loop_depths.append(brace_balance + 1)
                    max_depth = max(max_depth, len(loop_depths))
            elif stripped.endswith(";"):
                if not is_tc:
                    max_depth = max(max_depth, len(loop_depths) + 1)
            else:
                pending_loop = True
                is_outer_testcase_loop = is_tc
                if not is_tc:
                    max_depth = max(max_depth, len(loop_depths) + 1)

        for ch in stripped:
            if ch == "{":
                brace_balance += 1
                if pending_loop:
                    if not is_outer_testcase_loop:
                        loop_depths.append(brace_balance)
                        max_depth = max(max_depth, len(loop_depths))
                    pending_loop = False
                    is_outer_testcase_loop = False
            elif ch == "}":
                while loop_depths and loop_depths[-1] >= brace_balance:
                    loop_depths.pop()
                brace_balance -= 1
            elif ch == ";" and pending_loop and not has_loop:
                pending_loop = False
                is_outer_testcase_loop = False
    return max_depth


def find_endl_in_loop_line(cleaned_code: str) -> int | None:
    """Return 1-based line number of std::endl inside an active loop, or None."""
    brace_depth = 0
    loop_brace_stack: list[int] = []
    pending_loop_stmt = False
    for idx, line in enumerate(cleaned_code.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        has_loop_kw = bool(re.search(r"\b(for|while)\s*\(", stripped))
        in_active_loop = bool(loop_brace_stack) or pending_loop_stmt or has_loop_kw
        if in_active_loop and re.search(r"\bendl\b", stripped):
            return idx
        if has_loop_kw:
            if "{" in stripped:
                loop_brace_stack.append(brace_depth + 1)
            elif not stripped.endswith(";"):
                pending_loop_stmt = True
        for ch in stripped:
            if ch == "{":
                brace_depth += 1
                if pending_loop_stmt:
                    loop_brace_stack.append(brace_depth)
                    pending_loop_stmt = False
            elif ch == "}":
                while loop_brace_stack and loop_brace_stack[-1] >= brace_depth:
                    loop_brace_stack.pop()
                brace_depth = max(0, brace_depth - 1)
            elif ch == ";" and pending_loop_stmt and not has_loop_kw:
                pending_loop_stmt = False
    return None


def analyze_code(code: str, language: Language, problem_statement: str = "") -> StaticReport:
    """Run all deterministic static checks on C/C++/Python code."""
    findings: list[str] = []
    rag_hints: list[str] = []

    if _detect_prompt_injection(f"{problem_statement}\n{code}"):
        findings.append("Tentativa de Prompt Injection detectada nos comentarios ou enunciado.")
        return StaticReport(
            language=language,
            max_loop_depth=0,
            estimated_big_o="O(1)",
            findings=findings,
            rag_hints=["prompt injection"],
            is_injection_attempt=True,
        )

    if language == "python":
        max_depth, syntax_err = _python_loop_depth(code)
        if syntax_err:
            findings.append(syntax_err)
        if max_depth >= 2:
            findings.append(
                f"Profundidade de loops aninhados = {max_depth} ({_estimate_big_o(max_depth)}): risco de TLE para N >= 10^4."
            )
            rag_hints.append("tle complexidade loops")
        if re.search(r"range\s*\([^)]*len\([^)]+\)\s*\+\s*1\)", code) or re.search(
            r"<=\s*len\(", code
        ):
            findings.append("Uso de '<= len(...)' ou 'len(...) + 1': risco de IndexError em lista 0-indexed.")
            rag_hints.append("out of bounds")
        return StaticReport(
            language=language,
            max_loop_depth=max_depth,
            estimated_big_o=_estimate_big_o(max_depth),
            findings=findings,
            rag_hints=rag_hints,
        )

    # C / C++ static checks
    cleaned = _strip_cpp_comments_and_strings(code)
    lines = cleaned.splitlines()
    max_depth = _cpp_loop_depth(cleaned)

    if max_depth >= 2:
        findings.append(
            f"Profundidade de loops aninhados = {max_depth} ({_estimate_big_o(max_depth)}): risco de TLE para N >= 10^4."
        )
        rag_hints.append("tle complexidade o(n^2)")

    # 1. int without long long (overflow risk)
    has_long_long = "long long" in cleaned or "int64_t" in cleaned
    uses_int_math = bool(re.search(r"(?<!long\s)\bint\s+\w+", cleaned)) and bool(
        re.search(r"(\*=|\+=|\*|1e9|1e5|100000|1000000000)", cleaned)
    )
    if uses_int_math and not has_long_long:
        findings.append(
            "Uso de 'int' (32 bits) em operacoes de soma/multiplicacao sem 'long long': risco de Integer Overflow."
        )
        rag_hints.append("overflow int long long")

    # 2. <= in 0-indexed loop (ignore small fixed bounds like <= 9 or <= 26, or when container is sized n+1)
    has_n_plus_one_alloc = bool(re.search(r"\[\s*\w+\s*\+\s*\d+\s*\]|\(\s*\w+\s*\+\s*\d+", cleaned))
    for idx, line in enumerate(lines, start=1):
        m_le = re.search(r"for\s*\([^;]*\b\w+\s*=\s*0\s*;[^;]*<=\s*(\w+)", line)
        if m_le:
            bound = m_le.group(1)
            if bound.isdigit() and int(bound) <= 30:
                continue
            if has_n_plus_one_alloc:
                continue
            findings.append(
                f"Linha {idx}: condicao '<=' em laco comecando em 0 pode causar acesso fora dos limites (out-of-bounds)."
            )
            rag_hints.append("out of bounds edge case")
            break

    # 3. Uncleaned global containers across multiple test cases
    has_global_container = bool(
        re.search(
            r"^\s*(?:vector\s*<[^>]+>|int|long\s+long)\s+\w+\s*\[",
            cleaned,
            re.MULTILINE,
        )
    )
    has_multi_tests = bool(
        re.search(r"while\s*\(\s*t\s*--\s*\)|for\s*\([^)]*;\s*\w+\s*<\s*t\s*;", cleaned, re.IGNORECASE)
    )
    has_cleanup = ".clear()" in cleaned or "memset(" in cleaned
    if has_global_container and has_multi_tests and not has_cleanup:
        findings.append(
            "Vetor/array global com multiplos casos de teste sem '.clear()' ou 'memset': risco de contaminacao entre casos."
        )
        rag_hints.append("multiplos casos global clear memset")

    # 4. endl inside loop
    endl_line = find_endl_in_loop_line(cleaned)
    if endl_line is not None:
        findings.append(
            f"Linha {endl_line}: uso de 'endl' dentro de laco forca flush constante e pode causar TLE (prefira '\\n')."
        )
        rag_hints.append("endl tle fast io")

    return StaticReport(
        language=language,
        max_loop_depth=max_depth,
        estimated_big_o=_estimate_big_o(max_depth),
        findings=findings,
        rag_hints=rag_hints,
    )
