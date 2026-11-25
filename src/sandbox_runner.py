"""Safe local sandbox runner with compiler diagnostics, ASan/UBSan, and multi-testcase support."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src.static_analyzer import Language

ExecutionStatus = Literal[
    "not_run",
    "ac_sample",
    "pe",
    "wa",
    "rte",
    "tle",
    "ce",
]


class TestCaseInput(BaseModel):
    """Single test case input and optional expected output."""

    name: str = "Exemplo 1"
    sample_input: str = ""
    expected_output: str = ""


class TestCaseResult(BaseModel):
    """Execution result for a single test case."""

    name: str = "Exemplo 1"
    status: ExecutionStatus = "not_run"
    sample_input: str = ""
    expected_output: str = ""
    actual_output: str = ""
    stderr: str = ""
    runtime_ms: float = 0.0
    error_line: int | None = None
    sanitizer_issue: str = ""


class SandboxResult(BaseModel):
    """Structured result from compiling and running student code on 1 or more test cases."""

    ran: bool = False
    status: ExecutionStatus = "not_run"
    sample_input: str = ""
    expected_output: str = ""
    actual_output: str = ""
    stderr: str = ""
    runtime_ms: float = 0.0
    error_line: int | None = None
    compiler_warnings: list[str] = Field(default_factory=list)
    sanitizer_issue: str = ""
    sanitizers_used: bool = False
    cases: list[TestCaseResult] = Field(default_factory=list)


def extract_sample_io(problem_statement: str) -> tuple[str, str]:
    """Extract first Sample Input and Sample Output from Beecrowd, NEPS, Codeforces, or AtCoder statements."""
    cases = extract_all_sample_ios(problem_statement)
    if cases:
        return cases[0].sample_input, cases[0].expected_output
    return "", ""


def extract_all_sample_ios(problem_statement: str) -> list[TestCaseInput]:
    """Extract one or more Sample Input / Output pairs from a problem statement."""
    text = (problem_statement or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    patterns = [
        re.compile(
            r"exemplo\s+de\s+entrada(?:\s*\d+)?\s*[:\n]+(.*?)\n\s*exemplo\s+de\s+sa[ií]da(?:\s*\d+)?\s*[:\n]+(.*?)(?=\n\s*(?:nota|explica|observa|restri|beecrowd|neps|exemplo\s+de)|\Z)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"(?:sample\s+)?input(?:\s*\d+)?\s*[:\n]+(.*?)\n\s*(?:sample\s+)?output(?:\s*\d+)?\s*[:\n]+(.*?)(?=\n\s*(?:note|explanation|constraint|(?:sample\s+)?input)|\Z)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"\bentrada(?:\s*\d+)?\s*[:\n]+(.*?)\n\s*sa[ií]da(?:\s*\d+)?\s*[:\n]+(.*?)(?=\n\s*(?:nota|explica|observa|entrada)|\Z)",
            re.IGNORECASE | re.DOTALL,
        ),
    ]

    for pat in patterns:
        found: list[TestCaseInput] = []
        for idx, m in enumerate(pat.finditer(text), start=1):
            inp = m.group(1).strip()
            out = m.group(2).strip()
            if inp and out and len(inp) <= 2000 and len(out) <= 2000:
                found.append(
                    TestCaseInput(
                        name=f"Exemplo {idx}",
                        sample_input=inp,
                        expected_output=out,
                    )
                )
        if found:
            return found[:5]
    return []


def _normalize_pe(s: str) -> str:
    """Normalize whitespace/case to detect Presentation Error vs true Wrong Answer."""
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").strip().splitlines()]
    return "\n".join(lines).lower()


def _extract_error_line(stderr: str, language: Language) -> int | None:
    if language == "python":
        matches = re.findall(r'File\s+"[^"]+",\s+line\s+(\d+)', stderr)
        return int(matches[-1]) if matches else None
    # Sanitizer or compiler line e.g. solution.cpp:14:9: runtime error: ... or error: ...
    m = re.search(r"solution\.(?:cpp|c):(\d+)(?::\d+)?", stderr)
    if m:
        return int(m.group(1))
    m2 = re.search(r":(\d+):\d+:\s*(?:error|fatal error|runtime error):", stderr)
    return int(m2.group(1)) if m2 else None


def _parse_compiler_warnings(stderr: str) -> list[str]:
    warnings: list[str] = []
    for ln in (stderr or "").splitlines():
        if ": warning:" in ln:
            clean = re.sub(r"^.*?solution\.(?:cpp|c):", "Linha ", ln).strip()
            if clean not in warnings:
                warnings.append(clean)
    return warnings[:6]


def _parse_sanitizer_issue(stderr: str) -> str:
    low = (stderr or "").lower()
    if "signed integer overflow" in low:
        return "UBSan: Integer Overflow (estouro de int 32-bit em tempo de execução)."
    if "heap-buffer-overflow" in low or "stack-buffer-overflow" in low or "global-buffer-overflow" in low or "index-out-of-bounds" in low:
        return "ASan/UBSan: Acesso fora dos limites do vetor/array (Out-of-Bounds)."
    if "division by zero" in low:
        return "UBSan: Divisão por zero em tempo de execução."
    if "shift exponent" in low or "left shift of" in low:
        return "UBSan: Bit-shift inválido / overflow de deslocamento (ex: 1 << k)."
    if "null pointer" in low or "segmentation fault" in low or "addresssanitizer: segv" in low:
        return "ASan: Segmentation Fault / acesso a ponteiro inválido."
    if "runtime error:" in low:
        for ln in stderr.splitlines():
            if "runtime error:" in ln.lower():
                return f"UBSan: {ln.strip()[:180]}"
    return ""


def _compile_c_cpp(
    code: str,
    language: Language,
    tmp_path: Path,
) -> tuple[list[str] | None, list[str], bool, str, int | None]:
    """Compile C/C++ with -Wall -Wextra and ASan/UBSan when supported by the host toolchain."""
    compiler = shutil.which("g++" if language == "cpp" else "gcc")
    if not compiler:
        return None, [], False, "", None

    ext = ".cpp" if language == "cpp" else ".c"
    src_file = tmp_path / f"solution{ext}"
    bin_file = tmp_path / ("solution.exe" if os.name == "nt" else "solution")
    src_file.write_text(code, encoding="utf-8")

    std_flag = "-std=c++17" if language == "cpp" else "-std=c11"
    # Fast build (matches the precompiled bits/stdc++.h header generated in the Dockerfile)
    base_cmd = [compiler, std_flag, "-O0", "-Wall", "-Wextra", str(src_file), "-o", str(bin_file)]
    compile_timeout = float(os.getenv("SANDBOX_COMPILE_TIMEOUT", "25"))

    # ASan/UBSan are much heavier to compile; opt-in via SANDBOX_SANITIZERS=1 (slow on low-CPU free tiers)
    use_sanitizers = os.getenv("SANDBOX_SANITIZERS", "0").strip() == "1"
    sanitizers_used = False
    try:
        if use_sanitizers:
            san_cmd = [
                compiler,
                std_flag,
                "-O1",
                "-g",
                "-Wall",
                "-Wextra",
                "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer",
                str(src_file),
                "-o",
                str(bin_file),
            ]
            comp = subprocess.run(san_cmd, capture_output=True, text=True, timeout=compile_timeout)
            stderr_txt = (comp.stderr or "").strip()
            if comp.returncode != 0 and any(
                tok in stderr_txt.lower()
                for tok in ("libasan", "libubsan", "cannot find -lasan", "cannot find -lubsan", "unrecognized", "sanitizer")
            ):
                comp = subprocess.run(base_cmd, capture_output=True, text=True, timeout=compile_timeout)
                stderr_txt = (comp.stderr or "").strip()
            elif comp.returncode == 0:
                sanitizers_used = True
        else:
            comp = subprocess.run(base_cmd, capture_output=True, text=True, timeout=compile_timeout)
            stderr_txt = (comp.stderr or "").strip()
    except subprocess.TimeoutExpired:
        # Slow server: do NOT report a fake Compile Error, just skip execution
        return None, [], False, "", None

    warnings = _parse_compiler_warnings(stderr_txt)
    if comp.returncode != 0:
        return None, warnings, False, stderr_txt[:1400], _extract_error_line(stderr_txt, language)

    return [str(bin_file)], warnings, sanitizers_used, "", None


def run_test_cases_in_sandbox(
    code: str,
    language: Language,
    test_cases: list[TestCaseInput],
    timeout_sec: float = 2.0,
) -> SandboxResult:
    """Compile once (for C/C++) and execute code against one or more test cases."""
    valid_cases = [tc for tc in test_cases if tc.sample_input.strip() or tc.expected_output.strip()]
    if not valid_cases:
        return SandboxResult(ran=False, status="not_run")

    with tempfile.TemporaryDirectory(prefix="copilot_run_") as tmpdir:
        tmp_path = Path(tmpdir)
        warnings: list[str] = []
        sanitizers_used = False

        if language == "python":
            src_file = tmp_path / "solution.py"
            src_file.write_text(code, encoding="utf-8")
            cmd: list[str] = [sys.executable, "-I", str(src_file)]
        else:
            compiled_cmd, warnings, sanitizers_used, comp_err, err_line = _compile_c_cpp(
                code, language, tmp_path
            )
            if compiled_cmd is None:
                if not comp_err:
                    return SandboxResult(ran=False, status="not_run")
                first = valid_cases[0]
                return SandboxResult(
                    ran=True,
                    status="ce",
                    sample_input=first.sample_input,
                    expected_output=first.expected_output,
                    stderr=comp_err,
                    error_line=err_line,
                    compiler_warnings=warnings,
                    sanitizers_used=False,
                    cases=[
                        TestCaseResult(
                            name=first.name,
                            status="ce",
                            sample_input=first.sample_input,
                            expected_output=first.expected_output,
                            stderr=comp_err,
                            error_line=err_line,
                        )
                    ],
                )
            cmd = compiled_cmd

        # Minimal sanitized environment so student code never sees server API keys/secrets
        env: dict[str, str] = {}
        for safe_key in ("PATH", "SystemRoot", "TEMP", "TMP", "LANG", "LC_ALL"):
            if safe_key in os.environ:
                env[safe_key] = os.environ[safe_key]
        if sanitizers_used:
            env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1"
            env["UBSAN_OPTIONS"] = "print_stacktrace=1:halt_on_error=1"

        preexec = None
        if os.name != "nt":
            def _limit_resources() -> None:
                try:
                    import resource

                    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
                    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
                    if not sanitizers_used:
                        # ASan reserves large virtual address space, so only cap RLIMIT_AS when ASan is off
                        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
                except Exception:
                    pass

            preexec = _limit_resources

        case_results: list[TestCaseResult] = []
        worst_case: TestCaseResult | None = None
        severity = {"ce": 5, "rte": 4, "tle": 3, "wa": 2, "pe": 1, "ac_sample": 0, "not_run": -1}

        for tc in valid_cases[:6]:
            inp = tc.sample_input.replace("\r\n", "\n")
            exp = tc.expected_output.replace("\r\n", "\n")
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    input=inp,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                    env=env,
                    cwd=str(tmp_path),
                    preexec_fn=preexec,
                )
            except subprocess.TimeoutExpired:
                res = TestCaseResult(
                    name=tc.name,
                    status="tle",
                    sample_input=inp,
                    expected_output=exp,
                    runtime_ms=round(timeout_sec * 1000.0, 1),
                    stderr=f"Execução excedeu {timeout_sec:.1f}s no caso '{tc.name}'.",
                )
                case_results.append(res)
                if worst_case is None or severity[res.status] > severity[worst_case.status]:
                    worst_case = res
                continue

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            stdout_txt = (proc.stdout or "").replace("\r\n", "\n")[:4000]
            stderr_txt = (proc.stderr or "").strip()[:1400]
            san_issue = _parse_sanitizer_issue(stderr_txt)

            if proc.returncode != 0 or san_issue:
                res = TestCaseResult(
                    name=tc.name,
                    status="rte",
                    sample_input=inp,
                    expected_output=exp,
                    actual_output=stdout_txt,
                    stderr=stderr_txt or f"Processo encerrou com código {proc.returncode}.",
                    runtime_ms=elapsed_ms,
                    error_line=_extract_error_line(stderr_txt, language),
                    sanitizer_issue=san_issue,
                )
            elif not exp.strip():
                res = TestCaseResult(
                    name=tc.name,
                    status="ac_sample",
                    sample_input=inp,
                    expected_output="",
                    actual_output=stdout_txt,
                    runtime_ms=elapsed_ms,
                )
            elif stdout_txt.rstrip("\n") == exp.rstrip("\n"):
                res = TestCaseResult(
                    name=tc.name,
                    status="ac_sample",
                    sample_input=inp,
                    expected_output=exp,
                    actual_output=stdout_txt,
                    runtime_ms=elapsed_ms,
                )
            elif _normalize_pe(stdout_txt) == _normalize_pe(exp):
                res = TestCaseResult(
                    name=tc.name,
                    status="pe",
                    sample_input=inp,
                    expected_output=exp,
                    actual_output=stdout_txt,
                    runtime_ms=elapsed_ms,
                )
            else:
                res = TestCaseResult(
                    name=tc.name,
                    status="wa",
                    sample_input=inp,
                    expected_output=exp,
                    actual_output=stdout_txt,
                    runtime_ms=elapsed_ms,
                )

            case_results.append(res)
            if worst_case is None or severity[res.status] > severity[worst_case.status]:
                worst_case = res

        primary = worst_case or case_results[0]
        return SandboxResult(
            ran=True,
            status=primary.status,
            sample_input=primary.sample_input,
            expected_output=primary.expected_output,
            actual_output=primary.actual_output,
            stderr=primary.stderr,
            runtime_ms=primary.runtime_ms,
            error_line=primary.error_line,
            compiler_warnings=warnings,
            sanitizer_issue=primary.sanitizer_issue,
            sanitizers_used=sanitizers_used,
            cases=case_results,
        )


def run_in_sandbox(
    code: str,
    language: Language,
    sample_input: str = "",
    expected_output: str = "",
    timeout_sec: float = 2.0,
) -> SandboxResult:
    """Backward-compatible single-testcase sandbox helper."""
    if not (sample_input or "").strip() and not (expected_output or "").strip():
        return SandboxResult(ran=False, status="not_run")
    return run_test_cases_in_sandbox(
        code=code,
        language=language,
        test_cases=[
            TestCaseInput(
                name="Exemplo 1",
                sample_input=sample_input,
                expected_output=expected_output,
            )
        ],
        timeout_sec=timeout_sec,
    )
