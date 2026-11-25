"""Fetch and parse problem statements + sample I/O from NEPS, Beecrowd, AtCoder, CSES, and Codeforces URLs."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from src.real_problems_catalog import RealProblemsCatalog
from src.sandbox_runner import TestCaseInput, extract_all_sample_ios

_CATALOG = RealProblemsCatalog()


class FetchedProblem(BaseModel):
    """Normalized problem statement + sample test cases fetched from a URL."""

    ok: bool = True
    platform: str = "Web"
    title: str = ""
    statement: str = ""
    test_cases: list[TestCaseInput] = Field(default_factory=list)
    source_url: str = ""
    message: str = ""


class _TextExtractor(HTMLParser):
    """Clean HTML-to-text converter preserving newlines and code/pre blocks."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript", "svg", "nav", "footer"):
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr", "pre", "section"):
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript", "svg", "nav", "footer"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth == 0 and tag in ("p", "div", "h1", "h2", "h3", "h4", "pre", "section", "tr"):
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = html.unescape("".join(self._parts))
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.splitlines()]
        cleaned: list[str] = []
        for ln in lines:
            if ln:
                cleaned.append(ln)
            elif cleaned and cleaned[-1] != "":
                cleaned.append("")
        return "\n".join(cleaned).strip()


def _html_to_text(raw_html: str) -> str:
    parser = _TextExtractor()
    parser.feed((raw_html or "")[:500_000])
    return parser.get_text()


def _is_blocked_or_challenge(status_code: int, body: str) -> bool:
    if status_code in (403, 429, 503):
        return True
    low = (body or "").lower()
    return any(
        tok in low
        for tok in (
            "just a moment...",
            "cf-browser-verification",
            "cloudflare",
            "captcha",
            "verify you are human",
            "access denied",
        )
    )


def _fetch_http(url: str) -> tuple[int, str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
    }
    with httpx.Client(timeout=10.0, follow_redirects=True) as client:
        resp = client.get(url, headers=headers)
        return resp.status_code, resp.text, resp.headers.get("content-type", "")


def _parse_beecrowd_html(raw_html: str, url: str, problem_id: str) -> FetchedProblem:
    m_title = re.search(r"<h1[^>]*>([\s\S]*?)</h1>", raw_html, re.IGNORECASE)
    title = _html_to_text(m_title.group(1)) if m_title else f"Beecrowd {problem_id}"

    # Extract sample input/output tables from Beecrowd UOJ HTML
    cases: list[TestCaseInput] = []
    for idx, m_tr in enumerate(
        re.finditer(r"<tbody[^>]*>[\s\S]*?<tr[^>]*>\s*<td[^>]*>([\s\S]*?)</td>\s*<td[^>]*>([\s\S]*?)</td>", raw_html, re.IGNORECASE),
        start=1,
    ):
        inp = _html_to_text(m_tr.group(1)).strip()
        out = _html_to_text(m_tr.group(2)).strip()
        if inp or out:
            cases.append(TestCaseInput(name=f"Exemplo {idx}", sample_input=inp, expected_output=out))

    full_text = _html_to_text(raw_html)
    # Trim noisy header before title if present
    statement = f"Beecrowd {problem_id} — {title}\n\n{full_text}"[:6000]
    if not cases:
        cases = extract_all_sample_ios(statement)
    return FetchedProblem(
        ok=True,
        platform="Beecrowd",
        title=title,
        statement=statement,
        test_cases=cases[:5],
        source_url=url,
    )


def _parse_atcoder_html(raw_html: str, url: str) -> FetchedProblem:
    m_title = re.search(r"<title[^>]*>([\s\S]*?)</title>", raw_html, re.IGNORECASE)
    title = _html_to_text(m_title.group(1)).replace(" - AtCoder", "").strip() if m_title else "AtCoder"

    # Prefer English section if available
    m_en = re.search(r'<span class="lang-en">([\s\S]*?)</span>\s*</span>', raw_html, re.IGNORECASE)
    block = m_en.group(1) if m_en else raw_html

    cases: list[TestCaseInput] = []
    pres = re.findall(r"<h3[^>]*>\s*Sample\s+Input\s*(\d*)\s*</h3>\s*<pre[^>]*>([\s\S]*?)</pre>", block, re.IGNORECASE)
    outs = re.findall(r"<h3[^>]*>\s*Sample\s+Output\s*(\d*)\s*</h3>\s*<pre[^>]*>([\s\S]*?)</pre>", block, re.IGNORECASE)
    for i, ((_, inp_raw), (_, out_raw)) in enumerate(zip(pres, outs, strict=False), start=1):
        cases.append(
            TestCaseInput(
                name=f"Exemplo {i}",
                sample_input=html.unescape(inp_raw).strip(),
                expected_output=html.unescape(out_raw).strip(),
            )
        )

    stmt = _html_to_text(block)[:6000]
    if not cases:
        cases = extract_all_sample_ios(stmt)
    return FetchedProblem(
        ok=True,
        platform="AtCoder",
        title=title,
        statement=stmt,
        test_cases=cases[:5],
        source_url=url,
    )


def _parse_cses_html(raw_html: str, url: str) -> FetchedProblem:
    m_title = re.search(r"<h1[^>]*>([\s\S]*?)</h1>", raw_html, re.IGNORECASE)
    title = _html_to_text(m_title.group(1)) if m_title else "CSES"
    m_content = re.search(r'<div class="content">([\s\S]*?)</div>\s*<div class="nav', raw_html, re.IGNORECASE)
    block = m_content.group(1) if m_content else raw_html

    cases: list[TestCaseInput] = []
    pres = re.findall(r"<pre[^>]*>([\s\S]*?)</pre>", block, re.IGNORECASE)
    for i in range(0, len(pres) - 1, 2):
        inp = html.unescape(pres[i]).strip()
        out = html.unescape(pres[i + 1]).strip()
        if inp and out:
            cases.append(TestCaseInput(name=f"Exemplo {len(cases) + 1}", sample_input=inp, expected_output=out))

    stmt = f"CSES — {title}\n\n{_html_to_text(block)}"[:6000]
    return FetchedProblem(
        ok=True,
        platform="CSES",
        title=title,
        statement=stmt,
        test_cases=cases[:5],
        source_url=url,
    )


def _fetch_neps(url: str, exercise_id: str) -> FetchedProblem:
    # NEPS Academy exposes public exercise JSON at api.neps.academy
    api_url = f"https://api.neps.academy/exercises/{exercise_id}"
    try:
        status, body, ctype = _fetch_http(api_url)
        if status == 200 and "json" in ctype.lower():
            import json

            data: dict[str, Any] = json.loads(body)
            title = str(data.get("name") or data.get("title") or f"NEPS {exercise_id}")
            desc = str(data.get("description") or data.get("statement") or "")
            inp_desc = str(data.get("input_description") or data.get("input") or "")
            out_desc = str(data.get("output_description") or data.get("output") or "")
            constraints = str(data.get("constraints") or "")
            parts = [f"NEPS {exercise_id} — {title}", _html_to_text(desc)]
            if inp_desc:
                parts.append("Entrada:\n" + _html_to_text(inp_desc))
            if out_desc:
                parts.append("Saída:\n" + _html_to_text(out_desc))
            if constraints:
                parts.append("Restrições:\n" + _html_to_text(constraints))
            statement = "\n\n".join(p for p in parts if p.strip())[:6000]

            cases: list[TestCaseInput] = []
            for idx, ex in enumerate(data.get("examples") or data.get("test_cases") or [], start=1):
                if isinstance(ex, dict):
                    i_txt = str(ex.get("input") or "").strip()
                    o_txt = str(ex.get("output") or "").strip()
                    if i_txt or o_txt:
                        cases.append(TestCaseInput(name=f"Exemplo {idx}", sample_input=i_txt, expected_output=o_txt))
            if not cases:
                cases = extract_all_sample_ios(statement)
            return FetchedProblem(
                ok=True,
                platform="NEPS",
                title=title,
                statement=statement,
                test_cases=cases[:5],
                source_url=url,
            )
    except Exception:
        pass

    return FetchedProblem(
        ok=False,
        platform="NEPS",
        source_url=url,
        message="O NEPS bloqueou ou exigiu login para carregar este exercício automaticamente. Cole o enunciado ou dê Ctrl+V com um print.",
    )


def fetch_problem_from_url(raw_url: str) -> FetchedProblem:
    """Fetch problem statement and sample I/O from a competitive programming URL."""
    url = (raw_url or "").strip()
    if not url:
        return FetchedProblem(ok=False, message="Informe um link válido da questão.")
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""

    # 1. Beecrowd (try direct static repository UOJ HTML first — clean & fast)
    if "beecrowd" in host or "urionlinejudge" in host:
        m_id = re.search(r"(\d{4})", path)
        if m_id:
            pid = m_id.group(1)
            for repo_url in (
                f"https://resources.beecrowd.com/repository/UOJ_{pid}.html",
                f"https://resources.beecrowd.com/repository/UOJ_{pid}_en.html",
                url,
            ):
                try:
                    status, body, _ = _fetch_http(repo_url)
                    if status == 200 and not _is_blocked_or_challenge(status, body) and len(body) > 300:
                        return _parse_beecrowd_html(body, url, pid)
                except Exception:
                    continue
        return FetchedProblem(
            ok=False,
            platform="Beecrowd",
            source_url=url,
            message="Não foi possível acessar esta página do Beecrowd diretamente (proteção do site). Cole o enunciado ou dê Ctrl+V com um print.",
        )

    # 2. NEPS Academy
    if "neps.academy" in host:
        m_id = re.search(r"/exercise/(\d+)", path)
        if m_id:
            return _fetch_neps(url, m_id.group(1))
        return FetchedProblem(
            ok=False,
            platform="NEPS",
            source_url=url,
            message="Link do NEPS inválido. Use o formato https://neps.academy/br/exercise/123.",
        )

    # 3. Codeforces (try HTTP; if Cloudflare blocks, fallback to local 3,500+ Codeforces catalog!)
    if "codeforces.com" in host:
        m_cf = re.search(r"/(?:contest|problemset/problem)/(\d+)/(?:problem/)?([A-Za-z]\d?)", path)
        try:
            status, body, _ = _fetch_http(url)
            if status == 200 and not _is_blocked_or_challenge(status, body) and "problem-statement" in body:
                m_stmt = re.search(r'<div class="problem-statement">([\s\S]*?)</div>\s*</div>\s*<script', body, re.IGNORECASE)
                block = m_stmt.group(1) if m_stmt else body
                stmt = _html_to_text(block)[:6000]
                cases = extract_all_sample_ios(stmt)
                return FetchedProblem(
                    ok=True,
                    platform="Codeforces",
                    title=f"Codeforces {m_cf.group(1)}{m_cf.group(2)}" if m_cf else "Codeforces",
                    statement=stmt,
                    test_cases=cases[:5],
                    source_url=url,
                )
        except Exception:
            pass

        # Fallback to local catalog if contest ID is in our 3,500+ offline Codeforces index
        if m_cf:
            cid = m_cf.group(1)
            hits = _CATALOG.search(query=cid, limit=10)
            for h in hits:
                if str(h.get("contest_id")) == cid:
                    stmt = str(h.get("statement") or "")
                    return FetchedProblem(
                        ok=True,
                        platform="Codeforces (Catálogo Offline)",
                        title=str(h.get("name") or f"CF {cid}"),
                        statement=stmt,
                        test_cases=extract_all_sample_ios(stmt),
                        source_url=url,
                        message="Carregado do catálogo offline (Codeforces bloqueou requisição direta).",
                    )
        return FetchedProblem(
            ok=False,
            platform="Codeforces",
            source_url=url,
            message="O Codeforces ativou proteção anti-bot nesta URL. Cole o enunciado diretamente ou dê Ctrl+V com um print.",
        )

    # 4. AtCoder
    if "atcoder.jp" in host:
        try:
            status, body, _ = _fetch_http(url)
            if status == 200 and not _is_blocked_or_challenge(status, body):
                return _parse_atcoder_html(body, url)
        except Exception:
            pass
        return FetchedProblem(
            ok=False,
            platform="AtCoder",
            source_url=url,
            message="Não foi possível carregar esta página do AtCoder agora.",
        )

    # 5. CSES
    if "cses.fi" in host:
        try:
            status, body, _ = _fetch_http(url)
            if status == 200 and not _is_blocked_or_challenge(status, body):
                return _parse_cses_html(body, url)
        except Exception:
            pass
        return FetchedProblem(
            ok=False,
            platform="CSES",
            source_url=url,
            message="Não foi possível carregar esta página do CSES agora.",
        )

    # 6. Generic webpage fallback
    try:
        status, body, _ = _fetch_http(url)
        if _is_blocked_or_challenge(status, body):
            return FetchedProblem(
                ok=False,
                source_url=url,
                message="Este site bloqueou o acesso automático. Cole o texto do enunciado ou um print (Ctrl+V).",
            )
        m_title = re.search(r"<title[^>]*>([\s\S]*?)</title>", body, re.IGNORECASE)
        title = _html_to_text(m_title.group(1)) if m_title else host
        stmt = _html_to_text(body)[:5000]
        return FetchedProblem(
            ok=True,
            platform=host,
            title=title,
            statement=stmt,
            test_cases=extract_all_sample_ios(stmt)[:5],
            source_url=url,
        )
    except Exception as exc:
        return FetchedProblem(
            ok=False,
            source_url=url,
            message=f"Erro ao acessar o link: {exc}",
        )
