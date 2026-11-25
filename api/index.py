"""FastAPI Serverless Application for Vercel (Zero-API Dual-ML Code Club Monitor Copilot)."""

from __future__ import annotations

import hmac
import json
import os
import shutil
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.agent import DiagnosisResponse, run_copilot, save_monitor_feedback_case
from src.demo_cache import DEMO_SCENARIOS
from src.feature_extractor import ALGO_FAMILY_LABELS_PT, BUG_CLASS_LABELS_PT
from src.free_llm_engine import ask_socratic_followup, get_active_free_provider
from src.ml_predictor import DEFAULT_WEIGHTS_PATH
from src.problem_fetcher import FetchedProblem, fetch_problem_from_url
from src.real_problems_catalog import RealProblemsCatalog
from src.sandbox_runner import SandboxResult, TestCaseInput, run_test_cases_in_sandbox
from src.static_analyzer import Language
from src.storage import list_custom_problems, save_custom_problem
from src.vision_ocr import VisionMode, extract_from_multiple_images, stitch_overlapping_texts

ROOT_DIR = Path(__file__).resolve().parent.parent
PUBLIC_HTML_PATH = ROOT_DIR / "public" / "index.html"
_PROBLEM_CATALOG = RealProblemsCatalog()

app = FastAPI(
    title="Code Monitor Copilot",
    version="3.0.0",
    description="Diagnóstico instantâneo via Modelo Duplo Treinado (BugMLP + Algorithm Matcher) e RAG em 5 fontes canônicas.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
_RATE_LIMITED_PATHS = {
    "/api/diagnose",
    "/api/run-tests",
    "/api/chat",
    "/api/ocr",
    "/api/problems/fetch-url",
}


@app.middleware("http")
async def security_and_pin_middleware(request: Request, call_next: Any) -> Any:
    """Enforce optional MONITOR_ACCESS_PIN, sliding-window rate limits, and security headers."""
    path = request.url.path

    # 1. Optional PIN protection
    expected_pin = os.getenv("MONITOR_ACCESS_PIN", "").strip()
    if expected_pin and path.startswith("/api/") and path not in ("/api/health", "/api/auth/verify", "/api/demos"):
        provided = request.headers.get("x-monitor-pin", "").strip()
        if not hmac.compare_digest(provided, expected_pin):
            return JSONResponse(status_code=401, content={"detail": "PIN de monitor inválido ou ausente."})

    # 2. Lightweight per-IP rate limit (60 req / 60s) on compute/network endpoints
    if path in _RATE_LIMITED_PATHS:
        client_ip = (request.client.host if request.client else "") or "local"
        if client_ip != "testclient":
            now = time.monotonic()
            bucket = _RATE_LIMIT_BUCKETS[client_ip]
            while bucket and now - bucket[0] > 60.0:
                bucket.popleft()
            if len(bucket) >= 60:
                return JSONResponse(status_code=429, content={"detail": "Muitas requisições. Aguarde alguns segundos."})
            bucket.append(now)

    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


class DiagnoseRequest(BaseModel):
    """Request payload for code + problem diagnosis."""

    language: Language = "cpp"
    problem_statement: str = Field(default="", max_length=30000)
    code: str = Field(..., min_length=1, max_length=50000)
    verdict: str = "unknown"
    sample_input: str = Field(default="", max_length=10000)
    expected_output: str = Field(default="", max_length=10000)
    test_cases: list[TestCaseInput] = Field(default_factory=list)
    demo_id: str | None = None
    source_filter: str | None = None


class RunTestsRequest(BaseModel):
    """Standalone multi-testcase sandbox execution request."""

    language: Language = "cpp"
    code: str = Field(..., min_length=1, max_length=50000)
    test_cases: list[TestCaseInput] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """Socratic follow-up question from the monitor on the bench."""

    question: str = Field(..., min_length=1)
    language: Language = "cpp"
    code: str = ""
    problem_statement: str = ""
    previous_diagnosis: str = ""


class CustomProblemRequest(BaseModel):
    """Save a classroom problem (NEPS, Beecrowd, Codeforces, AtCoder, Aula)."""

    platform: str = "NEPS / Aula"
    title: str = Field(..., min_length=2)
    statement: str = Field(..., min_length=5)
    sample_input: str = ""
    expected_output: str = ""


class FetchUrlRequest(BaseModel):
    """Fetch a problem statement and sample I/O from a URL."""

    url: str = Field(..., min_length=4)


class FeedbackRequest(BaseModel):
    """Monitor classroom submission feedback for continuous dataset improvement."""

    language: Language = "cpp"
    problem_statement: str = ""
    code: str = Field(..., min_length=1)
    bug_class: str = "integer_overflow"
    algo_family: str = "math_ad_hoc"
    monitor_notes: str = ""


@app.get("/", response_class=HTMLResponse)
def serve_index() -> HTMLResponse:
    """Serve the single-page web UI locally and on Vercel."""
    if PUBLIC_HTML_PATH.exists():
        return HTMLResponse(PUBLIC_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Code Monitor Copilot API Online</h1>")


@app.get("/health")
def lightweight_health() -> dict[str, str]:
    """Lightweight health check for Docker / Hugging Face Spaces."""
    return {"status": "ok"}


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    """Return engine health and trained model metadata."""
    meta: dict[str, Any] = {}
    if DEFAULT_WEIGHTS_PATH.exists():
        raw = json.loads(DEFAULT_WEIGHTS_PATH.read_text(encoding="utf-8"))
        meta = raw.get("metadata", {})
    provider = get_active_free_provider()
    return {
        "status": "ok",
        "zero_api_mode": provider is None,
        "ai_provider": f"{provider['name']}:{provider['model']}" if provider else "local_sandbox_ml",
        "compiler_available": bool(shutil.which("g++")),
        "pin_required": bool(os.getenv("MONITOR_ACCESS_PIN", "").strip()),
        "weights_loaded": DEFAULT_WEIGHTS_PATH.exists(),
        "real_problems_count": _PROBLEM_CATALOG.total_count(),
        "model_metadata": meta,
        "bug_classes": BUG_CLASS_LABELS_PT,
        "algo_families": ALGO_FAMILY_LABELS_PT,
    }


@app.post("/api/auth/verify")
def verify_pin_endpoint(request: Request) -> dict[str, bool]:
    """Validate the optional MONITOR_ACCESS_PIN."""
    expected_pin = os.getenv("MONITOR_ACCESS_PIN", "").strip()
    if not expected_pin:
        return {"ok": True}
    provided = request.headers.get("x-monitor-pin", "").strip()
    if not hmac.compare_digest(provided, expected_pin):
        raise HTTPException(status_code=401, detail="PIN incorreto.")
    return {"ok": True}


@app.get("/api/problems/search")
def search_real_problems(
    q: str = "",
    tag: str = "",
    limit: int = 24,
) -> dict[str, Any]:
    """Search across the 3,500+ real Codeforces problems catalog."""
    items = _PROBLEM_CATALOG.search(query=q, tag_or_family=tag, limit=min(limit, 50))
    return {
        "total_catalog_size": _PROBLEM_CATALOG.total_count(),
        "returned": len(items),
        "problems": items,
    }


@app.get("/api/demos")
def list_demos() -> list[dict[str, Any]]:
    """Return the 5 canonical competitive programming classroom scenarios."""
    return [
        {
            "id": d["id"],
            "title": d["title"],
            "language": d["language"],
            "problem_statement": d["problem_statement"],
            "code": d["code"],
        }
        for d in DEMO_SCENARIOS
    ]


@app.post("/api/diagnose", response_model=DiagnosisResponse)
def diagnose_submission(req: DiagnoseRequest) -> DiagnosisResponse:
    """Run the 3-node LangGraph + Dual-ML inference pipeline (< 15ms, 0 API keys)."""
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="O codigo do aluno nao pode estar vazio.")
    filt = None if (not req.source_filter or req.source_filter == "all") else req.source_filter
    return run_copilot(
        language=req.language,
        code=req.code,
        problem_statement=req.problem_statement,
        verdict=req.verdict,
        sample_input=req.sample_input,
        expected_output=req.expected_output,
        test_cases=req.test_cases,
        demo_id=req.demo_id,
        source_filter=filt,
    )


@app.post("/api/run-tests", response_model=SandboxResult)
def run_tests_endpoint(req: RunTestsRequest) -> SandboxResult:
    """Compile and execute student code against one or more test cases (with ASan/UBSan when available)."""
    if not req.code.strip():
        raise HTTPException(status_code=400, detail="O código do aluno não pode estar vazio.")
    if not req.test_cases:
        raise HTTPException(status_code=400, detail="Adicione pelo menos um caso de teste.")
    return run_test_cases_in_sandbox(
        code=req.code,
        language=req.language,
        test_cases=req.test_cases,
    )


@app.post("/api/chat")
def socratic_chat_endpoint(req: ChatRequest) -> dict[str, Any]:
    """Answer a monitor's follow-up question on the bench."""
    reply, model_used = ask_socratic_followup(
        question=req.question,
        language=req.language,
        code=req.code,
        problem_statement=req.problem_statement,
        previous_diagnosis=req.previous_diagnosis,
    )
    return {"reply": reply, "model_used": model_used}


@app.get("/api/problems/custom")
def get_custom_problems(limit: int = 50) -> dict[str, Any]:
    """Return classroom problems saved by instructors/monitors."""
    items = list_custom_problems(limit=limit)
    return {"count": len(items), "problems": items}


@app.post("/api/problems/custom")
def create_custom_problem(req: CustomProblemRequest) -> dict[str, Any]:
    """Save a custom NEPS/Beecrowd/classroom problem into SQLite."""
    return save_custom_problem(
        platform=req.platform,
        title=req.title,
        statement=req.statement,
        sample_input=req.sample_input,
        expected_output=req.expected_output,
    )


@app.post("/api/problems/fetch-url", response_model=FetchedProblem)
def fetch_problem_url_endpoint(req: FetchUrlRequest) -> FetchedProblem:
    """Import problem statement and sample I/O from a NEPS, Beecrowd, AtCoder, CSES, or Codeforces link."""
    return fetch_problem_from_url(req.url)


@app.post("/api/ocr")
async def ocr_multiple_images(
    files: Annotated[list[UploadFile], File(...)],
    mode: Annotated[VisionMode, Form()] = "problem",
    existing_text: Annotated[str, Form()] = "",
) -> dict[str, Any]:
    """Extract and stitch text/code/exponents from 1 or more uploaded photos (100% Zero-API)."""
    if not files:
        raise HTTPException(status_code=400, detail="Envie pelo menos uma foto.")
    images_bytes: list[bytes] = []
    for f in files:
        raw = await f.read()
        if raw:
            images_bytes.append(raw)
    if not images_bytes:
        raise HTTPException(status_code=400, detail="Arquivos de imagem vazios.")
    try:
        out = extract_from_multiple_images(images_bytes, mode=mode)
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Leitura de fotos (OCR) indisponivel neste ambiente. Cole o texto diretamente.",
        ) from None
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Nao foi possivel ler a imagem: {exc}",
        ) from None
    if existing_text.strip() and out.get("extracted_text"):
        out["extracted_text"] = stitch_overlapping_texts(
            existing_text.strip(),
            str(out["extracted_text"]),
            mode=mode,
        )
    return out


@app.post("/api/feedback")
def record_monitor_feedback(req: FeedbackRequest) -> dict[str, Any]:
    """Save or export a real classroom case for retraining."""
    return save_monitor_feedback_case(
        language=req.language,
        problem_statement=req.problem_statement,
        code=req.code,
        bug_class=req.bug_class,
        algo_family=req.algo_family,
        monitor_notes=req.monitor_notes,
    )
