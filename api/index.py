"""FastAPI Serverless Application for Vercel (Zero-API Dual-ML Code Club Monitor Copilot)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.agent import DiagnosisResponse, run_copilot, save_monitor_feedback_case
from src.demo_cache import DEMO_SCENARIOS
from src.feature_extractor import ALGO_FAMILY_LABELS_PT, BUG_CLASS_LABELS_PT
from src.ml_predictor import DEFAULT_WEIGHTS_PATH
from src.real_problems_catalog import RealProblemsCatalog
from src.static_analyzer import Language
from src.vision_ocr import VisionMode, extract_from_multiple_images, stitch_overlapping_texts

ROOT_DIR = Path(__file__).resolve().parent.parent
PUBLIC_HTML_PATH = ROOT_DIR / "public" / "index.html"
_PROBLEM_CATALOG = RealProblemsCatalog()

app = FastAPI(
    title="Code Monitor Copilot (Zero-API ML)",
    version="2.0.0",
    description="Diagnóstico instantâneo via Modelo Duplo Treinado (BugMLP + Algorithm Matcher) e RAG em 5 fontes canônicas.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class DiagnoseRequest(BaseModel):
    """Request payload for code + problem diagnosis."""

    language: Language = "cpp"
    problem_statement: str = ""
    code: str = Field(..., min_length=1)
    demo_id: str | None = None
    source_filter: str | None = None


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


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    """Return engine health and trained model metadata."""
    meta: dict[str, Any] = {}
    if DEFAULT_WEIGHTS_PATH.exists():
        raw = json.loads(DEFAULT_WEIGHTS_PATH.read_text(encoding="utf-8"))
        meta = raw.get("metadata", {})
    return {
        "status": "ok",
        "zero_api_mode": True,
        "weights_loaded": DEFAULT_WEIGHTS_PATH.exists(),
        "real_problems_count": _PROBLEM_CATALOG.total_count(),
        "model_metadata": meta,
        "bug_classes": BUG_CLASS_LABELS_PT,
        "algo_families": ALGO_FAMILY_LABELS_PT,
    }


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
        demo_id=req.demo_id,
        source_filter=filt,
    )


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
    out = extract_from_multiple_images(images_bytes, mode=mode)
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
