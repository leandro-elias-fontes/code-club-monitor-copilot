"""FastAPI entrypoint for Vercel Serverless (`api/index.py`)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.agent import DiagnosisResponse, get_api_keys, run_copilot
from src.demo_cache import DEMO_SCENARIOS
from src.static_analyzer import Language

load_dotenv()

app = FastAPI(
    title="Code Club Monitor Copilot",
    description="Diagnostico em 3s via RAG para monitores de programacao competitiva",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


class DiagnoseRequest(BaseModel):
    """Input payload for /api/diagnose."""

    language: Language = "cpp"
    problem_statement: str = Field(default="", max_length=6000)
    code: str = Field(..., min_length=1, max_length=20000)
    demo_id: str | None = None
    source_filter: str | None = None


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    """Serve the clean Portuguese monitor UI."""
    if _FRONTEND_PATH.exists():
        return HTMLResponse(content=_FRONTEND_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Code Club Monitor Copilot</h1>")


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Healthcheck with model config and demo count."""
    return {
        "status": "ok",
        "primary_model": os.getenv("GEMINI_PRIMARY_MODEL", "gemini-3.8-flash"),
        "fallback_model": os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash-lite"),
        "gemini_keys_configured": len(get_api_keys()),
        "cached_demos": len(DEMO_SCENARIOS),
    }


@app.get("/api/demos")
def list_demos() -> list[dict[str, Any]]:
    """Return the 5 pre-calculated zero-token demo scenarios."""
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
def diagnose(req: DiagnoseRequest) -> DiagnosisResponse:
    """Run demo cache or 3-node LangGraph RAG pipeline."""
    return run_copilot(
        language=req.language,
        code=req.code,
        problem_statement=req.problem_statement,
        demo_id=req.demo_id,
        source_filter=req.source_filter,
    )
