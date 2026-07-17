"""FastAPI server for the Bias Detection Chrome extension.

Endpoints:
  GET  /health   -> readiness + current mode
  POST /analyze  -> AnalyzeResponse for a block of article text

The analyzer is loaded once at startup and kept in memory (persistent server).
Mode is chosen with the ANALYZER_MODE env var: "mock" (default) until the real
SBERT/Llama pipeline is wired in.

Run:
  python -m uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pipeline import build_analyzer
from schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
LOGGER = logging.getLogger("bias_backend")

ANALYZER_MODE = os.environ.get("ANALYZER_MODE", "mock")

app = FastAPI(title="Bias Detection Backend", version="0.1.0")

# Chrome extension pages call this from arbitrary origins (the popup runs on a
# chrome-extension:// origin). Allow all origins; the API is read-only analysis.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = build_analyzer(ANALYZER_MODE)
LOGGER.info("Analyzer ready in '%s' mode.", analyzer.mode)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    detail_parts = ["Backend reachable."]
    selection_config = getattr(analyzer, "selection_config", None)
    if selection_config is not None:
        detail_parts.append(f"selection={selection_config.effective_mode}")
    if getattr(analyzer, "enable_evidence", False):
        detail_parts.append("evidence=on")
    return HealthResponse(
        status="ok",
        mode=analyzer.mode,
        sbert_loaded=analyzer.sbert_loaded,
        llm_loaded=analyzer.llm_loaded,
        detail=" ".join(detail_parts),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    text = (request.text or "").strip()
    if not text:
        return AnalyzeResponse(
            overall_score=0.0,
            score_caption="No article text was provided.",
            report="No readable article text was found on this page.",
            meta={"mode": analyzer.mode, "sentence_count": 0, "selected_count": 0},
        )
    LOGGER.info("Analyzing %d characters (mode=%s).", len(text), analyzer.mode)
    return analyzer.analyze(request)
