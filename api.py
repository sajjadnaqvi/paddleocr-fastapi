"""
FastAPI application for the Agentic Document Understanding Pipeline.

Endpoints:
  POST /analyze        — Upload a document image, get full analysis
  POST /query          — Ask a follow-up question about the last analyzed document
  GET  /regions        — List all detected layout regions
  GET  /health         — Health check
"""
import compat  # must be first — patches langchain.docstore & text_splitter for paddlex
import os
import uuid
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Application state ──────────────────────────────────────────────────────────
class AppState:
    agent_executor = None
    ordered_text = []
    layout_regions = []
    last_image_path: Optional[str] = None
    page_count: int = 1
    is_pdf: bool = False

state = AppState()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Document Intelligence API starting up...")
    # Pre-load all heavy models in a thread so the event loop stays free.
    # This means the first /analyze request pays no model-loading cost.
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _warmup_models)
    print("All models loaded. API ready.")
    yield
    print("Shutting down.")


def _warmup_models() -> None:
    """Load PaddleOCR, LayoutDetection, and LayoutLMv3 into memory."""
    from ocr_extraction import warmup_models as ocr_warmup
    from layout_detection import _get_layout_engine
    ocr_warmup()          # PaddleOCR engine + LayoutLMv3
    _get_layout_engine()  # PaddleOCR LayoutDetection engine


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Document Intelligence API",
    description=(
        "An agentic document understanding API combining PaddleOCR, "
        "LayoutLM reading order, and a LangChain VLM agent."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str


class RegionInfo(BaseModel):
    region_id: int
    region_type: str
    bbox: list
    confidence: float


# ── Helpers ───────────────────────────────────────────────────────────────────
def _run_pipeline_sync(file_path: str, is_pdf: bool):
    """Run the heavy pipeline in a thread to not block the event loop."""
    if is_pdf:
        from pipeline import run_pipeline_pdf
        executor, ordered_text, layout_regions, page_count = run_pipeline_pdf(file_path)
        state.page_count = page_count
    else:
        from pipeline import run_pipeline
        executor, ordered_text, layout_regions = run_pipeline(file_path)
        state.page_count = 1
    state.agent_executor = executor
    state.ordered_text   = ordered_text
    state.layout_regions = layout_regions
    state.last_image_path = file_path
    state.is_pdf = is_pdf


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Check API health and pipeline status."""
    return {
        "status": "ok",
        "pipeline_ready": state.agent_executor is not None,
        "last_document": state.last_image_path,
        "document_type": "pdf" if state.is_pdf else "image",
        "page_count": state.page_count,
        "regions_detected": len(state.layout_regions),
    }


@app.post("/analyze", summary="Upload a document image or PDF and run the full pipeline")
async def analyze_document(file: UploadFile = File(...)):
    """
    Upload a PNG/JPG image **or a PDF**.
    The server will:
    1. (PDF only) Render each page to a PNG at 150 dpi
    2. Run PaddleOCR text extraction per page
    3. Compute LayoutLM reading order per page
    4. Run PaddleOCR layout detection per page
    5. Build a single LangChain agent over all pages

    Returns a summary of detected regions and ordered text excerpt.
    """
    ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/tiff", "application/pdf"}
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_TYPES and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. Supported: PNG, JPG, TIFF, PDF."
        )

    is_pdf = content_type == "application/pdf" or (file.filename or "").lower().endswith(".pdf")

    # Save upload
    ext = ".pdf" if is_pdf else (os.path.splitext(file.filename or "")[-1] or ".png")
    filename = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        contents = await file.read()
        f.write(contents)

    # Run pipeline in thread pool to avoid blocking the event loop
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _run_pipeline_sync, file_path, is_pdf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    region_summary = [
        {"region_id": r.region_id, "type": r.region_type, "confidence": round(r.confidence, 3)}
        for r in state.layout_regions
    ]
    text_preview = [item["text"] for item in state.ordered_text[:10]]

    return {
        "status": "ready",
        "document_type": "pdf" if is_pdf else "image",
        "page_count": state.page_count,
        "file_saved_as": filename,
        "regions_detected": len(state.layout_regions),
        "region_summary": region_summary,
        "text_preview_first_10": text_preview,
    }


@app.post("/query", response_model=QueryResponse, summary="Ask a question about the analyzed document")
async def query_document(body: QueryRequest):
    """
    Send a natural language question about the last analyzed document.

    The agent will decide whether to answer from OCR text alone,
    or call **AnalyzeChart** / **AnalyzeTable** VLM tools as needed.
    """
    if state.agent_executor is None:
        raise HTTPException(
            status_code=400,
            detail="No document analyzed yet. POST an image to /analyze first."
        )

    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: state.agent_executor.invoke({"input": body.question})
        )
        return QueryResponse(question=body.question, answer=response["output"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/regions", response_model=list[RegionInfo], summary="List all detected layout regions")
def list_regions():
    """Return all layout regions detected in the last analyzed document."""
    if not state.layout_regions:
        raise HTTPException(status_code=404, detail="No document analyzed yet.")
    return [
        RegionInfo(
            region_id=r.region_id,
            region_type=r.region_type,
            bbox=r.bbox,
            confidence=round(r.confidence, 3),
        )
        for r in state.layout_regions
    ]
