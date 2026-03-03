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
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Query, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
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
    page: int = 1


class TextItem(BaseModel):
    position: int        # reading-order index (0 = first to read)
    text: str
    confidence: float
    bbox: list           # [x1, y1, x2, y2]
    page: int = 1


class RegionsResponse(BaseModel):
    layout_regions: list[RegionInfo]
    extracted_text: list[TextItem]
    total_regions: int
    total_text_items: int


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


@app.get("/regions", response_model=RegionsResponse, summary="List layout regions and all extracted text")
def list_regions():
    """
    Return all layout regions **and** all extracted OCR text for the last
    analyzed document.

    Each `extracted_text` item includes:
    - `position` — reading-order index (0 = first item to read)
    - `text`     — the recognized string
    - `confidence` — OCR confidence score (0–1, higher is better)
    - `bbox`     — bounding box `[x1, y1, x2, y2]` in pixels
    - `page`     — 1-based page number (always 1 for single images)
    """
    if not state.layout_regions and not state.ordered_text:
        raise HTTPException(status_code=404, detail="No document analyzed yet.")

    regions = [
        RegionInfo(
            region_id=r.region_id,
            region_type=r.region_type,
            bbox=r.bbox,
            confidence=round(r.confidence, 3),
            page=getattr(r, "page", 1),
        )
        for r in state.layout_regions
    ]

    text_items = [
        TextItem(
            position=item.get("position", i),
            text=item["text"],
            confidence=round(item.get("confidence", 0.0), 4),
            bbox=item.get("bbox", []),
            page=item.get("page", 1),
        )
        for i, item in enumerate(state.ordered_text)
    ]

    return RegionsResponse(
        layout_regions=regions,
        extracted_text=text_items,
        total_regions=len(regions),
        total_text_items=len(text_items),
    )


@app.get(
    "/reconstruct",
    summary="Generate a visual reconstruction of the analyzed document",
    responses={
        200: {
            "description": "PNG image or HTML page",
            "content": {"image/png": {}, "text/html": {}},
        }
    },
)
async def reconstruct_document(
    mode: str = Query(
        "annotated",
        description=(
            "Reconstruction mode:\n"
            "- **annotated** *(default)* — original image with colored region boxes + labels\n"
            "- **text_layout** — white canvas with OCR text placed at detected coordinates\n"
            "- **crops** — grid of all cropped layout-region thumbnails with labels\n"
            "- **html** — standalone HTML with positioned text & region overlays"
        ),
    ),
    page: int = Query(1, ge=1, description="Page number to reconstruct (1-based; for PDFs only)"),
    show_text_boxes: bool = Query(
        False, description="(annotated mode) Also draw OCR word bounding boxes"
    ),
    show_layout_boxes: bool = Query(
        True, description="(annotated mode) Draw layout-region bounding boxes"
    ),
):
    """
    Generate a visual representation of the **last analyzed document**.

    | Mode | Returns | Description |
    |---|---|---|
    | `annotated` | PNG | Original image with colored region boxes and type labels |
    | `text_layout` | PNG | White canvas — every OCR word placed at its spatial coordinates |
    | `crops` | PNG | Thumbnail grid of every detected layout region |
    | `html` | HTML | Positioned text + region overlays as a standalone web page |

    For **PDF** documents use `?page=N` (1-based) to select which page to reconstruct.
    """
    if not state.layout_regions and not state.ordered_text:
        raise HTTPException(
            status_code=400,
            detail="No document analyzed yet. POST a document to /analyze first.",
        )

    from image_reconstruction import (
        reconstruct_annotated,
        reconstruct_text_layout,
        reconstruct_crops_grid,
        export_as_html,
        image_to_bytes,
    )

    # ── Resolve source image path for the requested page ──────────────────────
    def _page_image_path(page_num: int) -> str:
        """Return the PNG path for the given page number."""
        if not state.is_pdf:
            return state.last_image_path
        # PDF page images are stored alongside the PDF:
        # uploads/<stem>_pages/page_NNN.png  (written by pdf_utils)
        pdf_path = state.last_image_path
        pages_dir = os.path.join(
            os.path.dirname(pdf_path),
            f"{Path(pdf_path).stem}_pages",
        )
        page_img = os.path.join(pages_dir, f"page_{page_num:03d}.png")
        if not os.path.exists(page_img):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Page {page_num} image not found. "
                    f"This PDF has {state.page_count} page(s)."
                ),
            )
        return page_img

    # ── Effective page number (always 1 for single images) ────────────────────
    eff_page = page if state.is_pdf else 1
    if state.is_pdf and page > state.page_count:
        raise HTTPException(
            status_code=400,
            detail=f"Page {page} requested but document only has {state.page_count} page(s).",
        )

    mode = mode.strip().lower()
    try:
        # ── annotated ─────────────────────────────────────────────────────────
        if mode == "annotated":
            img_path = _page_image_path(eff_page)
            loop = asyncio.get_running_loop()
            img = await loop.run_in_executor(
                None,
                lambda: reconstruct_annotated(
                    image_path=img_path,
                    layout_regions=state.layout_regions,
                    ordered_text=state.ordered_text,
                    show_text_boxes=show_text_boxes,
                    show_layout_boxes=show_layout_boxes,
                    page=eff_page,
                ),
            )
            return StreamingResponse(
                BytesIO(image_to_bytes(img)),
                media_type="image/png",
                headers={"Content-Disposition": f'inline; filename="annotated_page{eff_page}.png"'},
            )

        # ── text_layout ───────────────────────────────────────────────────────
        elif mode == "text_layout":
            # Use the source image dimensions as canvas size when available
            try:
                from PIL import Image as _PIL
                with _PIL.open(_page_image_path(eff_page)) as _src:
                    canvas_size = (_src.width, _src.height)
            except Exception:
                canvas_size = (1200, 1600)

            loop = asyncio.get_running_loop()
            img = await loop.run_in_executor(
                None,
                lambda: reconstruct_text_layout(
                    ordered_text=state.ordered_text,
                    canvas_size=canvas_size,
                    page=eff_page,
                ),
            )
            return StreamingResponse(
                BytesIO(image_to_bytes(img)),
                media_type="image/png",
                headers={"Content-Disposition": f'inline; filename="text_layout_page{eff_page}.png"'},
            )

        # ── crops ─────────────────────────────────────────────────────────────
        elif mode == "crops":
            img_path = _page_image_path(eff_page)
            loop = asyncio.get_running_loop()
            img = await loop.run_in_executor(
                None,
                lambda: reconstruct_crops_grid(
                    image_path=img_path,
                    layout_regions=state.layout_regions,
                    page=eff_page,
                ),
            )
            return StreamingResponse(
                BytesIO(image_to_bytes(img)),
                media_type="image/png",
                headers={"Content-Disposition": f'inline; filename="crops_page{eff_page}.png"'},
            )

        # ── html ──────────────────────────────────────────────────────────────
        elif mode == "html":
            html_content = export_as_html(
                ordered_text=state.ordered_text,
                layout_regions=state.layout_regions,
                page=eff_page,
            )
            return HTMLResponse(content=html_content)

        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown mode '{mode}'. "
                    "Valid modes: annotated, text_layout, crops, html"
                ),
            )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reconstruction error: {exc}") from exc
