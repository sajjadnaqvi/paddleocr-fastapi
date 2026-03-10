# Agentic Document Understanding Pipeline

A document intelligence system that extracts, structures, and reasons over the content of images and PDFs. It combines OCR, layout detection, reading-order sorting, and a VLM-powered LangChain agent into a single pipeline — exposed both as a CLI and a FastAPI REST API.

---

## What It Does

Given a document image (PNG/JPG/TIFF) or a multi-page PDF, the pipeline:

1. **Extracts text** — PaddleOCR reads every text block along with its bounding box and confidence score.
2. **Sorts reading order** — LayoutLMv3 (via the bundled `layoutreader` module) reorders the text blocks into the natural human reading sequence.
3. **Detects layout regions** — PaddleOCR's LayoutDetection model classifies regions as `title`, `text`, `table`, `chart`, `figure`, `header`, `footer`, etc.
4. **Crops visual regions** — Each detected region is cropped from the image and base64-encoded for downstream visual analysis.
5. **Runs an AI agent** — A LangChain agent (backed by GPT-4o-mini) receives the full OCR text and layout map as context and can call two tools:
   - `AnalyzeChart` — sends a cropped chart/figure to the VLM and returns structured JSON (type, axes, data points, trends).
   - `AnalyzeTable` — sends a cropped table to the VLM and returns structured JSON (headers, rows, notes).
6. **Answers questions** — The agent answers free-form natural-language questions about the document, calling tools only when visual analysis is needed.

---

## File Overview

| File | Purpose |
|---|---|
| `pipeline.py` | Orchestrates all 4 stages; returns a ready `AgentExecutor`. Supports single images and multi-page PDFs. |
| `ocr_extraction.py` | Stage 1 — PaddleOCR text extraction + LayoutLMv3 reading-order sorting. |
| `layout_detection.py` | Stage 2 — PaddleOCR layout detection; crops and base64-encodes each region. |
| `agent_tools.py` | Stage 3 — LangChain `@tool` definitions for `AnalyzeChart` and `AnalyzeTable` (calls GPT-4o-mini). |
| `agent.py` | Stage 4 — Builds the system prompt from OCR/layout context and creates the `AgentExecutor`. |
| `api.py` | FastAPI server — exposes `/analyze`, `/query`, `/regions`, and `/health` endpoints. |
| `main.py` | CLI entry point — runs the pipeline on `report_original.png` and fires three test queries. |
| `pdf_utils.py` | Converts PDF pages to PNG images using PyMuPDF (no external dependencies required). |
| `image_reconstruction.py` | Generates visual debug views: annotated bboxes, text-layout canvas, region crop grid, HTML overlay. |
| `compat.py` | Monkey-patches LangChain internals for PaddleX compatibility. |
| `layoutreader/` | Bundled LayoutLMv3-based reading-order model (local copy). |
| `L6.ipynb` | Interactive notebook walkthrough of the full pipeline. |

---

## Processing Flow

```
Input (image or PDF)
        |
        v
[pdf_utils] PDF -> per-page PNGs          (PDF only)
        |
        v
[ocr_extraction] PaddleOCR -> text blocks + bboxes
        |
        v
[ocr_extraction] LayoutLMv3 -> reading-order index
        |
        v
[layout_detection] PaddleOCR LayoutDetection -> region labels + bboxes
        |
        v
[layout_detection] Crop & base64-encode each region
        |
        v
[agent] Build system prompt (ordered text + layout map)
        |
        v
[agent] LangChain AgentExecutor (GPT-4o-mini)
        |        |
        |        +-- AnalyzeChart tool -> VLM -> structured JSON
        |        +-- AnalyzeTable tool -> VLM -> structured JSON
        v
  Answer / structured output
```

---

## Prerequisites

- Python 3.10+
- An OpenAI API key

---

## Installation & Setup

### 1. Create and activate a virtual environment

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

> **Note:** `layoutreader` is bundled locally in the `layoutreader/` folder. No separate install is needed.

### 3. Configure environment

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

---

## Usage

### CLI

Place your document image in the project root as `report_original.png`, then run:

```powershell
python main.py
```

This runs the full pipeline and prints agent responses for three built-in test queries (document overview, table extraction, chart analysis).

### FastAPI Server

```powershell
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`. Key endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Pipeline status and last document info |
| `POST` | `/analyze` | Upload a PNG, JPG, TIFF, or PDF — runs the full pipeline |
| `POST` | `/query` | Ask a follow-up question about the last analyzed document |
| `GET` | `/regions` | List all detected layout regions and ordered text |

Interactive API docs: `http://localhost:8000/docs`

### Notebook

Open `L6.ipynb` in VS Code or Jupyter and run cells top to bottom for an interactive walkthrough.

---

## Supported Input Formats

- Images: PNG, JPG/JPEG, TIFF
- Documents: PDF (rendered at 150 dpi per page via PyMuPDF — no Poppler or Ghostscript required)
