"""
Core pipeline: runs all 4 stages and returns a ready AgentExecutor.
Imported by both main.py (CLI) and api.py (FastAPI).

Supports both single-image and multi-page PDF inputs.

Multi-page challenge notes
──────────────────────────
1. Region ID uniqueness: each page restarts IDs from 0; we offset by a running
   counter so region_images dict keys are globally unique across all pages.
2. Reading order: LayoutReader operates per-page (it needs normalised coords
   relative to that page's dimensions). Cross-page order is simply page-sequential.
3. Context size: for long PDFs the system prompt can exceed LLM token limits.
   format_ordered_text / format_layout_regions accept max_items to cap it.
4. Memory: pages are processed sequentially; only one page's pixmap is in
   memory at render time (handled in pdf_utils.pdf_to_images).
"""
import compat  # must be first — patches langchain.docstore for paddlex compatibility
import os
from typing import List, Tuple
from dotenv import load_dotenv

import agent_tools                          # holds global region_images dict
from ocr_extraction import run_ocr, get_reading_order, get_ordered_text
from layout_detection import process_document, prepare_region_images, LayoutRegion
from agent import (
    format_ordered_text,
    format_layout_regions,
    build_system_prompt,
    create_agent,
)
from agent_tools import AnalyzeChart, AnalyzeTable

load_dotenv(override=True)


# ── Single-image pipeline ─────────────────────────────────────────────────────

def run_pipeline(image_path: str):
    """
    Execute the full pipeline on a single image.

    Returns:
        agent_executor, ordered_text, layout_regions
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("\n=== Stage 1: OCR + Reading Order ===")
    ocr_regions = run_ocr(image_path)
    reading_order = get_reading_order(ocr_regions)
    ordered_text = get_ordered_text(ocr_regions, reading_order)

    print("\n=== Stage 2: Layout Detection ===")
    layout_regions = process_document(image_path)
    region_images = prepare_region_images(image_path, layout_regions)

    # Inject into agent_tools global so @tool functions can access them
    agent_tools.region_images.update(region_images)

    print("\n=== Stage 3: Building Agent ===")
    ordered_text_str   = format_ordered_text(ordered_text)
    layout_regions_str = format_layout_regions(layout_regions)
    system_prompt = build_system_prompt(ordered_text_str, layout_regions_str)
    agent_executor = create_agent([AnalyzeChart, AnalyzeTable], system_prompt)

    print("Pipeline ready.\n")
    return agent_executor, ordered_text, layout_regions


# ── Multi-page PDF pipeline ───────────────────────────────────────────────────

def _process_page(
    image_path: str,
    page_num: int,           # 1-based
    region_id_offset: int,   # ensures globally unique region IDs
) -> Tuple[List[dict], List[LayoutRegion], dict]:
    """
    Run OCR + layout detection on one page image.

    Returns:
        ordered_text    — list of dicts, each with an added 'page' key
        layout_regions  — list of LayoutRegion with IDs offset globally
        region_images   — dict keyed by globally-unique region_id
    """
    print(f"\n  -- Page {page_num}: OCR --")
    ocr_regions = run_ocr(image_path)
    reading_order = get_reading_order(ocr_regions)
    ordered_text = get_ordered_text(ocr_regions, reading_order)
    # Tag every text item with its page number
    for item in ordered_text:
        item["page"] = page_num

    print(f"  -- Page {page_num}: Layout Detection --")
    layout_regions_raw = process_document(image_path)

    # Offset region IDs so they are globally unique across pages
    for r in layout_regions_raw:
        r.region_id += region_id_offset
        r.page = page_num           # tag with page number for agent context
    region_images = prepare_region_images(image_path, layout_regions_raw)

    return ordered_text, layout_regions_raw, region_images


def run_pipeline_pdf(pdf_path: str, dpi: int = 150, max_pages: int | None = None):
    """
    Execute the full pipeline on a multi-page PDF.

    Steps:
      1. Render each PDF page to a PNG image (pdf_utils)
      2. Run OCR + reading order per page
      3. Run layout detection per page
      4. Merge all results with globally unique region IDs
      5. Build a single page-aware agent

    Returns:
        agent_executor, all_ordered_text, all_layout_regions, page_count
    """
    from pdf_utils import pdf_to_images

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("\n=== PDF → Images ===")
    image_paths = pdf_to_images(pdf_path, dpi=dpi, max_pages=max_pages)
    page_count = len(image_paths)
    print(f"Rendered {page_count} page(s)")

    all_ordered_text: List[dict] = []
    all_layout_regions: List[LayoutRegion] = []
    all_region_images: dict = {}
    region_id_offset = 0

    for page_num, img_path in enumerate(image_paths, start=1):
        print(f"\n=== Processing page {page_num}/{page_count} ===")
        ordered_text, layout_regions, region_images = _process_page(
            img_path, page_num, region_id_offset
        )
        all_ordered_text.extend(ordered_text)
        all_layout_regions.extend(layout_regions)
        all_region_images.update(region_images)
        # Next page's region IDs start after this page's last ID
        region_id_offset += len(layout_regions)

    # Inject merged region_images into agent_tools global
    agent_tools.region_images.clear()
    agent_tools.region_images.update(all_region_images)

    print("\n=== Building Agent ===")
    ordered_text_str   = format_ordered_text(all_ordered_text, include_page=True)
    layout_regions_str = format_layout_regions(all_layout_regions, include_page=True)
    system_prompt = build_system_prompt(
        ordered_text_str, layout_regions_str, page_count=page_count
    )
    agent_executor = create_agent([AnalyzeChart, AnalyzeTable], system_prompt)

    print(f"PDF pipeline ready ({page_count} pages, "
          f"{len(all_ordered_text)} text regions, "
          f"{len(all_layout_regions)} layout regions).\n")
    return agent_executor, all_ordered_text, all_layout_regions, page_count


def run_pipeline_ppt(ppt_path: str, dpi: int = 150, max_pages: int | None = None):
    """
    Execute the full pipeline on a PPT/PPTX file.

    Steps:
      1. Convert PPT/PPTX → PDF (ppt_utils)
      2. Reuse the PDF pipeline
    """
    if not os.path.exists(ppt_path):
        raise FileNotFoundError(f"PPT/PPTX not found: {ppt_path}")

    from ppt_utils import ppt_to_pdf

    print("\n=== PPT/PPTX → PDF ===")
    pdf_path = ppt_to_pdf(ppt_path)
    print(f"Converted to PDF: {pdf_path}")

    agent_executor, ordered_text, layout_regions, page_count = run_pipeline_pdf(
        pdf_path, dpi=dpi, max_pages=max_pages
    )
    return agent_executor, ordered_text, layout_regions, page_count, pdf_path
