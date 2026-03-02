"""
PDF utilities: convert a PDF to a list of per-page PNG images using PyMuPDF.

PyMuPDF (pymupdf) is self-contained — no external system dependencies
(no Poppler, no Ghostscript) needed on Windows.

Install:  pip install pymupdf
"""
import os
import tempfile
from pathlib import Path
from typing import List


def is_pdf(file_path: str) -> bool:
    """Return True if the file is a PDF (by extension and magic bytes)."""
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return False
    with open(file_path, "rb") as f:
        return f.read(4) == b"%PDF"


def pdf_to_images(
    pdf_path: str,
    output_dir: str | None = None,
    dpi: int = 150,
    max_pages: int | None = None,
) -> List[str]:
    """
    Render each page of a PDF to a PNG image.

    Args:
        pdf_path:   Path to the input PDF file.
        output_dir: Directory to write PNGs into.
                    Defaults to a temp directory alongside the PDF.
        dpi:        Render resolution. 150 dpi is a good balance between
                    OCR accuracy and speed/memory. Use 200+ for dense text.
        max_pages:  Cap the number of pages processed (None = all pages).

    Returns:
        Ordered list of absolute paths to the rendered PNG files,
        one per page: [page_001.png, page_002.png, ...]

    Challenges handled:
    - Large PDFs: pages are rendered and written to disk one at a time;
      only one pixmap lives in memory at a time.
    - High-DPI trade-off: 150 dpi (~1240×1754 for A4) is enough for PaddleOCR
      while keeping memory per page ≈15 MB (vs ≈60 MB at 300 dpi).
    - Colour space: always output RGB so PaddleOCR / PIL don't need to convert.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF support. "
            "Install it with:  pip install pymupdf"
        )

    if output_dir is None:
        stem = Path(pdf_path).stem
        output_dir = os.path.join(os.path.dirname(pdf_path), f"{stem}_pages")
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages_to_render = total_pages if max_pages is None else min(max_pages, total_pages)

    # Scale matrix: fitz default is 72 dpi
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    image_paths: List[str] = []
    for page_num in range(pages_to_render):
        page = doc[page_num]
        # get_pixmap with RGB colorspace — no alpha channel needed
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
        out_path = os.path.join(output_dir, f"page_{page_num + 1:03d}.png")
        pix.save(out_path)
        pix = None  # explicitly free pixmap memory before next page
        image_paths.append(out_path)

    doc.close()
    print(f"PDF rendered: {pages_to_render}/{total_pages} pages → {output_dir}")
    return image_paths
