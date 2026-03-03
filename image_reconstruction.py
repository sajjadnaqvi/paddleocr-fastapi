"""
image_reconstruction.py
Generate visual representations of a document from its extracted OCR / layout data.

Four modes
──────────
  annotated    — Original image with colored bounding boxes + type labels drawn over it.
  text_layout  — White canvas with every OCR word placed at its detected coordinates.
  crops        — Grid of all cropped layout-region thumbnails with labels.
  html         — Standalone HTML page with absolutely-positioned text & region overlays.

All image-returning functions return a PIL Image.
Call image_to_bytes() to get raw bytes suitable for a FastAPI StreamingResponse.
"""

import os
from io import BytesIO
from typing import List, Dict, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


# ── Color palette (region_type → RGB) ────────────────────────────────────────

REGION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "text":      (70,  130, 180),   # steel blue
    "title":     (34,  139,  34),   # forest green
    "table":     (210,  50,  50),   # red
    "chart":     (255, 140,   0),   # orange
    "figure":    (148,   0, 211),   # purple
    "list":      (  0, 139, 139),   # teal
    "header":    (184, 134,  11),   # dark gold
    "footer":    (105, 105, 105),   # dim gray
    "caption":   ( 72,  61, 139),   # dark slate blue
    "equation":  (  0, 100,   0),   # dark green
    "code":      (139,   0,   0),   # dark red
}
_DEFAULT_COLOR: Tuple[int, int, int] = (120, 120, 120)  # fallback gray


def _color(region_type: str) -> Tuple[int, int, int]:
    return REGION_COLORS.get(region_type.lower(), _DEFAULT_COLOR)


# ── Font helper ───────────────────────────────────────────────────────────────

def _font(size: int = 13) -> ImageFont.FreeTypeFont:
    """Load a TrueType font, falling back gracefully to the PIL default."""
    candidates = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ── Mode 1: annotated ─────────────────────────────────────────────────────────

def reconstruct_annotated(
    image_path: str,
    layout_regions,          # List[LayoutRegion]
    ordered_text: List[dict],
    *,
    show_text_boxes: bool = True,
    show_layout_boxes: bool = True,
    fill_opacity: float = 0.25,
    page: int = 1,
) -> Image.Image:
    """
    Draw colored bounding boxes and labels on the original document image.

    - Layout regions  → semi-transparent filled rectangles with type + ID label badges.
    - OCR text boxes  → thin blue outlines (optional, off by default for cleanliness).

    Args:
        image_path:        Path to the source document image (PNG/JPEG).
        layout_regions:    All LayoutRegion objects (multi-page aware via .page).
        ordered_text:      All OCR text dicts (multi-page aware via ['page']).
        show_text_boxes:   Whether to draw OCR word bounding boxes.
        show_layout_boxes: Whether to draw layout region boxes.
        fill_opacity:      Alpha of the region fill (0 = transparent, 1 = opaque).
        page:              Which page to render (always 1 for single images).

    Returns:
        Annotated PIL Image (RGB).
    """
    base = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_ov = ImageDraw.Draw(overlay)
    draw_base = ImageDraw.Draw(base)
    f = _font(13)

    page_regions = [r for r in layout_regions if r.page == page]
    page_text    = [t for t in ordered_text   if t.get("page", 1) == page]

    # ── Layout region boxes ──
    if show_layout_boxes:
        for region in page_regions:
            rgb = _color(region.region_type)
            x1, y1, x2, y2 = region.bbox

            # Semi-transparent fill on overlay
            draw_ov.rectangle(
                [x1, y1, x2, y2],
                fill=(*rgb, int(255 * fill_opacity)),
            )
            # Solid border on base
            draw_base.rectangle([x1, y1, x2, y2], outline=rgb, width=2)

            # Label badge above the box
            label = f"{region.region_type} #{region.region_id}"
            tb = draw_base.textbbox((0, 0), label, font=f)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            lx = max(0, x1)
            ly = max(0, y1 - th - 6)
            draw_base.rectangle([lx, ly, lx + tw + 6, ly + th + 6], fill=rgb)
            draw_base.text((lx + 3, ly + 3), label, fill=(255, 255, 255), font=f)

    # ── OCR text boxes ──
    if show_text_boxes:
        f_small = _font(10)
        for item in page_text:
            bbox = item.get("bbox")
            if not bbox:
                continue
            # Support both polygon [[x,y], ...] and flat [x1,y1,x2,y2]
            if isinstance(bbox[0], (list, tuple)):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            elif len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            else:
                continue
            draw_base.rectangle([x1, y1, x2, y2], outline=(80, 80, 220), width=1)

    # Merge semi-transparent overlay onto base
    base_rgba = base.convert("RGBA")
    merged = Image.alpha_composite(base_rgba, overlay)
    return merged.convert("RGB")


# ── Mode 2: text_layout ───────────────────────────────────────────────────────

def reconstruct_text_layout(
    ordered_text: List[dict],
    *,
    canvas_size: Optional[Tuple[int, int]] = None,
    page: int = 1,
) -> Image.Image:
    """
    White canvas with every OCR word rendered at its detected coordinates.

    Text color reflects OCR confidence:
      ≥ 0.90 → dark green  (high confidence)
      ≥ 0.70 → amber       (medium confidence)
       < 0.70 → red         (low confidence)

    A light 100 px grid is drawn to aid spatial orientation.

    Args:
        ordered_text:  All OCR text dicts.
        canvas_size:   (width, height) in pixels.  Defaults to (1200, 1600).
        page:          Which page to render.

    Returns:
        PIL Image (RGB).
    """
    page_text = [t for t in ordered_text if t.get("page", 1) == page]
    W, H = canvas_size if canvas_size else (1200, 1600)

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    f = _font(13)

    # Light grid
    for gx in range(0, W, 100):
        draw.line([(gx, 0), (gx, H)], fill=(220, 220, 220), width=1)
    for gy in range(0, H, 100):
        draw.line([(0, gy), (W, gy)], fill=(220, 220, 220), width=1)

    for i, item in enumerate(page_text):
        text = item.get("text", "")
        bbox = item.get("bbox")
        conf = item.get("confidence", 1.0)

        # Resolve position from bbox
        if bbox and isinstance(bbox[0], (list, tuple)):
            x, y = int(bbox[0][0]), int(bbox[0][1])
        elif bbox and len(bbox) >= 2:
            x, y = int(bbox[0]), int(bbox[1])
        else:
            # No bbox — stack vertically as fallback
            x, y = 10, 20 + i * 18

        # Clamp to canvas
        x = max(2, min(x, W - 6))
        y = max(2, min(y, H - 18))

        # Confidence color
        if conf >= 0.90:
            color = (0, 110, 0)
        elif conf >= 0.70:
            color = (180, 100, 0)
        else:
            color = (190, 20, 20)

        draw.text((x, y), text, fill=color, font=f)

    return img


# ── Mode 3: crops grid ────────────────────────────────────────────────────────

def reconstruct_crops_grid(
    image_path: str,
    layout_regions,          # List[LayoutRegion]
    *,
    page: int = 1,
    cols: int = 3,
    thumb_size: Tuple[int, int] = (400, 300),
    cell_padding: int = 12,
    label_height: int = 28,
) -> Image.Image:
    """
    Render all cropped layout regions as a labeled thumbnail grid.

    Each cell shows:
      - A colored header bar with region type + ID + confidence score
      - The cropped region thumbnail below it

    Args:
        image_path:    Source document image.
        layout_regions: All LayoutRegion objects.
        page:          Which page's regions to show.
        cols:          Number of columns in the grid.
        thumb_size:    Max (width, height) per thumbnail.
        cell_padding:  Gap between cells and borders.
        label_height:  Height of the colored label bar above each thumbnail.

    Returns:
        PIL Image (RGB).
    """
    from layout_detection import crop_region

    page_regions = [r for r in layout_regions if r.page == page]

    if not page_regions:
        img = Image.new("RGB", (500, 80), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((10, 28), "No layout regions detected on this page.", fill=(80, 80, 80), font=_font(14))
        return img

    src = Image.open(image_path).convert("RGB")
    f   = _font(13)

    thumbs: List[Tuple[Image.Image, object]] = []
    for region in page_regions:
        crop = crop_region(src, region.bbox, padding=6)
        crop = crop.copy()
        crop.thumbnail(thumb_size, Image.LANCZOS)
        thumbs.append((crop, region))

    rows   = (len(thumbs) + cols - 1) // cols
    cell_w = thumb_size[0] + 2 * cell_padding
    cell_h = thumb_size[1] + 2 * cell_padding + label_height

    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 245, 245))
    draw = ImageDraw.Draw(grid)

    for idx, (thumb, region) in enumerate(thumbs):
        col = idx % cols
        row = idx // cols
        cx  = col * cell_w + cell_padding
        cy  = row * cell_h + cell_padding

        # Colored label bar
        rgb = _color(region.region_type)
        draw.rectangle([cx, cy, cx + thumb_size[0], cy + label_height - 2], fill=rgb)
        label = f"#{region.region_id}  {region.region_type}  ({region.confidence:.2f})"
        draw.text((cx + 4, cy + 6), label, fill=(255, 255, 255), font=f)

        # Thumbnail — centered horizontally in the cell
        tx = cx + (thumb_size[0] - thumb.width)  // 2
        ty = cy + label_height
        grid.paste(thumb, (tx, ty))

    return grid


# ── Mode 4: html ──────────────────────────────────────────────────────────────

def export_as_html(
    ordered_text: List[dict],
    layout_regions,          # List[LayoutRegion]
    *,
    page: int = 1,
) -> str:
    """
    Generate a standalone HTML document with absolutely positioned OCR text
    and semi-transparent layout region overlays.

    The canvas size is derived from the maximum bounding-box coordinates, so
    no source image is required.

    Args:
        ordered_text:    All OCR text dicts.
        layout_regions:  All LayoutRegion objects.
        page:            Which page to export.

    Returns:
        HTML string (UTF-8).
    """
    page_text    = [t for t in ordered_text    if t.get("page", 1) == page]
    page_regions = [r for r in layout_regions  if r.page == page]

    # Infer canvas bounds
    max_x = max((r.bbox[2] for r in page_regions), default=850)
    max_y = max((r.bbox[3] for r in page_regions), default=1100)

    items: List[str] = []

    # ── Layout region boxes ──
    for region in page_regions:
        r, g, b = _color(region.region_type)
        x1, y1, x2, y2 = region.bbox
        w, h = x2 - x1, y2 - y1
        items.append(
            f'<div class="region" style="'
            f'left:{x1}px;top:{y1}px;width:{w}px;height:{h}px;'
            f'border:2px solid rgb({r},{g},{b});'
            f'background:rgba({r},{g},{b},0.08);">'
            f'<span class="rlabel" style="background:rgb({r},{g},{b});">'
            f'{region.region_type} #{region.region_id}'
            f'</span></div>'
        )

    # ── OCR text items ──
    for item in page_text:
        text = item.get("text", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        conf = item.get("confidence", 1.0)
        bbox = item.get("bbox")
        if not bbox:
            continue
        if isinstance(bbox[0], (list, tuple)):
            x, y = int(bbox[0][0]), int(bbox[0][1])
            x2t  = int(bbox[2][0]) if len(bbox) > 2 else x + 100
        elif len(bbox) >= 4:
            x, y, x2t = int(bbox[0]), int(bbox[1]), int(bbox[2])
        else:
            continue
        w = max(x2t - x, 60)
        # Confidence color
        if conf >= 0.90:
            col_css = "#006400"
        elif conf >= 0.70:
            col_css = "#b46400"
        else:
            col_css = "#b41414"
        items.append(
            f'<div class="word" style="left:{x}px;top:{y}px;width:{w}px;color:{col_css};">'
            f'{text}</div>'
        )

    items_html = "\n".join(items)
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        f"  <title>Document Reconstruction \u2014 Page {page}</title>\n"
        "  <style>\n"
        "    *, *::before, *::after { box-sizing: border-box; }\n"
        "    body  { font-family: Arial, sans-serif; background:#e8e8e8; margin:0; padding:16px; }\n"
        "    h1    { text-align:center; font-size:16px; color:#333; margin-bottom:12px; }\n"
        "    .canvas {\n"
        "      position: relative;\n"
        f"      width:  {max_x}px;\n"
        f"      height: {max_y}px;\n"
        "      background: #fff;\n"
        "      margin: 0 auto;\n"
        "      box-shadow: 0 2px 10px rgba(0,0,0,.25);\n"
        "      overflow: hidden;\n"
        "    }\n"
        "    .region { position: absolute; pointer-events: none; }\n"
        "    .rlabel {\n"
        "      position: absolute; top: -22px; left: 0;\n"
        "      font-size: 11px; color: #fff; padding: 2px 6px;\n"
        "      white-space: nowrap; border-radius: 3px 3px 3px 0;\n"
        "    }\n"
        "    .word {\n"
        "      position: absolute; font-size: 12px;\n"
        "      white-space: nowrap; overflow: hidden; line-height: 1.2;\n"
        "    }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"  <h1>Document Reconstruction &mdash; Page {page}</h1>\n"
        "  <div class=\"canvas\">\n"
        + items_html + "\n"
        "  </div>\n"
        "</body>\n"
        "</html>"
    )


# ── Serialization helper ──────────────────────────────────────────────────────

def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Serialize a PIL Image to raw bytes (PNG by default)."""
    buf = BytesIO()
    if fmt.upper() in ("JPEG", "JPG") and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()
