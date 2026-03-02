"""
Stage 2: Layout Detection with PaddleOCR + Region Cropping
"""
import compat  # must be first — patches langchain.docstore for paddlex compatibility
import base64
from dataclasses import dataclass
from io import BytesIO
from typing import List, Dict, Any

from paddleocr import LayoutDetection
from PIL import Image

# --- Data Class ---

@dataclass
class LayoutRegion:
    region_id: int
    region_type: str
    bbox: list   # [x1, y1, x2, y2]
    confidence: float
    page: int = 1   # 1-based page number (always 1 for single images)


# --- Layout Detection ---

_layout_engine = None

def _get_layout_engine():
    global _layout_engine
    if _layout_engine is None:
        _layout_engine = LayoutDetection()
    return _layout_engine


def process_document(image_path: str) -> List[LayoutRegion]:
    """Run layout detection and return structured LayoutRegion list."""
    engine = _get_layout_engine()
    layout_result = engine.predict(image_path)

    regions = []
    for box in layout_result[0]['boxes']:
        regions.append({
            'label': box['label'],
            'score': box['score'],
            'bbox': box['coordinate'],
        })

    # Sort by confidence descending
    regions = sorted(regions, key=lambda x: x['score'], reverse=True)

    layout_regions: List[LayoutRegion] = []
    for i, r in enumerate(regions):
        layout_regions.append(LayoutRegion(
            region_id=i,
            region_type=r['label'],
            bbox=[int(x) for x in r['bbox']],
            confidence=r['score']
        ))

    print(f"Detected {len(layout_regions)} layout regions")
    return layout_regions


# --- Region Cropping ---

def crop_region(image: Image.Image, bbox: list, padding: int = 10) -> Image.Image:
    """Crop a region from PIL image with optional padding."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width,  x2 + padding)
    y2 = min(image.height, y2 + padding)
    return image.crop((x1, y1, x2, y2))


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string.
    JPEG is used (quality=90): ~5x faster to encode than PNG and
    produces a smaller payload for the VLM API with no perceptible
    quality loss for chart/table analysis.
    """
    buffer = BytesIO()
    # Convert RGBA/P to RGB before JPEG encoding (JPEG has no alpha channel)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buffer, format='JPEG', quality=90, optimize=False)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def prepare_region_images(image_path: str, layout_regions: List[LayoutRegion]) -> Dict[int, Any]:
    """Crop all layout regions and store with base64 encoding."""
    pil_image = Image.open(image_path)
    region_images = {}
    for region in layout_regions:
        cropped = crop_region(pil_image, region.bbox)
        region_images[region.region_id] = {
            'image':  cropped,
            'base64': image_to_base64(cropped),
            'type':   region.region_type,
            'bbox':   region.bbox,
        }
    print(f"Cropped {len(region_images)} regions")
    return region_images
