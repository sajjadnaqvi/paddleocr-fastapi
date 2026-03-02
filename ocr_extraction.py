"""
Stage 1: Text Extraction with PaddleOCR + LayoutLM Reading Order
"""
import compat  # must be first — patches langchain.docstore for paddlex compatibility
import torch
from dataclasses import dataclass
from typing import List
from paddleocr import PaddleOCR
from transformers import LayoutLMv3ForTokenClassification
from layoutreader.v3.helpers import prepare_inputs, boxes2inputs, parse_logits

# --- Data Class ---

@dataclass
class OCRRegion:
    text: str
    bbox: list  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    confidence: float

    @property
    def bbox_xyxy(self):
        """Return bbox as [x1, y1, x2, y2] format."""
        x_coords = [p[0] for p in self.bbox]
        y_coords = [p[1] for p in self.bbox]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


# --- Singletons (loaded once, reused across all requests) ---

_ocr_engine = None
_layout_model = None


def _get_ocr_engine() -> PaddleOCR:
    global _ocr_engine
    if _ocr_engine is None:
        print("Loading PaddleOCR engine...")
        _ocr_engine = PaddleOCR(lang='en')
        print("PaddleOCR ready.")
    return _ocr_engine


def _get_layout_model() -> LayoutLMv3ForTokenClassification:
    global _layout_model
    if _layout_model is None:
        print("Loading LayoutReader model...")
        _layout_model = LayoutLMv3ForTokenClassification.from_pretrained("hantian/layoutreader")
        _layout_model.eval()   # disables dropout — required for inference
        print("LayoutReader ready.")
    return _layout_model


def warmup_models() -> None:
    """Pre-load both models into memory. Call once at server startup
    so the first real request pays no loading cost."""
    _get_ocr_engine()
    _get_layout_model()


# --- OCR ---

def run_ocr(image_path: str) -> List[OCRRegion]:
    """Run PaddleOCR on image and return structured OCRRegion list."""
    ocr = _get_ocr_engine()   # reuses the already-loaded engine
    result = ocr.predict(image_path)
    page = result[0]

    texts = page['rec_texts']
    scores = page['rec_scores']
    boxes = page['rec_polys']

    regions: List[OCRRegion] = []
    for text, score, box in zip(texts, scores, boxes):
        regions.append(OCRRegion(
            text=text,
            bbox=box.astype(int).tolist(),
            confidence=score
        ))

    print(f"Extracted {len(regions)} text regions")
    return regions


# --- Reading Order ---

def get_reading_order(ocr_regions: List[OCRRegion]) -> List[int]:
    """Use LayoutReader to determine reading order of OCR regions."""
    model = _get_layout_model()

    # 1. Compute bbox_xyxy once per region (avoid recalculating per loop)
    xyxy_list = [r.bbox_xyxy for r in ocr_regions]

    # 2. Calculate image size with 10% padding (single pass)
    max_x = max(coords[2] for coords in xyxy_list)
    max_y = max(coords[3] for coords in xyxy_list)
    image_width  = max_x * 1.1
    image_height = max_y * 1.1

    # 3. Normalize to 0-1000 range for LayoutLM
    scale_x = 1000.0 / image_width
    scale_y = 1000.0 / image_height
    boxes = [
        [
            int(x1 * scale_x), int(y1 * scale_y),
            int(x2 * scale_x), int(y2 * scale_y),
        ]
        for x1, y1, x2, y2 in xyxy_list
    ]

    # 4. Prepare inputs and run inference
    inputs = boxes2inputs(boxes)
    inputs = prepare_inputs(inputs, model)

    with torch.no_grad():   # ~30% faster — no gradient tracking needed for inference
        logits = model(**inputs).logits.cpu().squeeze(0)

    return parse_logits(logits, len(boxes))


def get_ordered_text(ocr_regions: List[OCRRegion], reading_order: List[int]) -> List[dict]:
    """Return OCR regions sorted by reading order."""
    indexed = [(reading_order[i], i, ocr_regions[i]) for i in range(len(ocr_regions))]
    indexed.sort(key=lambda x: x[0])

    return [
        {
            "position": pos,
            "text": region.text,
            "confidence": region.confidence,
            "bbox": region.bbox_xyxy,
        }
        for pos, _, region in indexed
    ]
