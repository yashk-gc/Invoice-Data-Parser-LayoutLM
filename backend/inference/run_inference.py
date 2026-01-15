import json
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

from backend.config.labels import ID2LABEL
from backend.services.ocr_service import run_tesseract
from backend.utils.bbox_utils import normalize_bbox
from backend.utils.heuristics import heuristic_extract


MODEL_PATH = "models/layoutlm_invoice"


def extract_invoice_fields(image_path: str):
    ocr_json = image_path.replace(".jpg", ".json").replace(".png", ".json")

    # 1️⃣ OCR
    run_tesseract(image_path, ocr_json)

    with open(ocr_json, "r", encoding="utf-8") as f:
        ocr = json.load(f)

    words = ocr["words"]
    raw_boxes = ocr["bboxes"]

    image = Image.open(image_path)
    width, height = image.size

    boxes = [
        normalize_bbox(b, width, height) for b in raw_boxes
    ]

    # 2️⃣ LayoutLM
    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base",
        apply_ocr=False
    )

    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()

    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length"
    )

    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    labels = [ID2LABEL[p] for p in predictions]

    extracted = {}
    for word, label in zip(words, labels):
        if label != "O":
            extracted.setdefault(label, []).append(word)

    extracted = {k: " ".join(v) for k, v in extracted.items()}

    # 3️⃣ FALLBACK (ALWAYS WORKS)
    if not extracted:
        extracted = heuristic_extract(" ".join(words))

    return extracted
