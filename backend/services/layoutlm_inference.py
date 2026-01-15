import json
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image

MODEL_NAME = "microsoft/layoutlmv3-base"

processor = LayoutLMv3Processor.from_pretrained(
    MODEL_NAME,
    apply_ocr=False
)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2  # placeholder (will change after fine-tuning)
)
model.eval()


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]


def run_layoutlm(ocr_json_path):
    with open(ocr_json_path, "r") as f:
        data = json.load(f)

    image = Image.open(data["image_path"]).convert("RGB")

    words = data["tokens"]
    boxes = [
        normalize_bbox(b, data["width"], data["height"])
        for b in data["bboxes"]
    ]

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

    return list(zip(words, predictions))


if __name__ == "__main__":
    result = run_layoutlm("data/ocr/sample_page_0.jpg.json")
    print(result)
