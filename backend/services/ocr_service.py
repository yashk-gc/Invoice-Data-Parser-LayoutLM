import json
import pytesseract
from PIL import Image


def run_tesseract(image_path: str, output_json: str):
    image = Image.open(image_path)

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT
    )

    words = []
    bboxes = []

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text:
            words.append(text)
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            bboxes.append([x, y, x + w, y + h])

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "words": words,
                "bboxes": bboxes
            },
            f,
            indent=2
        )
