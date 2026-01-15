import os
import json
import cv2
import pytesseract
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

RAW_DIR = "data/raw"
OCR_DIR = "data/ocr"
os.makedirs(OCR_DIR, exist_ok=True)


def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    paths = []
    for i, img in enumerate(images):
        out = pdf_path.replace(".pdf", f"_page_{i}.png")
        img.save(out)
        paths.append(out)
    return paths


def run_ocr(image_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    tokens, bboxes = [], []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text:
            x = data["left"][i]
            y = data["top"][i]
            bw = data["width"][i]
            bh = data["height"][i]
            tokens.append(text)
            bboxes.append([x, y, x + bw, y + bh])

    return {
        "tokens": tokens,
        "bboxes": bboxes,
        "width": w,
        "height": h,
        "image_path": image_path
    }


def process():
    for f in os.listdir(RAW_DIR):
        path = os.path.join(RAW_DIR, f)
        images = pdf_to_images(path) if f.endswith(".pdf") else [path]

        for img in images:
            data = run_ocr(img)
            out = os.path.join(OCR_DIR, os.path.basename(img) + ".json")
            with open(out, "w") as fp:
                json.dump(data, fp, indent=2)
            print("OCR saved ->", out)


if __name__ == "__main__":
    process()
