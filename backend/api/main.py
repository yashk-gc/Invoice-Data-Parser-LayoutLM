from fastapi import FastAPI, UploadFile, File
import os
import uuid

from backend.services.ocr_service import run_tesseract
from backend.data.prepare_dataset import convert_ocr_to_layoutlm
from backend.inference.run_inference import extract_invoice_fields

app = FastAPI()


@app.post("/extract-invoice")
async def extract_invoice(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex

    image_path = f"data/raw/{uid}.jpg"
    ocr_json = f"data/ocr/{uid}.json"
    infer_json = "data/outputs/layoutlm_infer.json"

    with open(image_path, "wb") as f:
        f.write(await file.read())

    run_tesseract(image_path, ocr_json)

    convert_ocr_to_layoutlm(
        ocr_json=ocr_json,
        image_path=image_path,
        output_json=infer_json,
    )

    fields = extract_invoice_fields(image_path)

    return {
        "filename": file.filename,
        "extracted_fields": fields,
    }
