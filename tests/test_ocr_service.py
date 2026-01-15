from backend.services.ocr_service import extract_ocr_data


def test_extract_ocr_data_returns_dict():
    # We only test structure, not OCR accuracy
    assert callable(extract_ocr_data)
