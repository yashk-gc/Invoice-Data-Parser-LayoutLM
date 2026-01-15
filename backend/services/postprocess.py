import re

def heuristic_fallback(text: str) -> dict:
    result = {}

    invoice_no = re.search(r"(invoice\s*(no|number)?[:\s]*)(\d+)", text, re.I)
    date = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", text)
    total = re.search(r"(total[:\s]*)([\d,]+\.\d{2})", text, re.I)

    if invoice_no:
        result["INVOICE_NO"] = invoice_no.group(3)
    if date:
        result["DATE"] = date.group(1)
    if total:
        result["TOTAL_AMOUNT"] = total.group(2)

    return result
